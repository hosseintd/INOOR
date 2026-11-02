import os
import sys
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval

# Import exposure time utilities
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)
from utils.exptime_utils import get_exptime_seconds, format_exptime as format_exptime_util

def get_exptime(header):
    """
    Extract exposure time (seconds) from header.
    Handles 'EXPTIME', 'EXP_TIME', and standard high-value 10us conversion if detected.
    
    DEPRECATED: Use utils.exptime_utils.get_exptime_seconds() instead.
    Kept for backward compatibility.
    """
    return get_exptime_seconds(header)

def format_exptime(header):
    """
    Extract and format exposure time from header with appropriate units.
    For short exposures (< 1s / 100000 * 10us), returns in microseconds.
    For long exposures, returns in seconds.
    
    DEPRECATED: Use utils.exptime_utils.format_exptime() instead.
    Kept for backward compatibility.
    
    Returns: formatted string like "30µs", "1.5s", etc.
    """
    return format_exptime_util(header)

def load_fits(file_path):
    """Load a FITS file and return the primary data as float32."""
    with fits.open(file_path, memmap=False) as hdul:
        data = hdul[0].data.astype('float32')
    return data

def sigclip(img, lower_sigma, upper_sigma):
    """Apply sigma clipping to a single image and fill clipped pixels with the image median."""
    clipped = sigma_clip(img, sigma_lower=lower_sigma, sigma_upper=upper_sigma, maxiters=None)
    filled = clipped.filled(fill_value=np.median(img))
    return np.array(filled, dtype='float32')

def analyze_sigma_bounds(sample_frame, low_percentile=0.5, high_percentile=99.5):
    """
    Estimate lower and upper sigma clipping values using robust stats on a sample frame.
    Returns (lower_sigma, upper_sigma).
    """
    arr = np.array(sample_frame).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 3.0, 3.0

    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    sigma_est = 1.4826 * mad if mad > 0 else np.std(arr) if np.std(arr) > 0 else 1.0

    low_val = np.percentile(arr, low_percentile)
    high_val = np.percentile(arr, high_percentile)

    lower_sigma = (median - low_val) / sigma_est if sigma_est else 3.0
    upper_sigma = (high_val - median) / sigma_est if sigma_est else 3.0

    lower_sigma = float(np.clip(lower_sigma, 0.5, 20.0))
    upper_sigma = float(np.clip(upper_sigma, 0.5, 20.0))

    return lower_sigma, upper_sigma

def zscale_limits(img):
    """Return vmin, vmax determined by ZScaleInterval for display."""
    interval = ZScaleInterval()
    try:
        vmin, vmax = interval.get_limits(img)
    except Exception:
        vmin, vmax = np.min(img), np.max(img)
    return vmin, vmax

from datetime import datetime, timezone

def apply_2d_sigma_clip(data, sigma_lower, sigma_upper, kernel_size, auto_sigma=False,
                        maxiters=3):
    """
    Apply sigma clipping to a 2D image block-by-block (non-overlapping ks x ks
    tiles). Outliers in each block are replaced by that block's median.

    In auto_sigma mode:
    - sigma_lower is ignored (set to very high).
    - sigma_upper is used to find outliers.

    Implementation note: this used to call astropy's sigma_clip once per block
    in a Python double loop -- ~670k calls for a 4096x4096 frame, ~75 s per
    master. It is now vectorised over all tiles at once (~4.6 s), producing
    bit-identical output; see tests/test_master_stacking.py.

    WHAT THIS COSTS, measured on a real 6-frame master bias (kernel 35,
    auto, upper 3.0). It replaced 30,922 of 16.7M pixels (0.184%), and on
    pure Gaussian noise of the same width it replaced 0.144% -- so roughly
    four fifths of what it "rejects" is the noise tail, not a defect. Worse,
    it erases stable detector structure: three pixels at the end of row 4095
    that sit at 1400-1900 ADU in ALL SIX frames were flattened to the block
    median of 98 ADU, which leaves a +1486 ADU spike in every science frame
    the master is subtracted from. A master bias/dark must KEEP its hot
    pixels; that is what it is for. Hence the default is off.
    """
    ny, nx = data.shape
    ks = max(3, int(kernel_size))

    # In auto mode we don't clip low values: a very high lower threshold
    # effectively disables that side.
    slo = 100.0 if auto_sigma else float(sigma_lower)
    shi = float(sigma_upper)

    # Pad up to a whole number of tiles with NaN, then fold into
    # (n_block_y, n_block_x, ks*ks) so every block is one row of pixels.
    by = -(-ny // ks)
    bx = -(-nx // ks)
    pad = np.full((by * ks, bx * ks), np.nan, dtype='float64')
    pad[:ny, :nx] = data
    tiles = pad.reshape(by, ks, bx, ks).transpose(0, 2, 1, 3).reshape(by, bx, ks * ks)

    # Replacement value is the median of the ORIGINAL block (outliers included),
    # matching the previous behaviour.
    orig_median = np.nanmedian(tiles, axis=2)

    # Non-finite pixels start out masked, exactly as astropy's sigma_clip does.
    masked = ~np.isfinite(tiles)
    with np.errstate(invalid='ignore'):
        for _ in range(maxiters):
            work = np.where(masked, np.nan, tiles)
            cen = np.nanmedian(work, axis=2, keepdims=True)
            std = np.nanstd(work, axis=2, keepdims=True)
            new_masked = (~np.isfinite(tiles)) | (tiles < cen - slo * std) | (tiles > cen + shi * std)
            if np.array_equal(new_masked, masked):
                break
            masked = new_masked

        filled = np.where(masked, orig_median[:, :, None], tiles)

    out = filled.reshape(by, bx, ks, ks).transpose(0, 2, 1, 3).reshape(by * ks, bx * ks)
    return np.asarray(out[:ny, :nx], dtype=data.dtype)

def create_master(output_path, file_list, method='median', do_sigma_clip=False,
                  sigma_lower=3.0, sigma_upper=3.0, exclude_indices=None,
                  progress_callback=None, hdr_extra=None,
                  kernel=5, auto_sigma=False, frame_transform=None,
                  scale_to_common_level=False):
    """
    Create a master frame from file_list and write to output_path.

    1. Load frames (applying frame_transform to each, e.g. binning to a
       common shape, if provided).
    2. Optionally scale every frame to a common level (flats only).
    3. Stack frames using method (NaN-safe).
    4. Apply 2D Post-filter Sigma Clipping if requested.

    This is the SINGLE canonical stacking path: all master bias/dark/flat
    creation (with or without pre-binning) must go through here so the
    outlier-rejection algorithm never silently depends on frame shape.
    """
    if exclude_indices is None:
        exclude_indices = set()
    else:
        exclude_indices = set(exclude_indices)

    files = [f for i, f in enumerate(file_list) if i not in exclude_indices]
    n = len(files)
    if n == 0:
        raise ValueError("No frames selected for stacking.")

    if progress_callback:
        progress_callback(0.0, f"Loading {n} frames...")

    # --- Pre-flight memory estimate ---
    # The whole stack is held in RAM as float32 (plus ~2x transient during
    # np.dstack). Warn before starting a combination that will thrash/crash.
    try:
        hdr0 = fits.getheader(files[0])
        est_bytes = 2.0 * n * hdr0.get('NAXIS1', 0) * hdr0.get('NAXIS2', 0) * 4
        try:
            import psutil
            avail = psutil.virtual_memory().available
        except ImportError:
            avail = 4 * 1024**3  # conservative fallback threshold
        if est_bytes > 0.8 * avail:
            msg = (f"Warning: stacking {n} frames needs ~{est_bytes / 1024**3:.1f} GB RAM "
                   f"(~{avail / 1024**3:.1f} GB available). The combination may be very "
                   f"slow or fail; consider excluding frames or binning first.")
            print(f"[MASTER] {msg}")
            if progress_callback:
                progress_callback(0.0, msg)
    except Exception:
        pass

    # --- Pre-flight shape check ---
    # np.dstack below raises on a shape mismatch, but only after every frame
    # has been read into RAM, and its message names an array index rather
    # than a file. Reading NAXIS from the headers costs milliseconds and can
    # say which file is the odd one out. Skipped when frame_transform is set,
    # since that exists precisely to bring frames to a common shape.
    if frame_transform is None:
        shapes = {}
        for fpath in files:
            h = fits.getheader(fpath)
            shapes.setdefault((h.get('NAXIS2'), h.get('NAXIS1')), []).append(fpath)
        if len(shapes) > 1:
            majority = max(shapes, key=lambda s: len(shapes[s]))
            lines = []
            for shp, fl in sorted(shapes.items(), key=lambda kv: -len(kv[1])):
                names = ", ".join(os.path.basename(f) for f in fl[:3])
                more = f" (+{len(fl) - 3} more)" if len(fl) > 3 else ""
                lines.append(f"  {shp[1]}x{shp[0]}: {len(fl)} frame(s) - {names}{more}")
            raise ValueError(
                "Cannot combine frames of different sizes. Most frames are "
                f"{majority[1]}x{majority[0]}:\n" + "\n".join(lines) +
                "\nRemove the odd frames, or use the Calibration tab, which can "
                "bin frames to a common shape.")

    # Load all data into a 3D stack
    data_list = []
    base_header = None

    for idx, fpath in enumerate(files):
        img = load_fits(fpath)
        if frame_transform is not None:
            img = frame_transform(img)
        data_list.append(img)
        
        # Take header from the first frame as template
        if base_header is None:
            try:
                with fits.open(fpath) as hdul:
                    base_header = hdul[0].header.copy()
            except:
                base_header = fits.Header()
                
        if progress_callback:
            progress_callback((idx + 1) / (n * 2.0), f"Loaded {idx+1}/{n}")

    # Sky flats fade while the sequence is being taken, and combining them
    # without putting them on a common level throws away most of the benefit
    # of taking several: the per-pixel median simply picks whichever frame
    # sits in the middle, so the result carries ONE frame's noise. Measured on
    # 5 real g-band twilight flats (medians 3046 -> 2668 ADU, a 13.4% fade),
    # high-pass noise in a central 1000x1000 patch was
    #     one frame 57.4 ADU | unscaled median 50.3 | level-scaled median 33.4
    # i.e. unscaled, five flats bought about as much as 1.3 flats.
    #
    # This is WRONG for bias and dark frames, whose absolute level IS the
    # signal, so it is opt-in and always recorded in the header.
    scale_ref = None
    if scale_to_common_level:
        if progress_callback:
            progress_callback(0.5, "Scaling frames to a common level...")
        levels = [float(np.nanmedian(img)) for img in data_list]
        for lvl, fpath in zip(levels, files):
            if not np.isfinite(lvl) or lvl <= 0:
                raise ValueError(
                    f"Cannot scale {os.path.basename(fpath)} to a common level: "
                    f"its median is {lvl!r}. Level scaling is meant for flats; "
                    f"switch it off for frames that may legitimately sit at or "
                    f"below zero.")
        scale_ref = float(np.median(levels))
        for i, lvl in enumerate(levels):
            data_list[i] *= np.float32(scale_ref / lvl)

    if progress_callback:
        progress_callback(0.5, "Stacking frames...")

    # Build 3D stack: Shape (NY, NX, N)
    stack = np.dstack(data_list)
    del data_list # Free memory

    # 1. Perform Stacking.
    # np.nanmedian/nanmean are ~3x slower than their plain counterparts; most
    # real stacks contain no NaN at all, so pay for the check (one cheap pass)
    # instead of the slow path. Results are identical when no NaN is present.
    # isfinite, not isnan: a single +/-inf slipped past the isnan check, took
    # the fast np.mean path and poisoned the whole mean master. (Median
    # survived it; mean did not.)
    has_nan = not bool(np.isfinite(stack).all())
    if has_nan:
        # nanmedian/nanmean drop NaN but NOT infinities: np.nanmean([1, inf, 1])
        # is inf. Map every non-finite value to NaN so the reduction actually
        # ignores it, instead of one bad pixel poisoning the whole master.
        stack = np.where(np.isfinite(stack), stack, np.nan)
    if method == 'median':
        master = np.nanmedian(stack, axis=2) if has_nan else np.median(stack, axis=2)
    else:
        master = np.nanmean(stack, axis=2) if has_nan else np.mean(stack, axis=2)

    del stack # Free memory

    # 2. Apply 2D Sigma Clipping Post-filter if requested
    n_replaced = None
    if do_sigma_clip:
        mode_str = "Auto" if auto_sigma else "Manual"
        if progress_callback:
            progress_callback(0.7, f"Applying 2D Sigma Clip ({mode_str}, Kernel: {kernel})...")
        before = master
        master = apply_2d_sigma_clip(master, sigma_lower, sigma_upper, kernel, auto_sigma=auto_sigma)
        # How much of the master this rewrote is otherwise invisible: the
        # filter silently replaces real hot pixels with the local median.
        n_replaced = int(np.count_nonzero(master != before))
        del before

    # ensure float32
    master = np.array(master, dtype='float32')
    
    # --- Prepare Header ---
    if base_header is None: base_header = fits.Header()
    
    # Update standard metadata
    base_header['STK_METH'] = (method.upper(), 'Stacking Method used')
    base_header['STK_NUM'] = (n, 'Number of frames combined')
    base_header['STK_SCAL'] = (bool(scale_to_common_level),
                               'Frames scaled to a common level before stacking')
    if scale_ref is not None:
        base_header['STK_SREF'] = (scale_ref,
                                   'Common level frames were scaled to (ADU)')
    base_header['STK_CLIP'] = (do_sigma_clip, '2D Sigma clipping applied?')
    if do_sigma_clip:
        base_header['STK_CMOD'] = ("AUTO" if auto_sigma else "MANUAL", 'Sigma clipping mode')
        if not auto_sigma:
            base_header['STK_SLO'] = (sigma_lower, 'Lower sigma value')
        base_header['STK_SHI'] = (sigma_upper, 'Upper sigma value')
        base_header['STK_KERN'] = (kernel, 'Kernel size for clipping')
        if n_replaced is not None:
            base_header['STK_NREP'] = (n_replaced,
                                       'Pixels replaced by the local block median')
    
    # UTC, like every other date INOOR writes. This used to be local time
    # with no zone marker. Reading a local time as UT is the bug that put the
    # GRB target below the horizon; see the DATE-OBS handling in
    # core/ext_coeff.py.
    base_header['CRE_TIME'] = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        'UT time of creation')
    
    # Add list of files to HISTORY - FIX: Remove any potential non-ASCII or Control Chars
    base_header.add_history("Combined frames:")
    for f in files:
        fname = os.path.basename(f)
        # Clean non-printable characters
        clean_name = "".join(filter(lambda x: x.isprintable(), fname))
        base_header.add_history(f" - {clean_name}")

    if hdr_extra:
        for k, v in hdr_extra.items():
            if isinstance(v, str):
                v = "".join(filter(lambda x: x.isprintable(), v))
            base_header[k] = v

    if progress_callback:
        progress_callback(0.9, "Processing output...")

    if output_path:
        # float32: a master is a COMBINED frame, not detector output. Callers
        # that want the 16-bit ADU file on disk go through
        # calibration.save_fits_16bit, which writes BSCALE/BZERO correctly.
        hdu = fits.PrimaryHDU(np.asarray(master, dtype="float32"), header=base_header)
        hdu.writeto(output_path, overwrite=True)
        
        if progress_callback:
            progress_callback(1.0, "Done")
        return output_path
    else:
        # If no path provided, return the data and header for review
        if progress_callback:
            progress_callback(1.0, "Ready for review")
        return master, base_header
