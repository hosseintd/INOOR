# calibration.py  -- updated
import os
import tempfile
import numpy as np
from astropy.io import fits
from . import masterFrame_creator as mfc

import re

# New imports for auto-flat creation
from astropy.stats import sigma_clipped_stats
from astropy.convolution import interpolate_replace_nans, Kernel
import scipy.ndimage as nd

from utils.logging_config import get_logger

log = get_logger(__name__)

# ---------------- existing calibrate functions (kept, slightly refactored) ----------------

def subtract_pedestal(light_frame, master_dark, master_bias):
    """Remove the additive offset from a light frame.

    A master dark ALREADY contains the bias pedestal, so when a dark is
    available subtracting it removes bias and dark current in one step; the
    bias frame is then only needed for exposure-time scaling of the dark
    (done by the caller). With no dark, the bias is the best pedestal we have.

    The previous formulation was
        lf - mb - clip(md - mb, min=0)
    which is algebraically lf - md EXCEPT where md < mb. There the clip left
    the pixel at lf - mb, truncating the negative half of the dark-frame noise
    and leaving a systematic positive offset in the background. It also clipped
    the result itself at zero, which biases the sky estimate upward and breaks
    the noise model -- the negative wing of the sky distribution is real data.
    """
    lf = np.asarray(light_frame, dtype='float32')

    if master_dark is not None:
        return lf - np.asarray(master_dark, dtype='float32')
    if master_bias is not None:
        return lf - np.asarray(master_bias, dtype='float32')
    return lf.copy()


def prepare_flat_frame(frame, pedestal=None, scale_to_unity=True):
    """Make one raw flat frame ready for combination.

    Two steps, and the ORDER matters:

    1. Remove the additive pedestal. A raw flat is  bias + dark_f + F*I, so the
       bias structure -- on the KL4040 the quadrant-seam '+' -- is still inside
       it. Dividing a science frame by a flat that still contains the pedestal
       re-imprints that structure INVERTED, with amplitude
           bias_step / flat_level
       which is why the seam survives flat-fielding even though dark
       subtraction removed it from the light frame.

    2. Scale to unity by the frame's own median. Twilight and dome flats drift
       in level between exposures; median-combining them unscaled weights the
       brightest frames. Scaling by the inverse median first is what IRAF
       flatcombine (scale=mode) and ccdproc (scale=inv_median) do.

    Scaling must come AFTER pedestal removal -- scaling a frame that still
    carries the bias would scale the bias too and make the seam worse.
    """
    f = np.asarray(frame, dtype='float32')

    if pedestal is not None:
        f = f - np.asarray(pedestal, dtype='float32')

    if scale_to_unity:
        level = np.median(f)
        if np.isfinite(level) and level > 0:
            f = f / level
        else:
            log.warning("Flat frame has non-positive median (%.4g) after pedestal "
                        "removal; leaving it unscaled.", level)
    return f


MIN_FLAT_RESPONSE = 0.2


def normalize_flat(master_flat, min_response=MIN_FLAT_RESPONSE):
    """Return the master flat divided by its median (robust against hot pixels).

    Pixels whose normalised response falls below `min_response` are treated as
    dead and their divisor is set to 1.0, leaving the science pixel
    uncorrected rather than amplified.

    This guard is essential, not cosmetic. The previous code clamped the
    divisor at eps=1e-6, so a dead pixel multiplied the science frame by up to
    1e6 and produced values of +/-1e8 that wrecked the frame statistics and
    the display scaling. It only became visible once the flat's pedestal was
    subtracted, which pushes genuinely dead pixels slightly negative; before
    that they sat near 0.03 and "merely" amplified by ~30x.

    0.2 is far below real vignetting (~0.7-0.9) and dust donuts (~0.5), so it
    only ever catches dead pixels.
    """
    mf = np.asarray(master_flat, dtype='float32')

    ref_val = np.median(mf)
    if not np.isfinite(ref_val) or ref_val <= 0:
        # Fallback to mean if median is bad
        ref_val = np.mean(mf)

    if not np.isfinite(ref_val) or ref_val <= 0:
        log.warning("Master flat has no usable scale (median and mean are both "
                    "non-positive); skipping flat division.")
        return np.ones_like(mf)

    norm_flat = mf / ref_val

    dead = ~np.isfinite(norm_flat) | (norm_flat < min_response)
    n_dead = int(dead.sum())
    if n_dead:
        norm_flat[dead] = 1.0
        log.warning("Master flat: %d pixel(s) below %.2f of the median response "
                    "treated as dead; left uncorrected instead of amplified.",
                    n_dead, min_response)
    return norm_flat


def calibrate(light_frame, master_dark, master_bias, master_flat):
    calibrated_frame = subtract_pedestal(light_frame, master_dark, master_bias)
    calibrated_frame = calibrated_frame / normalize_flat(master_flat)
    return np.array(calibrated_frame, dtype='float32')

# ---------------- master creation helper (kept) ----------------

def create_master_from_list(file_list, method='median', do_sigma_clip=False,
                            lower_sigma=3.0, upper_sigma=3.0,
                            exclude_indices=None, progress_callback=None,
                            frame_transform=None):
    """
    Combine a list of FITS files into a master frame via the canonical
    stacking path (masterFrame_creator.create_master). If frame_transform is
    given it is applied to every frame after loading (e.g. binning to a
    common shape) BEFORE stacking, so the combination/clipping algorithm is
    identical whether or not frames needed resizing.
    """
    if not file_list:
        raise ValueError("Empty file list for master creation.")

    if len(file_list) == 1:
        if progress_callback:
            progress_callback(0.0, "Loading 1 frame...")
        arr = mfc.load_fits(file_list[0])
        if frame_transform is not None:
            arr = frame_transform(arr)
        if progress_callback:
            progress_callback(1.0, "Done")
        return arr

    tmp = tempfile.NamedTemporaryFile(suffix='.fits', delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        mfc.create_master(tmp_path, file_list,
                          method=method,
                          do_sigma_clip=do_sigma_clip,
                          sigma_lower=lower_sigma,
                          sigma_upper=upper_sigma,
                          exclude_indices=exclude_indices or set(),
                          progress_callback=progress_callback,
                          frame_transform=frame_transform)
        arr = mfc.load_fits(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return arr

import re

def _parse_dateobs(dateobs_raw):
    """
    Robust parse for strings like:
      '2025-10-15T19:45:00.282 NOGPS'
      "2025-10-15 19:45:00.282"
      b'2025-10-15T19:45:00.282 NOGPS'
    Returns a datetime.datetime or None.
    """
    # local import of the class to avoid name-shadow bugs
    from datetime import datetime as _datetime

    if dateobs_raw is None:
        return None

    # decode bytes safe
    if isinstance(dateobs_raw, bytes):
        try:
            s = dateobs_raw.decode('utf-8', errors='ignore')
        except Exception:
            s = str(dateobs_raw)
    else:
        s = str(dateobs_raw)

    s = s.strip()
    if not s:
        return None

    # take first whitespace-separated token (removes trailing "NOGPS" etc.)
    token = s.split()[0]
    # strip surrounding quotes if any
    token = token.strip("'\"")

    # regex to capture ISO-like timestamp components
    m = re.match(
        r'^(\d{4})-(\d{2})-(\d{2})[T ]'    # date + T or space
        r'(\d{2}):(\d{2}):(\d{2})'         # HH:MM:SS
        r'(?:\.(\d{1,6}))?'                # optional .fraction (1..6 digits)
        r'$',
        token
    )

    if not m:
        # not matched exactly — try a looser match (timestamp at start of token)
        m2 = re.match(
            r'^(\d{4})-(\d{2})-(\d{2})[T ]'
            r'(\d{2}):(\d{2}):(\d{2})'
            r'(?:\.(\d+))?',
            token
        )
        if not m2:
            return None
        parts = m2.groups()
    else:
        parts = m.groups()

    try:
        year = int(parts[0]); month = int(parts[1]); day = int(parts[2])
        hour = int(parts[3]); minute = int(parts[4]); second = int(parts[5])
        frac = parts[6] or '0'
        # pad or truncate to microseconds (6 digits)
        micro = int((frac + '0'*6)[:6])
        return _datetime(year, month, day, hour, minute, second, micro)
    except Exception:
        return None


def _make_short_timestamp(dt, include_ms=False):
    """Return short timestamp string for filenames. Example: '20251015_194500' or with ms '20251015_194500_282'."""
    if dt is None:
        return None
    if include_ms and getattr(dt, "microsecond", 0):
        ms = int(dt.microsecond / 1000)  # micro -> ms
        return dt.strftime("%Y%m%d_%H%M%S") + f"_{ms:03d}"
    else:
        return dt.strftime("%Y%m%d_%H%M%S")

UINT16_MAX = 65535

# Integer-ADU storage rounds to whole ADU, so it is only meaningful for data
# that actually sits on the detector's ADU scale. This camera's pedestal alone
# is ~90 ADU, so anything peaking below this is not ADU data -- it is a
# normalised flat (~1.0) or a ratio image, and rounding it to integers would
# flatten it to a constant.
MIN_ADU_MAX = 32.0


def _require_adu_scale(arr, path):
    """Raise unless `arr` is on a raw ADU scale that survives integer storage."""
    a = np.asarray(arr, dtype='float64')
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        raise ValueError(
            f"Refusing to save {os.path.basename(path)}: the image has no finite pixels.")
    dmax = float(finite.max())
    if dmax < MIN_ADU_MAX:
        raise ValueError(
            f"Refusing to save {os.path.basename(path)} as integer ADU: the data "
            f"maximum is {dmax:.4g}, far below the detector's ~90 ADU pedestal. "
            f"This looks like normalised data (a flat scaled to ~1.0 or a ratio "
            f"image); rounding it to whole ADU would destroy it. "
            f"Save such an image as float32 instead.")


def _to_uint16_adu(arr):
    """Round a calibrated float image to integer ADU stored as uint16.

    This deliberately reproduces the EXACT storage convention of the raw FLI
    Kepler frames: BITPIX=16 with BSCALE=1 and BZERO=32768, which astropy
    emits automatically for a uint16 array. Do not hand-roll a different
    BSCALE/BZERO here.

    An earlier attempt mapped the full float range onto signed int16 with a
    computed BSCALE/BZERO. That is legal FITS and round-trips perfectly in
    astropy, but ImageJ -- and therefore AstroImageJ -- has no signed 16-bit
    pixel type: it reads BITPIX=16 as UNSIGNED, so a sky of 65 ADU stored as a
    small negative integer displayed as ~59000, and a whole night of frames
    looked like noise. Matching the raw convention is worth more than the extra
    sub-ADU precision that scaling bought.

    Quantisation to whole ADU costs 1/sqrt(12) = 0.29 ADU rms against a read
    noise of ~3.4 ADU: the noise grows by 0.4%, which is not measurable.

    Returns (stored_uint16, n_clipped_low, n_clipped_high, n_nonfinite).
    """
    a = np.asarray(arr, dtype='float64')
    finite = np.isfinite(a)
    n_bad = int(a.size - finite.sum())

    rounded = np.rint(np.where(finite, a, 0.0))
    n_lo = int((rounded < 0).sum())
    n_hi = int((rounded > UINT16_MAX).sum())
    np.clip(rounded, 0, UINT16_MAX, out=rounded)

    return rounded.astype(np.uint16), n_lo, n_hi, n_bad

import os
import numpy as np
from astropy.io import fits

def _unique_filename(path):
    """If path exists, add _1, _2 ... before extension to avoid overwrite."""
    base, ext = os.path.splitext(path)
    i = 1
    candidate = path
    while os.path.exists(candidate):
        candidate = f"{base}_{i}{ext}"
        i += 1
    return candidate

def save_fits_16bit(path, data, header=None, saturate_level=4095, overwrite=False,
                    include_ms=False, add_timestamp=True, stats_out=None):
    """
    Save a calibrated float image as integer ADU in a 16-bit FITS file, using
    the SAME storage convention as the raw FLI Kepler frames: BITPIX=16 with
    BSCALE=1 and BZERO=32768 (unsigned 16-bit), which astropy emits for a
    uint16 array.

    What this fixes versus the original writer: the ceiling is 65535, not 4095.
    After flat division a bright star in a vignetted corner (where norm_flat<1
    scales counts UP) legitimately exceeds the raw 4095 ADC ceiling, and the
    old clamp marked it saturated when the detector never saturated. The raw
    full scale is preserved separately in the SATURATE keyword so photometry
    can still flag genuinely saturated pixels.

    What this deliberately does NOT do: invent a custom BSCALE/BZERO to carry
    the exact float range. That is valid FITS and round-trips in astropy, but
    ImageJ -- and therefore AstroImageJ -- has no signed 16-bit pixel type and
    reads BITPIX=16 as unsigned, so a sky of 65 ADU stored as a small negative
    integer showed up as ~59000. Interoperability with the tools actually used
    for the analysis outweighs sub-ADU precision.

    Remaining limitation: values below 0 cannot be represented and are clamped,
    with the count recorded in the header and returned by the logs. For INO340
    data the sky sits many sigma above zero, so only dead/bad pixels clamp.

    Args:
      path: desired output path (can include directory and extension)
      data: ndarray to save
      header: astropy Header or mapping (optional)
      saturate_level: raw detector full-scale in ADU, recorded as SATURATE so
                      downstream photometry can still flag truly saturated
                      pixels. None to omit.
      overwrite: whether to overwrite existing file
      include_ms: include milliseconds in suffix if available
      add_timestamp: append a DATE-OBS-derived suffix to the filename. Right for
                     calibrated science frames, wrong for a master whose name the
                     user typed into a Save dialog.
      stats_out: optional dict, filled with 'clamped_low', 'clamped_high' and
                 'nonfinite'. The header and the log already record these, but
                 neither reaches the operator, who sees only "Calibration
                 complete" while thousands of pixels per frame become 0.
    Returns:
      the path actually written
    """
    _require_adu_scale(data, path)
    out, n_lo, n_hi, n_bad = _to_uint16_adu(data)

    arr = np.asarray(data, dtype='float64')
    finite = np.isfinite(arr)

    # build header/hdu
    if header is None:
        hdr = fits.Header()
    else:
        hdr = fits.Header(header) if not isinstance(header, fits.Header) else header.copy()
    # Let astropy derive these from the uint16 array; a stale BSCALE/BZERO
    # inherited from the source header would fight with it.
    for key in ('BSCALE', 'BZERO'):
        if key in hdr:
            del hdr[key]

    hdr['BUNIT'] = hdr.get('BUNIT', 'ADU')
    if saturate_level is not None:
        hdr['SATURATE'] = (float(saturate_level), 'Raw detector full scale (ADU)')
    if finite.any():
        hdr['DATAMIN'] = (float(arr[finite].min()), 'Min calibrated value (ADU)')
        hdr['DATAMAX'] = (float(arr[finite].max()), 'Max calibrated value (ADU)')
    hdr.add_history("Saved as unsigned 16-bit ADU (BSCALE=1, BZERO=32768), "
                    "same convention as the raw frames")
    if n_lo:
        hdr.add_history(f"{n_lo} pixel(s) below 0 ADU clamped to 0")
    if n_hi:
        hdr.add_history(f"{n_hi} pixel(s) above 65535 ADU clamped")
    if n_bad:
        hdr.add_history(f"{n_bad} non-finite pixel(s) set to 0")

    if n_lo or n_hi or n_bad:
        log.warning("%s: %d pixel(s) clamped low, %d clamped high, %d non-finite",
                    os.path.basename(path), n_lo, n_hi, n_bad)

    if stats_out is not None:
        stats_out['clamped_low'] = n_lo
        stats_out['clamped_high'] = n_hi
        stats_out['nonfinite'] = n_bad
        stats_out['size'] = int(np.asarray(data).size)

    hdu = fits.PrimaryHDU(out, header=hdr)
    final_path = _final_output_path(path, hdr, add_timestamp, include_ms, overwrite)
    hdu.writeto(final_path, overwrite=overwrite)
    return final_path


def _final_output_path(path, hdr, add_timestamp, include_ms, overwrite):
    """The file name a calibrated frame is written to: `path` plus a
    DATE-OBS-derived suffix, '.fits' when it has no extension, and _1, _2 ...
    when it must not overwrite."""
    # Try to get DATE-OBS from header
    dateobs_val = hdr.get('DATE-OBS', None) if add_timestamp else None
    short_suffix = ''
    if dateobs_val is not None:
        try:
            dt_suffix = _parse_dateobs(dateobs_val)
            s = _make_short_timestamp(dt_suffix, include_ms=include_ms)
            if s:
                short_suffix = '_' + s
        except Exception:
            short_suffix = ''

    # Ensure path has an extension; if not, use .fits
    base_dir = os.path.dirname(path)
    fname = os.path.basename(path)
    name, ext = os.path.splitext(fname)
    if ext == '':
        ext = '.fits'


    final_fname = f"{name}{short_suffix}{ext}"
    final_path = os.path.join(base_dir, final_fname) if base_dir else final_fname

    if not overwrite:
        final_path = _unique_filename(final_path)
    return final_path


# Storage format of a calibration run. save_fits_16bit clamps everything below
# 0 ADU to 0, which is harmless while the sky sits many sigma above zero -- and
# not on the low-gain channel with a dark sky: PSR J2215 g Low (94 s) calibrates
# to a sky of 2.30 +- 1.63 ADU, 15% of its sky pixels were stored as 0, and
# stars at SNR 5-50 moved by 10-30 mmag. With z = (sky + 0.5) / sigma (rounding
# turns -0.5 ADU into a negative integer), the clamp biases the sky of a
# sky-limited star by sigma * (phi(z) - z * Phi(-z)) per pixel, which in units of
# the star's own error bar is about sqrt(A) times that: for an A ~ 530 px
# aperture ~1% at z = 3, 5% at 2.5, 20% at 2, 67% at 1.5. Below z = 3 the run is
# stored as 32-bit float instead.
FLOAT_STORAGE_Z = 3.0


def sky_clip_margin(data):
    """
    How far the sky of a calibrated frame sits above the 16-bit writer's clamp.
    Returns (z, sky, sigma), z = (sky + 0.5) / sigma, where sky and sigma are the
    3-sigma-clipped median and standard deviation of every 4th pixel in each
    axis. Hot, dead and star pixels do not move them: those clamp in almost
    every frame and say nothing about whether the SKY survives integer storage.

    Measured z: PSR J2215 g Low 1.7, GRB 251013C g Low 4.4-5.0, PSR J2215 g High
    4.7-4.8, SA 92 g High 2x2 ~30.
    """
    a = np.asarray(data, dtype='float64')[::4, ::4]
    a = a[np.isfinite(a)]
    _, sky, sigma = sigma_clipped_stats(a, sigma=3.0, maxiters=5)
    if not sigma > 0:
        return (np.inf if sky + 0.5 > 0 else -np.inf), float(sky), float(sigma)
    return float((sky + 0.5) / sigma), float(sky), float(sigma)


def save_fits_float32(path, data, header=None, saturate_level=4095, overwrite=False,
                      include_ms=False, add_timestamp=True, reason=None):
    """
    Save a calibrated frame as 32-bit float ADU (BITPIX=-32), keeping negative
    values and fractions of an ADU. Used for every frame of a calibration run in
    which some frame's sky sits too close to 0 ADU for save_fits_16bit (see
    sky_clip_margin and FLOAT_STORAGE_Z). Header keywords and the output file
    name follow save_fits_16bit; non-finite pixels stay NaN.

    reason: one HISTORY line saying why the run is stored as float.
    Returns the path actually written.
    """
    arr = np.asarray(data, dtype='float32')
    finite = np.isfinite(arr)
    n_bad = int(arr.size - finite.sum())

    if header is None:
        hdr = fits.Header()
    else:
        hdr = fits.Header(header) if not isinstance(header, fits.Header) else header.copy()
    # A raw frame's BZERO = 32768 would turn the floats back into scaled integers.
    for key in ('BSCALE', 'BZERO'):
        if key in hdr:
            del hdr[key]

    hdr['BUNIT'] = hdr.get('BUNIT', 'ADU')
    if saturate_level is not None:
        hdr['SATURATE'] = (float(saturate_level), 'Raw detector full scale (ADU)')
    if finite.any():
        hdr['DATAMIN'] = (float(arr[finite].min()), 'Min calibrated value (ADU)')
        hdr['DATAMAX'] = (float(arr[finite].max()), 'Max calibrated value (ADU)')
    hdr.add_history("Saved as 32-bit float ADU: negative sky values and fractions kept")
    if reason:
        hdr.add_history(reason)
    if n_bad:
        hdr.add_history(f"{n_bad} non-finite pixel(s) kept as NaN")

    final_path = _final_output_path(path, hdr, add_timestamp, include_ms, overwrite)
    fits.PrimaryHDU(arr, header=hdr).writeto(final_path, overwrite=overwrite)
    return final_path

def remove_outliers(data, sigma=5.0, box_size=3):
    """
    Advanced Outlier Remover (Thresholded Median Filter) inspired by AstroImageJ.
    Identifies pixels that deviate (up or down) from the local median by more than 
    a specified threshold and replaces them with that median.
    """
    from scipy.ndimage import median_filter
    from astropy.stats import mad_std
    
    # 1. Calculate local median map
    local_median = median_filter(data, size=box_size, mode='mirror')
    
    # 2. Identify deviations
    diff = data - local_median
    
    # 3. Robust noise estimate (MAD is less sensitive to the outliers we're trying to find)
    # We sample a chunk for speed if image is huge
    sample_size = min(diff.size, 500000)
    if diff.size > sample_size:
        idx = np.random.choice(diff.size, sample_size, replace=False)
        std_est = mad_std(diff.ravel()[idx], ignore_nan=True)
    else:
        std_est = mad_std(diff, ignore_nan=True)
        
    if std_est <= 0: return data
    
    # 4. Create Mask for outliers (> sigma * standard_deviation)
    # Absolute difference handles both Hot (positive) and Cold (negative) pixels
    mask = (np.abs(diff) > sigma * std_est)
    
    cleaned = np.copy(data)
    cleaned[mask] = local_median[mask]
    
    num_fixed = np.sum(mask)
    if num_fixed > 0:
        print(f"[CLEAN] Removed {num_fixed} outliers (sigma={sigma}, box={box_size})")
        
    return cleaned

def bin_ndarray(ndarray, bin_factor):
    """
    Bin a 2D array by a factor (integer). Uses pixel averaging.
    """
    if bin_factor <= 1:
        return ndarray
    
    new_shape = (ndarray.shape[0] // bin_factor, ndarray.shape[1] // bin_factor)
    
    # Crop to multiple of bin_factor
    arr = ndarray[:new_shape[0]*bin_factor, :new_shape[1]*bin_factor]
    
    # Reshape and mean
    return arr.reshape(new_shape[0], bin_factor, new_shape[1], bin_factor).mean(axis=(1, 3))

# ----------------- helpers for 2D polynomial modeling (used by auto-flat) -----------------

def build_poly_design(x, y, deg):
    terms = []
    for i in range(deg+1):
        for j in range(deg+1):
            if i + j <= deg:
                terms.append((i, j))
    X = np.vstack([ (x**i)*(y**j) for (i,j) in terms ]).T
    return X, terms


def eval_poly_2d(coeffs, terms, ny, nx, deg):
    """Evaluate sum_ij c_ij * x^i * y^j on a full ny x nx grid.

    The term x^i * y^j is separable, so the whole surface is
        Ypow @ C @ Xpow.T
    with Ypow (ny, deg+1) and Xpow (nx, deg+1). Building the dense design
    matrix instead would allocate ny*nx*len(terms) float64 values -- 1.3 GB for
    a 4096x4096 frame at degree 3 -- which is what made the auto-flat too slow
    to run per frame.
    """
    x = (np.arange(nx, dtype='float64') - nx / 2) / (nx / 2)
    y = (np.arange(ny, dtype='float64') - ny / 2) / (ny / 2)

    xpow = np.vstack([x ** i for i in range(deg + 1)]).T   # (nx, deg+1)
    ypow = np.vstack([y ** j for j in range(deg + 1)]).T   # (ny, deg+1)

    C = np.zeros((deg + 1, deg + 1), dtype='float64')      # C[j, i]
    for c, (i, j) in zip(coeffs, terms):
        C[j, i] += c

    return (ypow @ C @ xpow.T).astype('float32')

def fit_2d_poly(image, mask_good, deg=3):
    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    x_norm = (xx - nx/2) / (nx/2)
    y_norm = (yy - ny/2) / (ny/2)
    X_all, terms = build_poly_design(x_norm.ravel(), y_norm.ravel(), deg)
    mask = mask_good.ravel()
    yvec = image.ravel()[mask]
    X = X_all[mask,:]
    if X.shape[0] < X.shape[1]:
        # Not enough pixels for a good fit, return flat model
        return np.zeros(len(terms)), terms, np.ones((ny, nx))
    
    coeffs, *_ = np.linalg.lstsq(X, yvec, rcond=None)
    model_flat = (X_all @ coeffs).reshape((ny, nx))
    return coeffs, terms, model_flat

# ----------------- create auto flat from a single light frame -----------------

def create_auto_flat_from_light(light_frame, master_dark=None, master_bias=None,
                                polydeg=3, star_sigma=4.0):
    """
    Build a modeled (normalized) flat from a single light frame.
    Speed Optimized: Fits directly to unmasked pixels.
    """
    img = light_frame.astype('float32')
    ny, nx = img.shape

    # 1. Remove the additive pedestal, using the same rule as the main path.
    corrected = subtract_pedestal(img, master_dark, master_bias)
    # The fit wants sky, not noise below it; clipping here only affects which
    # pixels feed the surface fit, not the science frame.
    corrected = np.where(corrected < 0, 0.0, corrected)

    # 2. Downsample for Speed (Background/Vignetting is low-frequency)
    # Using a factor of 8 reduces pixels by 64x
    bin_f = 8
    target_ny, target_nx = ny // bin_f, nx // bin_f
    if target_ny > 10 and target_nx > 10:
        img_small = corrected[:target_ny*bin_f, :target_nx*bin_f].reshape(target_ny, bin_f, target_nx, bin_f).mean(axis=(1, 3))
    else:
        img_small = corrected
        bin_f = 1

    # 3. Detect bright sources on small image
    _, med, std = sigma_clipped_stats(img_small, sigma=3.0, maxiters=3)
    threshold = med + star_sigma * std
    good_pixels = (img_small < threshold) & (img_small > 0)

    # 4. Fit 2D Polynomial to small image
    coeffs, terms, model_small = fit_2d_poly(img_small, good_pixels, deg=polydeg)

    # 5. Re-evaluate the fitted surface at full resolution.
    # Separable evaluation: no ny*nx*n_terms design matrix (1.3 GB at 4k/deg3).
    if bin_f > 1:
        model = eval_poly_2d(coeffs, terms, ny, nx, polydeg)
    else:
        model = np.asarray(model_small, dtype='float32')

    # 6. Normalize
    model[model <= 0] = 1.0 # Safety
    med_model = np.median(model)
    if med_model <= 0 or not np.isfinite(med_model):
        med_model = np.mean(model)
    if med_model <= 0 or not np.isfinite(med_model):
        med_model = 1.0
    
    model /= med_model
    
    return {
        'model': model.astype('float32'),
        'norm_value': float(med)
    }

def apply_hot_pixel_mask(data, mask, box_size=3):
    """
    Interpolate pixels identified by the mask (mask == 1).
    Avoids using other hot pixels during interpolation.
    Expands the box if no valid neighbors are found.
    """
    ny, nx = data.shape
    cleaned = np.copy(data).astype('float32')
    
    # Ensure mask shape matches data (simple matching, logic for mismatch should be handled by caller)
    if mask.shape != data.shape:
        return data

    hot_y, hot_x = np.where(mask > 0.5)
    if hot_y.size == 0:
        return cleaned

    for y, x in zip(hot_y, hot_x):
        d = box_size // 2
        found = False
        # Limit expansion to reasonable size (e.g., 21x21)
        while d < 10: 
            y0, y1 = max(0, y - d), min(ny, y + d + 1)
            x0, x1 = max(0, x - d), min(nx, x + d + 1)
            
            sub_data = data[y0:y1, x0:x1]
            sub_mask = mask[y0:y1, x0:x1]
            
            # Mask out the hot pixels in the sub-region
            valid_mask = (sub_mask < 0.5)
            valid_vals = sub_data[valid_mask]
            
            if valid_vals.size > 0:
                cleaned[y, x] = np.mean(valid_vals)
                found = True
                break
            else:
                d += 1 # Expand search radius
        
    print(f"[HOTPIX] Interpolated {hot_y.size} pixels using mask.")
    return cleaned



