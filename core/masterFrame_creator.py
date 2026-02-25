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

# -------------------------
# New helper: write image as N-bit by storing in uint16 and writing BSCALE/BZERO
# -------------------------
def write_nbit_fits(output_path, image_float, nbits=12, hdr_extra=None, overwrite=True):
    """
    Save a float image array as N-bit integer stored in a uint16 FITS image.
    Uses FITS BSCALE/BZERO so the original float range can be recovered:
        real = stored * BSCALE + BZERO
    - If image values are already in 0..(2**nbits - 1), they are rounded directly.
    - Otherwise a linear mapping min->0, max->(2**nbits - 1) is used.
    """
    if hdr_extra is None:
        hdr_extra = {}

    nmax = (1 << nbits) - 1  # 2**nbits - 1

    # handle NaNs/infs
    img = np.array(image_float, dtype='float32')  # use higher precision for scaling math
    finite_mask = np.isfinite(img)
    if not np.any(finite_mask):
        # all bad -> write zeros
        stored = np.zeros_like(img, dtype=np.uint16)
        bscale = 1.0
        bzero = 0.0
    else:
        dmin = float(np.nanmin(img))
        dmax = float(np.nanmax(img))
        # If already inside 0..nmax, simply round and store (no scaling)
        if dmin >= 0.0 and dmax <= nmax:
            bscale = 1.0
            bzero = 0.0
            # round finite values, set non-finite to 0
            stored = np.zeros_like(img, dtype=np.float32)
            stored[finite_mask] = np.round(img[finite_mask])
            stored[~finite_mask] = 0.0
        else:
            # linear scale: map dmin->0, dmax->nmax
            if dmax == dmin:
                # constant image outside expected range -> clamp to middle
                bscale = 1.0
                bzero = dmin
                stored = np.zeros_like(img, dtype=np.float32)
                stored[finite_mask] = np.round((img[finite_mask] - bzero) / bscale)
            else:
                bscale = (dmax - dmin) / float(nmax)
                bzero = dmin
                stored = np.zeros_like(img, dtype=np.float32)
                # (real - bzero) / bscale -> stored integer
                stored[finite_mask] = np.round((img[finite_mask] - bzero) / bscale)
                stored[~finite_mask] = 0.0

        # clip into valid stored integer range
        stored = np.clip(stored, 0, nmax).astype(np.uint16)

    # final stored array dtype uint16
    stored_uint16 = stored.astype(np.uint16)

    # Build HDU and write header keywords
    hdu = fits.PrimaryHDU(stored_uint16)
    hdr = hdu.header
    # Use BSCALE/BZERO so reading will allow reconstructing original floats:
    hdr['BSCALE'] = (float(bscale), 'Scale factor to recover original floats')
    hdr['BZERO']  = (float(bzero), 'Zero point to recover original floats')
    hdr['NBITS']  = (int(nbits), 'Original bit depth requested')
    hdr['BUNIT']  = hdr.get('BUNIT', 'ADU')

    # store some provenance (original min/max before scaling)
    try:
        hdr['ORIGMIN'] = (float(np.nanmin(image_float)), 'Original image min (float)')
        hdr['ORIGMAX'] = (float(np.nanmax(image_float)), 'Original image max (float)')
    except Exception:
        pass

    # add any extra header keys efficiently
    if hdr_extra:
        if hasattr(hdr_extra, 'cards'):
            # If it's a Header object, we can extend/overlap safely
            # We avoid core keywords
            for card in hdr_extra.cards:
                k = card.keyword.upper()
                if k in ('SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXTEND', 'BSCALE', 'BZERO', 'NBITS', 'PCOUNT', 'GCOUNT'):
                    continue
                # For HISTORY/COMMENT, we append. For others, we update.
                if k in ('HISTORY', 'COMMENT'):
                    hdr.add_history(card.value) if k == 'HISTORY' else hdr.add_comment(card.value)
                else:
                    # Clean value of newlines or control chars
                    val = card.value
                    if isinstance(val, str):
                        val = "".join(ch for ch in val if ch.isprintable())
                    hdr[k] = (val, card.comment)
        else:
            # It's a dict
            for k, v in hdr_extra.items():
                if k.upper() in ('BSCALE', 'BZERO', 'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2'):
                    continue
                if isinstance(v, str):
                    # SAFETY: FITS values cannot contain \n. 
                    # If dict(header) was passed, HISTORY might be a \n separated string.
                    if k.upper() == 'HISTORY':
                        for line in v.split('\n'):
                            if line.strip(): hdr.add_history(line.strip())
                    elif k.upper() == 'COMMENT':
                        for line in v.split('\n'):
                            if line.strip(): hdr.add_comment(line.strip())
                    else:
                        v = "".join(ch for ch in v if ch.isprintable())
                        hdr[k] = v
                else:
                    hdr[k] = v

    # Write file
    hdu.writeto(output_path, overwrite=overwrite)
    return output_path

from datetime import datetime

def apply_2d_sigma_clip(data, sigma_lower, sigma_upper, kernel_size, auto_sigma=False):
    """
    Apply sigma clipping to a 2D image by iterating area-by-area (blocks).
    Outliers in each block are replaced by the block median.
    
    In auto_sigma mode:
    - sigma_lower is ignored (set to very high).
    - sigma_upper is used to find outliers (defaults to 3.0 if not specified).
    """
    ny, nx = data.shape
    processed = data.copy()
    
    # Ensure kernel_size is at least 3
    ks = max(3, kernel_size)
    
    # In auto mode, we don't clipp lower values.
    # We use a very high lower threshold to effectively disable it.
    slo = 100.0 if auto_sigma else sigma_lower
    shi = sigma_upper # 3.0 is a good 'appropriate' default if sigma_upper is the default 3.0
    
    for y in range(0, ny, ks):
        for x in range(0, nx, ks):
            y_end = min(y + ks, ny)
            x_end = min(x + ks, nx)
            
            block = data[y:y_end, x:x_end]
            if block.size == 0: continue
            
            # Use astropy's sigma_clip
            clipped = sigma_clip(block, sigma_lower=slo, sigma_upper=shi, maxiters=3)
            if hasattr(clipped, 'mask') and np.any(clipped.mask):
                block_median = np.nanmedian(block)
                # Fill masked (outlier) values with median
                block_processed = clipped.filled(block_median)
                processed[y:y_end, x:x_end] = block_processed
                
    return processed

def create_master(output_path, file_list, method='median', do_sigma_clip=False,
                  sigma_lower=3.0, sigma_upper=3.0, exclude_indices=None,
                  progress_callback=None, save_as_nbit=True, nbits=12, hdr_extra=None, 
                  kernel=5, auto_sigma=False):
    """
    Create a master frame from file_list and write to output_path.
    
    1. Stack frames using method.
    2. Apply 2D Post-filter Sigma Clipping if requested.
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

    # Load all data into a 3D stack
    data_list = []
    base_header = None
    
    for idx, fpath in enumerate(files):
        img = load_fits(fpath)
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

    if progress_callback:
        progress_callback(0.5, "Stacking frames...")

    # Build 3D stack: Shape (NY, NX, N)
    stack = np.dstack(data_list)
    del data_list # Free memory

    # 1. Perform Stacking
    if method == 'median':
        master = np.nanmedian(stack, axis=2)
    else:
        master = np.nanmean(stack, axis=2)
    
    del stack # Free memory

    # 2. Apply 2D Sigma Clipping Post-filter if requested
    if do_sigma_clip:
        mode_str = "Auto" if auto_sigma else "Manual"
        if progress_callback:
            progress_callback(0.7, f"Applying 2D Sigma Clip ({mode_str}, Kernel: {kernel})...")
        master = apply_2d_sigma_clip(master, sigma_lower, sigma_upper, kernel, auto_sigma=auto_sigma)

    # ensure float32
    master = np.array(master, dtype='float32')
    
    # --- Prepare Header ---
    if base_header is None: base_header = fits.Header()
    
    # Update standard metadata
    base_header['STK_METH'] = (method.upper(), 'Stacking Method used')
    base_header['STK_NUM'] = (n, 'Number of frames combined')
    base_header['STK_CLIP'] = (do_sigma_clip, '2D Sigma clipping applied?')
    if do_sigma_clip:
        base_header['STK_CMOD'] = ("AUTO" if auto_sigma else "MANUAL", 'Sigma clipping mode')
        if not auto_sigma:
            base_header['STK_SLO'] = (sigma_lower, 'Lower sigma value')
        base_header['STK_SHI'] = (sigma_upper, 'Upper sigma value')
        base_header['STK_KERN'] = (kernel, 'Kernel size for clipping')
    
    base_header['CRE_TIME'] = (datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), 'Time of creation')
    
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
        if save_as_nbit:
            # write as n-bit using helper; BSCALE/BZERO will be added
            write_nbit_fits(output_path, master, nbits=nbits, hdr_extra=dict(base_header), overwrite=True)
        else:
            hdu = fits.PrimaryHDU(master, header=base_header)
            hdu.writeto(output_path, overwrite=True)
        
        if progress_callback:
            progress_callback(1.0, "Done")
        return output_path
    else:
        # If no path provided, return the data and header for review
        if progress_callback:
            progress_callback(1.0, "Ready for review")
        return master, base_header

# -------------------------
# Gain table / polynomial fit helpers (unchanged)
# -------------------------
def _poly_terms_indices(degree):
    pairs = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            pairs.append((i, j))
    return pairs

def fit_2d_polynomial(flat_data, degree=2, max_samples=500_000, progress_callback=None):
    """
    Fit a 2D polynomial surface to flat_data. For very large images uses random sampling
    (max_samples) to keep the linear system manageable.
    Returns (coeffs, degree).
    """
    ny, nx = flat_data.shape
    X_coords = np.arange(nx)
    Y_coords = np.arange(ny)
    X_grid, Y_grid = np.meshgrid(X_coords, Y_coords)
    X_flat = X_grid.ravel()
    Y_flat = Y_grid.ravel()
    Z_flat = flat_data.ravel()
    finite_mask = np.isfinite(Z_flat)
    X_flat = X_flat[finite_mask]
    Y_flat = Y_flat[finite_mask]
    Z_flat = Z_flat[finite_mask]
    Npix = Z_flat.size
    if Npix == 0:
        raise ValueError("No finite pixels in flat_data for fitting.")

    if progress_callback:
        progress_callback(0.0, "Preparing samples for fit...")

    if Npix > max_samples:
        rng = np.random.default_rng()
        idx = rng.choice(Npix, size=max_samples, replace=False)
        Xs = X_flat[idx]
        Ys = Y_flat[idx]
        Zs = Z_flat[idx]
    else:
        Xs = X_flat
        Ys = Y_flat
        Zs = Z_flat

    if progress_callback:
        progress_callback(0.1, f"Building design matrix ({Xs.size} samples)...")

    pairs = _poly_terms_indices(degree)
    A_cols = []
    for k, (i, j) in enumerate(pairs):
        A_cols.append((Xs**i) * (Ys**j))
        if progress_callback and (k % 5 == 0):
            progress_callback(0.1 + 0.8 * (k / max(1, len(pairs))), f"Building term {k+1}/{len(pairs)}")

    A = np.column_stack(A_cols)
    if progress_callback:
        progress_callback(0.9, "Solving least squares...")

    coeffs, *_ = np.linalg.lstsq(A, Zs, rcond=None)
    if progress_callback:
        progress_callback(1.0, "Fit complete")
    return coeffs, degree

def reconstruct_polynomial(coeffs, degree, shape, progress_callback=None):
    """
    Reconstruct the fitted polynomial surface (full resolution).
    """
    ny, nx = shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(shape, dtype='float32')
    pairs = _poly_terms_indices(degree)
    for idx, (i, j) in enumerate(pairs):
        Z += coeffs[idx] * (X**i) * (Y**j)
        if progress_callback and (idx % 5 == 0):
            progress_callback(idx / max(1, len(pairs)), f"Reconstructing term {idx+1}/{len(pairs)}")
    if progress_callback:
        progress_callback(1.0, "Reconstruction complete")
    return Z.astype('float32')

def create_gain_table_from_master(master_array, degree=2, fit_max_samples=500_000, progress_callback=None):
    """
    Fit polynomial to master flat and compute gain table:
      z_fitted = fitted surface
      f2 = z_fitted / master_array
      gain_table = f2 / median(f2)
    Returns (z_fitted, gain_table, coeffs)
    """
    if progress_callback:
        progress_callback(0.0, "Starting polynomial fit...")
    coeffs, deg = fit_2d_polynomial(master_array, degree=degree, max_samples=fit_max_samples, progress_callback=progress_callback)
    if progress_callback:
        progress_callback(0.2, "Reconstructing fitted surface...")
    z_fitted = reconstruct_polynomial(coeffs, deg, master_array.shape, progress_callback=progress_callback)
    if progress_callback:
        progress_callback(0.8, "Calculating gain table...")
    eps = 1e-12
    f2 = z_fitted / (master_array + eps)
    med = np.median(f2[np.isfinite(f2)])
    if med == 0 or not np.isfinite(med):
        med = 1.0
    gain_table = (f2 / med).astype('float32')
    if progress_callback:
        progress_callback(1.0, "Gain table ready")
    return z_fitted, gain_table, coeffs
