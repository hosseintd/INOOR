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

# ---------------- existing calibrate functions (kept, slightly refactored) ----------------

def calibrate_by_gaintable(light_frame, master_dark, master_bias, gain_table):
    lf = light_frame.astype('float32')
    md = master_dark.astype('float32') if master_dark is not None else np.zeros_like(lf)
    mb = master_bias.astype('float32') if master_bias is not None else np.zeros_like(lf)
    gt = gain_table.astype('float32')

    dark_corrected = md - mb
    dark_corrected[dark_corrected < 0] = 0.0

    calibrated_frame = (lf - mb - dark_corrected)
    calibrated_frame[calibrated_frame < 0] = 0.0

    calibrated_frame = calibrated_frame * gt
    return np.array(calibrated_frame, dtype='float32')


def calibrate(light_frame, master_dark, master_bias, master_flat):
    lf = light_frame.astype('float32')
    md = master_dark.astype('float32') if master_dark is not None else np.zeros_like(lf)
    mb = master_bias.astype('float32') if master_bias is not None else np.zeros_like(lf)
    mf = master_flat.astype('float32')

    dark_corrected = md - mb
    dark_corrected[dark_corrected < 0] = 0.0

    calibrated_frame = (lf - mb - dark_corrected)
    calibrated_frame[calibrated_frame < 0] = 0.0

    # master_flat may already be normalized (median ~ 1) or not.
    # Normalize by median to be robust against outliers/hot pixels.
    ref_val = np.median(mf)
    if not np.isfinite(ref_val) or ref_val <= 0:
        # Fallback to mean if median is bad
        ref_val = np.mean(mf)
    
    if not np.isfinite(ref_val) or ref_val <= 0:
        norm_flat = np.ones_like(mf)
    else:
        norm_flat = mf / ref_val
        eps = 1e-6
        norm_flat[norm_flat < eps] = eps

    calibrated_frame = calibrated_frame / norm_flat
    return np.array(calibrated_frame, dtype='float32')

# ---------------- master creation helper (kept) ----------------

def create_master_from_list(file_list, method='median', do_sigma_clip=False,
                            lower_sigma=3.0, upper_sigma=3.0,
                            exclude_indices=None, progress_callback=None):
    if not file_list:
        raise ValueError("Empty file list for master creation.")

    if len(file_list) == 1:
        return mfc.load_fits(file_list[0])

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
                          save_as_nbit=False)
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

def _to_12bit_uint16(arr, clip_min=0, clip_max=4095):
    a = np.array(arr, dtype='float32', copy=True)
    a[~np.isfinite(a)] = 0.0
    a[a < clip_min] = clip_min
    a = np.rint(a)
    a = np.clip(a, clip_min, clip_max)
    return a.astype(np.uint16)

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

def save_fits_16bit(path, data, header=None, clip_max=4095, overwrite=False, include_ms=False):
    """
    Save data as uint16 (0..clip_max) inside a 16-bit FITS file.
    The DATE-OBS short timestamp (if found) will be inserted BEFORE the file extension.

    Args:
      path: desired output path (can include directory and extension)
      data: ndarray to save
      header: astropy Header or mapping (optional)
      clip_max: max value for 12-bit range (default 4095)
      overwrite: whether to overwrite existing file
      include_ms: include milliseconds in suffix if available
    Returns:
      the path actually written
    """
    # convert & clip data to 12-bit -> uint16
    out = _to_12bit_uint16(data, clip_min=0, clip_max=clip_max)

    # build header/hdu
    if header is None:
        hdu = fits.PrimaryHDU(out)
        hdr = hdu.header
    else:
        hdr = fits.Header(header) if not isinstance(header, fits.Header) else header.copy()
        hdr['BUNIT'] = hdr.get('BUNIT', 'ADU')
        hdr.add_history(f"saved as 16-bit with 12-bit range (0..{clip_max})")
        hdu = fits.PrimaryHDU(out, header=hdr)

    # Try to get DATE-OBS from header
    dateobs_val = hdr.get('DATE-OBS', None)
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

    hdu.writeto(final_path, overwrite=overwrite)

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
    
    # 1. Dark/Bias Correction
    mb = master_bias.astype('float32') if master_bias is not None else np.zeros_like(img)
    md = master_dark.astype('float32') if master_dark is not None else np.zeros_like(img)
    corrected = img - mb - (md - mb)
    corrected[corrected < 0] = 0.0

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

    # 5. Upsample Model to full size
    if bin_f > 1:
        # Use simple zoom or just re-eval poly on full grid (re-eval is safer)
        yy, xx = np.mgrid[0:ny, 0:nx]
        x_norm = (xx - nx/2) / (nx/2)
        y_norm = (yy - ny/2) / (ny/2)
        X_full, _ = build_poly_design(x_norm.ravel(), y_norm.ravel(), polydeg)
        model = (X_full @ coeffs).reshape((ny, nx))
    else:
        model = model_small

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



