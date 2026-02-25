# align_utils.py
# Utilities for local stamp detection, centroid/2D-Gaussian refinement and robust translation-only alignment (RANSAC-like).
# Designed to be robust to astropy-modeling parameter name changes and to return consistent types.

import os
import numpy as np
import warnings
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from astropy.modeling import models, fitting
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, warp
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from scipy.spatial import cKDTree as KDTree
import math

warnings.filterwarnings('ignore')

# ---------- parameters (tuneable) ----------
DEFAULT_FWHM_GUESS = 15
DEFAULT_STAMP_RADIUS = 90
DEFAULT_MAX_SEARCH_RADIUS = 30
DEFAULT_UPSAMPLE_XCORR = 50
# -------------------------------------------

def load_frame(path):
    """Load primary HDU data as float32 2D array (squeezes extra dims)."""
    with fits.open(path, memmap=False) as hd:
        data = hd[0].data
        if data is None:
            raise ValueError(f"No image data found in {path}")
        data = np.array(data, dtype=float)
        if data.ndim > 2:
            data = np.squeeze(data)
        return data

def cut_stamp(data, cx, cy, rad):
    """Return stamp, x0, y0 so that stamp == data[y0:y1, x0:x1]."""
    ny, nx = data.shape
    x0 = int(round(cx)); y0 = int(round(cy))
    x1 = max(0, x0 - rad); x2 = min(nx, x0 + rad + 1)
    y1 = max(0, y0 - rad); y2 = min(ny, y0 + rad + 1)
    stamp = data[y1:y2, x1:x2].copy()
    return stamp, x1, y1

def estimate_background(stamp):
    """Edge-median background estimate and std (robust for small stamps)."""
    h, w = stamp.shape
    p = int(max(1, 0.12 * min(h, w)))
    edges = np.concatenate([stamp[:p,:].ravel(), stamp[-p:,:].ravel(),
                            stamp[:,:p].ravel(), stamp[:,-p:].ravel()])
    bg = np.median(edges)
    bgstd = float(np.std(edges))
    return float(bg), float(bgstd)

def _get_param_value(model, base_name):
    """
    Given an Astropy model (possibly compound), find a parameter whose name starts with base_name
    and return its numeric value. Handles Parameter objects (.value) and direct numeric attributes.
    """
    for nm in getattr(model, "param_names", ()):
        if nm.startswith(base_name):
            val = getattr(model, nm)
            if hasattr(val, "value"):
                return float(val.value)
            try:
                return float(val)
            except Exception:
                return None
    # fallback to attribute name
    if hasattr(model, base_name):
        val = getattr(model, base_name)
        if hasattr(val, "value"):
            return float(val.value)
        try:
            return float(val)
        except Exception:
            return None
    return None

def twoD_Gaussian_fit(stamp, fwhm_guess=DEFAULT_FWHM_GUESS):
    """
    Fit a 2D Gaussian + constant to the stamp using Astropy modeling.
    Returns dict with x_mean, y_mean (in stamp coords) or None on failure.
    Works with compound models across astropy versions.
    """
    if stamp is None or stamp.size == 0:
        return None
    h, w = stamp.shape
    
    # Defensive check: ensure stamp has valid (non-NaN) data
    # This prevents crashes on bias frames or fully-masked regions
    if np.all(~np.isfinite(stamp)):
        # All NaN or inf - cannot fit
        return None
    
    yy, xx = np.mgrid[0:h, 0:w]
    med = float(np.median(stamp))
    amp_guess = max(float(np.nanmax(stamp) - med), 1.0)
    
    try:
        max_idx = np.unravel_index(np.nanargmax(stamp), stamp.shape)
    except ValueError:
        # "All-NaN slice encountered" - stamp has no finite values
        return None
    
    y0_guess, x0_guess = float(max_idx[0]), float(max_idx[1])
    sigma_guess = fwhm_guess / (2.0 * np.sqrt(2*np.log(2)))
    g_init = models.Gaussian2D(amplitude=amp_guess,
                               x_mean=x0_guess, y_mean=y0_guess,
                               x_stddev=max(0.5, sigma_guess), y_stddev=max(0.5, sigma_guess), theta=0.0)
    c_init = models.Const2D(med)
    compound = g_init + c_init

    fitter = fitting.LevMarLSQFitter()
    try:
        g = fitter(compound, xx, yy, stamp, maxiter=200)
    except Exception:
        return None

    x_mean_val = _get_param_value(g, 'x_mean')
    y_mean_val = _get_param_value(g, 'y_mean')
    amp_val = _get_param_value(g, 'amplitude')
    x_std = _get_param_value(g, 'x_stddev') or _get_param_value(g, 'x_std')
    y_std = _get_param_value(g, 'y_stddev') or _get_param_value(g, 'y_std')

    if x_mean_val is None or y_mean_val is None:
        return None
    return {'x_mean': float(x_mean_val),
            'y_mean': float(y_mean_val),
            'amplitude': float(amp_val) if amp_val is not None else float(amp_guess),
            'x_std': float(x_std) if x_std is not None else float(sigma_guess),
            'y_std': float(y_std) if y_std is not None else float(sigma_guess)}

def centroid_com(stamp):
    """Center-of-mass centroid after background subtraction (stamp coords)."""
    med = np.median(stamp)
    data = stamp - med
    data = np.where(data > 0, data, 0.0)
    s = data.sum()
    if s <= 0 or not np.isfinite(s):
        return None
    yy, xx = np.mgrid[0:stamp.shape[0], 0:stamp.shape[1]]
    x = (data * xx).sum() / s
    y = (data * yy).sum() / s
    return {'x_mean': float(x), 'y_mean': float(y)}

def find_star_in_frame(frame, expected_xy, stamp_radius=DEFAULT_STAMP_RADIUS,
                       fwhm_guess=DEFAULT_FWHM_GUESS, expand_steps=(0,10,20), max_search_radius=200):
    """
    Try to locate the star around expected_xy using progressively larger local stamps,
    Gaussian fit, then centroid. Returns (found_xy (full-frame coords), measured_parameters, stamp_smoothed)
    or (None, None, None).
    """
    cx, cy = expected_xy
    
    # Sensible expansion of search area
    radii = [stamp_radius]
    for s in expand_steps:
        if s > 0:
            new_r = min(max_search_radius, stamp_radius + s)
            if new_r not in radii:
                radii.append(new_r)
    
    # Ensure we try a large-ish one if others fail
    last_r = min(max_search_radius, stamp_radius + 100)
    if last_r not in radii:
        radii.append(last_r)

    for rad in radii:
        stamp, x0, y0 = cut_stamp(frame, cx, cy, rad)
        if stamp.size == 0:
            continue
        bg, bgstd = estimate_background(stamp)
        stamp_bs = stamp - bg
        stamp_s = gaussian_filter(stamp_bs, sigma=max(0.8, fwhm_guess/3.0))
        fit = twoD_Gaussian_fit(stamp_s, fwhm_guess)
        if fit is not None:
            xf = x0 + fit['x_mean']; yf = y0 + fit['y_mean']
            return (xf, yf), fit, stamp_s
        cent = centroid_com(stamp_s)
        if cent is not None:
            xf = x0 + cent['mean_x'] if 'mean_x' in cent else x0 + cent['x_mean']
            yf = y0 + cent['mean_y'] if 'mean_y' in cent else y0 + cent['y_mean']
            return (xf, yf), cent, stamp_s
    return None, None, None

def find_star_adaptive(frame, expected_xy, previous_detected_xy=None,
                       stamp_radius=DEFAULT_STAMP_RADIUS,
                       fwhm_guess=DEFAULT_FWHM_GUESS, max_search_radius=200):
    """
    Adaptive source detection that uses frame-to-frame position tracking.
    
    If previous_detected_xy is provided (from previous frame), predicts new position
    based on drift and searches around that. Falls back to fixed-position search.
    
    Returns (found_xy, measured_parameters, stamp_smoothed) or (None, None, None).
    """
    cx, cy = expected_xy
    
    # Step 1: If we have previous position, estimate drift and search near predicted location
    if previous_detected_xy is not None:
        px, py = previous_detected_xy
        # Estimate frame drift from last detection relative to initial ref
        drift_x = px - cx
        drift_y = py - cy
        
        # Search near predicted position with progressively larger radii
        predicted_cx = cx + drift_x
        predicted_cy = cy + drift_y
        
        search_radii = [
            stamp_radius,
            min(max_search_radius, stamp_radius + 50),
            min(max_search_radius, stamp_radius + 150)
        ]
        
        for rad in search_radii:
            stamp, x0, y0 = cut_stamp(frame, predicted_cx, predicted_cy, rad)
            if stamp.size == 0:
                continue
            bg, bgstd = estimate_background(stamp)
            stamp_bs = stamp - bg
            stamp_s = gaussian_filter(stamp_bs, sigma=max(0.8, fwhm_guess/3.0))
            fit = twoD_Gaussian_fit(stamp_s, fwhm_guess)
            if fit is not None:
                xf = x0 + fit['x_mean']; yf = y0 + fit['y_mean']
                return (xf, yf), fit, stamp_s
            cent = centroid_com(stamp_s)
            if cent is not None:
                xf = x0 + cent['x_mean']; yf = y0 + cent['y_mean']
                return (xf, yf), cent, stamp_s
    
    # Step 2: Fall back to standard fixed-position search
    return find_star_in_frame(frame, expected_xy, stamp_radius=stamp_radius,
                             fwhm_guess=fwhm_guess, max_search_radius=max_search_radius)

# ---------------- multi-star matching + robust translation estimator ----------------

def _detect_sources_global(image, fwhm=8.0, threshold_sigma=5.0, max_sources=1000):
    """
    Run DAOStarFinder on the whole image and return Nx2 array of centroids ordered by flux descending.
    Returns numpy array shape (N,2) with columns (x, y). May return empty array.
    """
    img = np.array(image, dtype=float)
    if img.size == 0:
        return np.empty((0,2), dtype=float)
    mean, med, std = sigma_clipped_stats(img, sigma=3.0)
    thresh = threshold_sigma * max(std, 1e-12)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=thresh)
    try:
        sources = daofind(img - med)
    except Exception:
        sources = None
    if sources is None or len(sources) == 0:
        return np.empty((0,2), dtype=float)
    # prefer brightest first
    try:
        flux = np.asarray(sources['flux'], dtype=float)
        order = np.argsort(flux)[::-1]
        xcent = np.asarray(sources['xcentroid'], dtype=float)[order]
        ycent = np.asarray(sources['ycentroid'], dtype=float)[order]
    except Exception:
        xcent = np.asarray(sources['xcentroid'], dtype=float)
        ycent = np.asarray(sources['ycentroid'], dtype=float)
    pts = np.column_stack((xcent, ycent))
    if pts.shape[0] > max_sources:
        pts = pts[:max_sources]
    return pts

def _robust_translation_from_matches(src_pts, ref_pts, match_radius=5.0):
    """
    Given src_pts (Nx2) and ref_pts (M x2), find nearest matches and compute robust translation (dx,dy).
    Returns dx, dy, inlier_mask (len = number of matches) or (None, None, None) if no matches.
    Algorithm:
      - use KDTree to find nearest ref for each src
      - keep matches with dist <= match_radius
      - compute median dx,dy; compute residuals and MAD; keep inliers within 3*MAD (or match_radius)
      - return median of inliers
    """
    if src_pts.size == 0 or ref_pts.size == 0:
        return None, None, None
    tree = KDTree(ref_pts)
    dists, idxs = tree.query(src_pts, k=1)
    dists = np.atleast_1d(dists); idxs = np.atleast_1d(idxs)
    mask = dists <= match_radius
    if not np.any(mask):
        return None, None, None
    matched_src = src_pts[mask]
    matched_ref = ref_pts[idxs[mask]]
    shifts = matched_ref - matched_src  # (dx, dy) per match
    if shifts.shape[0] == 0:
        return None, None, None
    dx_med = np.median(shifts[:,0])
    dy_med = np.median(shifts[:,1])
    # MAD-based rejection
    resid = np.sqrt((shifts[:,0]-dx_med)**2 + (shifts[:,1]-dy_med)**2)
    mad = np.median(np.abs(resid - np.median(resid))) if resid.size>0 else 0.0
    if mad <= 0:
        # fallback to simple threshold
        inlier_mask = resid <= match_radius
    else:
        inlier_mask = resid <= max(3.0 * mad, 1.0)
    if not np.any(inlier_mask):
        # fallback to median of initial set
        return float(dx_med), float(dy_med), mask
    final_dx = np.median(shifts[inlier_mask, 0])
    final_dy = np.median(shifts[inlier_mask, 1])
    # build result mask aligned to original src_pts (True where inlier present)
    inliers_global = np.zeros(src_pts.shape[0], dtype=bool)
    inlier_indices = np.nonzero(mask)[0][inlier_mask]
    inliers_global[inlier_indices] = True
    return float(final_dx), float(final_dy), inliers_global

def ransac_align(data, ref_data, fwhm=8.0, threshold_sigma=5.0, max_sources=500, match_radius=8.0):
    """
    Translation-only alignment using multi-star matching + robust translation estimate.
    - detects bright sources in both images (up to max_sources),
    - finds nearest matches and computes robust dx,dy,
    - applies translation to data and returns warped image (same shape as ref_data) and method string.
    If no reliable translation found, raises ValueError.
    """
    # detect sources
    src_pts = _detect_sources_global(data, fwhm=fwhm, threshold_sigma=threshold_sigma, max_sources=max_sources)
    ref_pts = _detect_sources_global(ref_data, fwhm=fwhm, threshold_sigma=threshold_sigma, max_sources=max_sources)
    if src_pts.size == 0 or ref_pts.size == 0:
        raise ValueError("Insufficient sources detected for ransac_align")
    dx, dy, inliers = _robust_translation_from_matches(src_pts, ref_pts, match_radius=match_radius)
    if dx is None:
        raise ValueError("No reliable translation found by ransac_align")
    # apply translation (dx,dy) to data: note dx is in x-direction (cols), dy in y-direction (rows)
    tform = AffineTransform(translation=(dx, dy))
    warped = warp(data, inverse_map=tform.inverse, output_shape=ref_data.shape, preserve_range=True, cval=np.nan).astype(np.float32)
    n_inliers = int(np.sum(inliers)) if inliers is not None else 0
    method = f"ransac_translate dx={dx:.3f},dy={dy:.3f},inliers={n_inliers}"
    return warped, method

def fallback_phase_align(data, ref_data, upsample=DEFAULT_UPSAMPLE_XCORR):
    """
    Align using phase_cross_correlation (normalized images) and return (aligned, method_str).
    """
    # normalize images to reduce influence of offsets
    a = np.array(ref_data, dtype=float)
    b = np.array(data, dtype=float)
    a = a - np.nanmedian(a); a = a / (np.nanstd(a) + 1e-12)
    b = b - np.nanmedian(b); b = b / (np.nanstd(b) + 1e-12)
    try:
        shift, error, diff = phase_cross_correlation(a, b, upsample_factor=upsample)
        dy, dx = shift
        tform = AffineTransform(translation=(dx, dy))
        warped = warp(data, inverse_map=tform.inverse, output_shape=ref_data.shape, preserve_range=True, cval=np.nan).astype(np.float32)
        method = f"phase_xcorr dx={dx:.3f},dy={dy:.3f},err={error:.4g}"
        return warped, method
    except Exception as e:
        raise RuntimeError(f"phase_xcorr failed: {e}")
