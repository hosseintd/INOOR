# photometry_core.py
# Core photometry helpers used by GUI and bulk worker.
# - load_fits
# - detect_sources_in_crop
# - pick_nearest_source
# - detect_then_refine (returns 4-tuple)
# - compute_snr_vs_radius
# - perform_aperture_photometry
# - compute_radial_profile

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
import math
from scipy.spatial import cKDTree as KDTree

from .align_utils import find_star_in_frame

# constants
DEFAULT_FWHM_GUESS = 15
DEFAULT_STAMP_RADIUS = 90

def load_fits(path):
    """Load FITS primary HDU as float32 2D array (memmap safe)."""
    hdul = fits.open(path, memmap=False)
    data = hdul[0].data.astype('float32')
    hdul.close()
    return data

def nearest_point(points, target_point):
    """
    Find nearest point using cKDTree. points must be shape (N,2) or list of (x,y).
    Returns (x,y), index or (None, None) when no points.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return None, None
    if pts.ndim == 1:
        if pts.shape[0] == 2:
            return (float(pts[0]), float(pts[1])), 0
        pts = pts.reshape(1, -1)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be shape (N,2)")
    try:
        tree = KDTree(pts)
        dist, idx = tree.query([target_point], k=1)
        idx0 = int(np.atleast_1d(idx)[0])
        return (float(pts[idx0, 0]), float(pts[idx0, 1])), idx0
    except Exception:
        d2 = np.sum((pts - np.asarray(target_point))**2, axis=1)
        i = int(np.argmin(d2))
        return (float(pts[i,0]), float(pts[i,1])), i

def detect_sources_in_crop(image, center_xy, crop_half_size=500, fwhm=5.0, threshold_sigma=3.0):
    """
    Crop around center_xy in full-image coords and run DAOStarFinder.
    Returns (sources_table or None, crop_origin (x0,y0), crop_image)
    """
    x, y = int(round(center_xy[0])), int(round(center_xy[1]))
    ny, nx = image.shape
    x0 = max(0, x - crop_half_size)
    x1 = min(nx, x + crop_half_size)
    y0 = max(0, y - crop_half_size)
    y1 = min(ny, y + crop_half_size)
    crop = image[y0:y1, x0:x1].astype('float32')

    mean, median, std = sigma_clipped_stats(crop)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * max(std, 1e-12))
    try:
        sources = daofind(crop - median)
    except Exception:
        sources = None
    return sources, (x0, y0), crop

def pick_nearest_source(sources, crop_origin, target_xy):
    """
    Pick the nearest source to target_xy (full-image coords).
    Returns:
      (x_full,y_full) (floats), (x_crop,y_crop), index
    or (None, None, None) when not found.
    """
    if sources is None or len(sources) == 0:
        return None, None, None

    # robust extraction of xcentroid,ycentroid
    try:
        xcol = np.asarray(sources['xcentroid'], dtype=float)
        ycol = np.asarray(sources['ycentroid'], dtype=float)
    except Exception:
        try:
            pts = list(zip(sources['xcentroid'], sources['ycentroid']))
            if len(pts) == 0:
                return None, None, None
            xcol = np.asarray([p[0] for p in pts], dtype=float)
            ycol = np.asarray([p[1] for p in pts], dtype=float)
        except Exception:
            return None, None, None

    if xcol.size == 0:
        return None, None, None

    pts = np.column_stack((xcol, ycol))  # shape (N,2)
    x0, y0 = crop_origin
    tx = float(target_xy[0]) - x0
    ty = float(target_xy[1]) - y0

    # try KDTree then fallback to manual distances
    try:
        nearest, idx = nearest_point(pts, (tx, ty))
    except Exception:
        d2 = (pts[:,0] - tx)**2 + (pts[:,1] - ty)**2
        idx = int(np.argmin(d2))
        nearest = (float(pts[idx,0]), float(pts[idx,1]))

    if nearest is None:
        return None, None, None
    x_crop, y_crop = nearest
    x_full = x0 + x_crop
    y_full = y0 + y_crop
    return (float(x_full), float(y_full)), (float(x_crop), float(y_crop)), int(idx)

def compute_snr_vs_radius(image, center_xy, fwhm=12.0, radii=None,
                          read_noise=3.18, dark_noise=4.15, gain=16.5, sample_max_pixels=200000):
    """
    Compute SNR vs aperture radius around center_xy.
    Returns radii array, snr array, best_radius, mag_error.
    """
    if radii is None:
        max_r = max(40, 5 * fwhm)
        radii = np.linspace(1, max_r, 200)

    x, y = center_xy
    arr = image
    flat = arr.ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size > sample_max_pixels:
        idx = np.random.choice(flat.size, sample_max_pixels, replace=False)
        sample = flat[idx]
    else:
        sample = flat
    background_per_pixel = float(np.mean(sample)) if sample.size > 0 else 0.0
    
    # Also compute radial profile for HWHM
    rad_prof, m_prof = compute_radial_profile(image, center_xy, max_radius=max(60, 5*fwhm))
    hwhm = calculate_hwhm(rad_prof, m_prof)

    snrs = []
    for r in radii:
        ap = CircularAperture((x, y), r=r)
        r_in = r
        r_out = max(r + 1.0, r * 1.5) # Dynamic annulus for SNR check
        ann = CircularAnnulus((x, y), r_in=r_in, r_out=r_out)
        try:
            tab = aperture_photometry(arr, [ap, ann])
            source_flux = float(tab['aperture_sum_0'][0])
            ann_flux = float(tab['aperture_sum_1'][0])
            ann_area = ann.area if hasattr(ann,'area') else math.pi*(r_out**2 - r_in**2)
            bkg_per_pix = ann_flux / ann_area if ann_area > 0 else background_per_pixel
            n_ap = ap.area if hasattr(ap,'area') else math.pi*(r**2)
            total_bkg = bkg_per_pix * n_ap
            net_signal_counts = (source_flux - total_bkg)
            net_signal = net_signal_counts / gain
            noise_sq = (source_flux / gain) + (((bkg_per_pix + (read_noise**2) + dark_noise) * n_ap) / gain)
            noise = np.sqrt(noise_sq) if noise_sq > 0 else 1e-12
            snr = net_signal / noise if noise > 0 else 0.0
        except Exception:
            snr = 0.0
        snrs.append(snr)

    snrs = np.array(snrs)
    if snrs.size == 0:
        return np.asarray(radii), snrs, float(radii[0]), np.inf
    # Check if all values are NaN before calling nanargmax
    if np.all(np.isnan(snrs)):
        # All SNR values are NaN - return default values
        return np.asarray(radii), snrs, float(radii[0] if len(radii) > 0 else 1.0), np.inf
    idx_max = int(np.nanargmax(snrs))
    r_best_snr = float(radii[idx_max])
    
    # New Aperture Rule: (3*HWHM + SNR_Peak_Radius) / 2
    # If HWHM is invalid, fallback to SNR peak.
    if hwhm and hwhm > 0:
        recommended_r = (3.0 * hwhm + r_best_snr) / 2.0
    else:
        recommended_r = r_best_snr
        
    snr_best = float(snrs[idx_max])
    mag_error = 1.09 / snr_best if snr_best > 0 else np.inf
    return np.asarray(radii), snrs, recommended_r, mag_error

def calculate_hwhm(radii, profile):
    """
    Calculate Half-Width at Half-Maximum from a radial profile.
    Assumes profile starts at or near peak (center).
    """
    if len(profile) < 2: return None
    
    # Basic peak (usually first few points)
    # Background subtraction: use the last value as simple background estimate
    bkg = np.nanmin(profile)
    centered = profile - bkg
    peak = np.nanmax(centered)
    half_max = peak / 2.0
    
    if peak <= 0: return None
    
    # Find crossing
    for i in range(len(centered) - 1):
        if centered[i] >= half_max and centered[i+1] < half_max:
            # Linear interpolation
            r1, r2 = radii[i], radii[i+1]
            v1, v2 = centered[i], centered[i+1]
            frac = (half_max - v1) / (v2 - v1)
            return r1 + frac * (r2 - r1)
            
    return None

def perform_aperture_photometry(image, center_xy, aperture_radius,
                                inner_coef=2.0, outer_coef=3.0,
                                read_noise=3.18, dark_noise=4.15, gain=16.5,
                                zeropoint=0.0, exptime=None):
    """
    Returns photometry dict: instr_mag, mag_err, flux, bkg_mean, aperture_area, snr
    """
    ap = CircularAperture((center_xy[0], center_xy[1]), r=aperture_radius)
    
    # Robust annulus handling: ensure r_out > r_in
    r_in = inner_coef * aperture_radius
    r_out = outer_coef * aperture_radius
    if r_out <= r_in:
        r_out = r_in + 1.0
        
    ann = CircularAnnulus((center_xy[0], center_xy[1]), r_in=r_in, r_out=r_out)
    try:
        phot = aperture_photometry(image, [ap, ann])
        source_flux = float(phot['aperture_sum_0'][0])
        ann_flux = float(phot['aperture_sum_1'][0])
        ann_area = ann.area if hasattr(ann,'area') else math.pi * ((outer_coef*aperture_radius)**2 - (inner_coef*aperture_radius)**2)
        bkg_mean = ann_flux / ann_area if ann_area > 0 else 0.0
        ap_area = ap.area if hasattr(ap,'area') else math.pi * aperture_radius**2
        total_bkg = bkg_mean * ap_area
        net_counts = source_flux - total_bkg
        # Defensive check: exptime must be positive and reasonable (>= 1e-6 s = 1 microsecond)
        # Very small exposure times (bias frames at 30µs) are treated as invalid
        if exptime is None or exptime <= 0 or not np.isfinite(exptime):
            instr_mag = None
            mag = None
        else:
            val = max(1e-9, net_counts / exptime)
            instr_mag = -2.5 * math.log10(val)
            mag = instr_mag + zeropoint
        n_ap = ap_area
        noise_sq = (source_flux / gain) + (((bkg_mean + (read_noise**2) + dark_noise) * n_ap) / gain)
        noise = math.sqrt(noise_sq) if noise_sq > 0 else 1e-12
        net_signal = (net_counts) / gain
        snr = net_signal / noise if noise > 0 else 0.0
        mag_err = 1.09 / snr if snr > 0 else np.inf
        return dict(instr_mag=instr_mag, mag=mag, mag_err=mag_err, flux=net_counts,
                    bkg_mean=bkg_mean, aperture_area=ap_area, snr=snr)
    except Exception:
        return dict(instr_mag=None, mag_err=np.inf, flux=0.0, bkg_mean=0.0, aperture_area=0.0, snr=0.0)

def compute_radial_profile(image, center_xy, max_radius):
    """
    Compute radial mean profile in 1-pixel-thick annuli up to max_radius.
    Returns (radii_array, profile_array)
    """
    cx, cy = float(center_xy[0]), float(center_xy[1])
    ny, nx = image.shape
    if max_radius <= 0:
        return np.array([]), np.array([])
    ys, xs = np.indices(image.shape)
    r = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    radii = np.arange(1, int(max_radius) + 1)
    profile = []
    for ri in radii:
        mask = (r >= (ri - 0.5)) & (r < (ri + 0.5))
        vals = image[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            profile.append(np.nan)
        else:
            profile.append(np.mean(vals))
    return radii, np.asarray(profile, dtype='float32')

def pick_best_source_crowded(sources, crop_origin, target_xy, fwhm=8.0):
    """
    Pick the best source in crowded fields using multiple criteria:
    1. Proximity to click (weighted distance)
    2. Brightness (flux)
    3. Roundness (morphological quality)
    4. Sharpness (concentration)
    
    Uses weighted scoring to prefer:
    - Close sources (40% weight)
    - Bright sources (35% weight)
    - Round/point-like sources (15% weight)
    - Sharp sources (10% weight)
    
    Returns: (full_coords, crop_coords, source_index, score)
    """
    if sources is None or len(sources) == 0:
        return None, None, None, 0.0
    
    try:
        xcol = np.asarray(sources['xcentroid'], dtype=float)
        ycol = np.asarray(sources['ycentroid'], dtype=float)
        flux = np.asarray(sources['flux'], dtype=float)
        
        # Extract morphological metrics if available
        try:
            roundness = np.asarray(sources['roundness'], dtype=float)
        except:
            roundness = np.zeros_like(flux)
        
        try:
            sharpness = np.asarray(sources['sharpness'], dtype=float)
        except:
            sharpness = np.zeros_like(flux)
    except Exception:
        return None, None, None, 0.0
    
    if xcol.size == 0:
        return None, None, None, 0.0
    
    x0, y0 = crop_origin
    tx = float(target_xy[0]) - x0
    ty = float(target_xy[1]) - y0
    
    # Distances from click
    distances = np.sqrt((xcol - tx)**2 + (ycol - ty)**2)
    
    # Normalize metrics to [0, 1] range
    dist_norm = distances / (np.max(distances) + 1e-6)  # Lower is better
    flux_norm = flux / (np.max(flux) + 1e-6)             # Higher is better
    
    # Roundness: ideal is ~0, so penalize deviation
    round_norm = np.clip(np.abs(roundness), 0, 1)        # Lower is better
    
    # Sharpness: higher is better, clip to [0,1]
    sharp_norm = np.clip(sharpness, -1, 1)               # Higher is better
    sharp_norm = (sharp_norm + 1) / 2.0                  # Map [-1,1] -> [0,1]
    
    # Combined score (weighted):
    score = (0.40 * (1.0 - dist_norm) +        # Closer = higher
             0.35 * flux_norm +                 # Brighter = higher
             0.15 * (1.0 - round_norm) +        # Rounder = higher
             0.10 * sharp_norm)                 # Sharper = higher
    
    best_idx = int(np.argmax(score))
    best_score = float(score[best_idx])
    
    x_crop = float(xcol[best_idx])
    y_crop = float(ycol[best_idx])
    x_full = x0 + x_crop
    y_full = y0 + y_crop
    
    return (x_full, y_full), (x_crop, y_crop), best_idx, best_score

def detect_sources_in_crop(image, center_xy, crop_half_size=500, fwhm=5.0, threshold_sigma=3.0):
    """
    Crop around center_xy in full-image coords and run DAOStarFinder.
    Returns (sources_table or None, crop_origin (x0,y0), crop_image)
    """
    x, y = int(round(center_xy[0])), int(round(center_xy[1]))
    ny, nx = image.shape
    x0 = max(0, x - crop_half_size)
    x1 = min(nx, x + crop_half_size)
    y0 = max(0, y - crop_half_size)
    y1 = min(ny, y + crop_half_size)
    
    # Ensure crop has some minimum size
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None, (x0, y0), None
        
    crop = image[y0:y1, x0:x1].astype('float32')

    # Use robust stats calculation for local area
    try:
        # Mask NaNs and extremely high values (cosmics/saturation) for better std
        mask = np.isfinite(crop)
        if not np.any(mask): return None, (x0, y0), crop
        
        # Clip top-end outliers that might be stars to get true background noise
        mean, median, std = sigma_clipped_stats(crop, mask=~mask, sigma=3.0, maxiters=5)
        
        # If std is too low/zero, use a fallback
        if std <= 0 or not np.isfinite(std):
             std = np.nanstd(crop) if np.any(mask) else 1.0
             
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * max(std, 1e-12))
        sources = daofind(crop - median)
    except Exception:
        sources = None
    return sources, (x0, y0), crop

# ---------- detection + refinement orchestration ----------
def detect_then_refine(image, seed_xy, crop_half_size=300, fwhm=8.0, threshold_sigma=3.0,
                       stamp_radius=90, fwhm_guess=None, expand_steps=(0,10,20), use_crowded_mode=True):
    """
    1) Run DAOStarFinder on a crop around seed_xy.
    2) Pick the best DAO source (uses crowded-field aware selection if use_crowded_mode=True)
    3) Use find_star_in_frame (stamp-based fitting) around the DAO centroid.
    
    Returns: (found_full_xy (x,y) or None, refined_params dict or None, stamp_smooth or None, method_string)
    """
    if fwhm_guess is None:
        fwhm_guess = fwhm if fwhm is not None else DEFAULT_FWHM_GUESS

    # Step A: DAO on a localized crop
    # If explicitly requested for target area, we can use a smaller crop
    # but for detection we need some surrounding background
    sources, crop_origin, crop_img = detect_sources_in_crop(image, seed_xy,
                                                            crop_half_size=crop_half_size,
                                                            fwhm=fwhm, threshold_sigma=threshold_sigma)
    dao_xy = None
    if sources is not None and len(sources) > 0:
        if use_crowded_mode:
            nearest_full, nearest_crop, idx, score = pick_best_source_crowded(sources, crop_origin, seed_xy, fwhm=fwhm)
        else:
            nearest_full, nearest_crop, idx = pick_nearest_source(sources, crop_origin, seed_xy)
        
        if nearest_full is not None:
            # Check if it's within a reasonable distance of the click (localized)
            dist = np.sqrt((nearest_full[0] - seed_xy[0])**2 + (nearest_full[1] - seed_xy[1])**2)
            if dist <= crop_half_size:
                dao_xy = nearest_full

    # Step B: refine with stamp fitter:
    # If DAO failed to find anything in the crop, we shouldn't proceed to "refine-only" 
    # if the user expects strict localization to that stamp.
    if dao_xy is None:
        return None, None, None, 'not-found-in-crop'

    found, params, stamp = find_star_in_frame(image, dao_xy,
                                              stamp_radius=stamp_radius,
                                              fwhm_guess=fwhm_guess,
                                              expand_steps=expand_steps)
    if found is not None:
        return found, params, stamp, 'dao+refine'

    # fallback: try smaller centroid
    found2, params2, stamp2 = find_star_in_frame(image, dao_xy, stamp_radius=max(20, int(stamp_radius/2)),
                                                 fwhm_guess=fwhm_guess, expand_steps=(0,))
    if found2 is not None:
        return found2, params2, stamp2, 'dao+centroid-fallback'

    return None, None, None, 'not-found'
