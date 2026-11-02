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
from skimage.transform import AffineTransform, SimilarityTransform, warp
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from scipy.spatial import cKDTree as KDTree
import math

# No global warnings.filterwarnings('ignore') here. It used to sit at module
# scope, so merely importing this file silenced EVERY warning in the whole
# process -- numpy's 'invalid value in sqrt/divide', astropy's 'Input data
# contains invalid values (NaNs)', every deprecation -- which is exactly the
# class of signal utils.logging_config was introduced to preserve. Silence a
# specific warning at the call site that provokes it, never globally.

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

MIN_STELLAR_FWHM = 1.5      # px; below this it is a hot pixel or a cosmic ray
MAX_AXIS_RATIO = 3.0        # a star is round-ish; 15:1 is a fit running away
DEFAULT_MIN_SNR = 3.0       # kept for callers; see _validate_gaussian
SEED_OFFSET_FRACTION = 0.4  # a fit may land this far into the stamp, no further


def _validate_gaussian(fit, shape, noise, min_snr=DEFAULT_MIN_SNR,
                       fwhm_guess=None):
    """
    Is this fit a star-shaped thing where we looked, or fitter garbage?

    Nothing used to check. On pure Gaussian noise with no source at all,
    `find_star_in_frame` on a 600 px stamp returned a centre at x = -340 --
    off the array -- with a 125 px width, reported as a detection. The bulk
    worker then skipped alignment for that frame, seeded the next frame from
    it, and measured photometry there.

    What is checked, and what deliberately is NOT:

    * the centre must be inside the stamp it was fitted in;
    * the width must be stellar in absolute terms -- above a hot pixel, below
      a gradient across the whole stamp;
    * the axis ratio must be sane.

    What separates a real marginal source from a noise bump is neither of
    those, and it is not brightness either -- it is DISTANCE FROM THE SEED,
    which `_try_stamp` enforces. See the note there.

    There is NO brightness threshold, and that is deliberate.

    The source near (3138, 2242) on the GRB 22-Oct i frame is a 37.5-sigma
    matched-filter detection -- not marginal at all -- yet its FITTED amplitude
    comes out at only 1.9-2.4x the stamp's background rms, at every seed, stamp
    radius and FWHM guess tried. The reason is `estimate_background`: it takes
    the background from the stamp's EDGE, and a 17.5 px FWHM PSF still has real
    flux out there, so part of the star is counted as sky and the fitted
    amplitude undershoots (raw peak 150 ADU, fitted 82). A 3-sigma bar on that
    ratio therefore throws away strong, real sources.

    Nor does integrated significance work: over 40 pure-noise stamps it reaches
    83 while this source sits at 30, because the statistic rewards WIDE fits and
    a fitter finds broad shallow bumps in noise.

    Distance from the seed is what separates them; `_try_stamp` enforces it.

    `noise` and `min_snr` are still accepted so callers do not change, but only
    a non-positive or non-finite amplitude is rejected.
    """
    h, w = shape
    x, y = fit['x_mean'], fit['y_mean']
    if not (np.isfinite(x) and np.isfinite(y)):
        return False
    if not (0.0 <= x <= w - 1.0 and 0.0 <= y <= h - 1.0):
        return False

    sx, sy = abs(fit['x_std']), abs(fit['y_std'])
    if not (np.isfinite(sx) and np.isfinite(sy)) or min(sx, sy) <= 0:
        return False
    to_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
    fwhm_lo, fwhm_hi = sx * to_fwhm, sy * to_fwhm
    max_fwhm = 0.5 * min(h, w)
    if min(fwhm_lo, fwhm_hi) < MIN_STELLAR_FWHM or max(fwhm_lo, fwhm_hi) > max_fwhm:
        return False
    if max(sx, sy) / min(sx, sy) > MAX_AXIS_RATIO:
        return False

    amp = fit.get('amplitude')
    if amp is None or not np.isfinite(amp) or amp <= 0:
        return False
    return True


def twoD_Gaussian_fit(stamp, fwhm_guess=DEFAULT_FWHM_GUESS, guess_from=None):
    """
    Fit a 2D Gaussian + constant to the stamp using Astropy modeling.
    Returns dict with x_mean, y_mean (in stamp coords) or None on failure.

    `guess_from` is an optional smoothed copy used ONLY to locate the starting
    peak. The fit itself must run on the unsmoothed stamp: smoothing adds its
    own width in quadrature, sigma_fit^2 = sigma_true^2 + sigma_smooth^2, so
    fitting the smoothed copy returned ~19 px for a 15 px PSF. The centroid was
    unbiased either way, but the width was not -- and the width is what
    `_validate_gaussian` and the callers' PSF reporting rely on.
    """
    if stamp is None or stamp.size == 0:
        return None
    h, w = stamp.shape

    if np.all(~np.isfinite(stamp)):
        return None

    peak_src = guess_from if guess_from is not None else stamp
    yy, xx = np.mgrid[0:h, 0:w]
    med = float(np.nanmedian(stamp))
    amp_guess = max(float(np.nanmax(stamp) - med), 1.0)

    try:
        max_idx = np.unravel_index(np.nanargmax(peak_src), peak_src.shape)
    except ValueError:
        # "All-NaN slice encountered" - stamp has no finite values
        return None

    y0_guess, x0_guess = float(max_idx[0]), float(max_idx[1])
    sigma_guess = fwhm_guess / (2.0 * np.sqrt(2 * np.log(2)))
    g_init = models.Gaussian2D(amplitude=amp_guess,
                               x_mean=x0_guess, y_mean=y0_guess,
                               x_stddev=max(0.5, sigma_guess), y_stddev=max(0.5, sigma_guess), theta=0.0)
    c_init = models.Const2D(med)
    compound = g_init + c_init

    fitter = fitting.LevMarLSQFitter()
    finite = np.isfinite(stamp)
    try:
        if finite.all():
            g = fitter(compound, xx, yy, stamp, maxiter=200)
        else:
            # NaNs (warped frame edges) make LevMar return NaN parameters.
            g = fitter(compound, xx[finite], yy[finite], stamp[finite], maxiter=200)
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

def disk_centroid(image, xy, radius, sky):
    """
    Light-weighted centre of the pixels whose centres lie within `radius` of
    xy, after subtracting `sky`; pixels at or below the sky count as zero and
    NaNs are ignored. One pass, no iteration (see the bulk worker's
    _recentre). Returns (x, y) in image coordinates, or None when nothing in
    the disk is above the sky.
    """
    ny, nx = image.shape
    x0, y0 = float(xy[0]), float(xy[1])
    R = int(math.ceil(float(radius)))
    xa, xb = max(0, int(round(x0)) - R), min(nx, int(round(x0)) + R + 1)
    ya, yb = max(0, int(round(y0)) - R), min(ny, int(round(y0)) + R + 1)
    if xb <= xa or yb <= ya:
        return None
    sub = np.asarray(image[ya:yb, xa:xb], dtype=float) - float(sky)
    yy, xx = np.mgrid[ya:yb, xa:xb]
    inside = (xx - x0) ** 2 + (yy - y0) ** 2 <= float(radius) ** 2
    use = inside & np.isfinite(sub) & (sub > 0)
    w = sub[use]
    s = w.sum()
    if not s > 0:
        return None
    return float((w * xx[use]).sum() / s), float((w * yy[use]).sum() / s)

def _try_stamp(frame, cx, cy, rad, fwhm_guess, min_snr=DEFAULT_MIN_SNR):
    """
    Measure one stamp centred on (cx, cy). Returns (xy, params, smoothed) or
    (None, None, None) when nothing in it is a credible star.

    The Gaussian is fitted to the UNSMOOTHED background-subtracted stamp and
    seeded from the smoothed copy, and the result must pass
    `_validate_gaussian` before it is believed.
    """
    stamp, x0, y0 = cut_stamp(frame, cx, cy, rad)
    if stamp.size == 0:
        return None, None, None

    bg, bgstd = estimate_background(stamp)
    stamp_bs = stamp - bg
    stamp_s = gaussian_filter(np.nan_to_num(stamp_bs, nan=0.0),
                              sigma=max(0.8, fwhm_guess / 3.0))

    fit = twoD_Gaussian_fit(stamp_bs, fwhm_guess, guess_from=stamp_s)
    if fit is not None and _validate_gaussian(fit, stamp_bs.shape, bgstd,
                                              min_snr=min_snr):
        fx, fy = x0 + fit['x_mean'], y0 + fit['y_mean']
        # The fit must land NEAR the seed. This is what tells a real source from
        # a bump the fitter found in noise, and no brightness bar can do it
        # (see _validate_gaussian): measured over 60 pure-noise stamps the fits
        # land a median 43 px from the seed and only 1 in 60 within 10 px, while
        # the real source on the GRB 22-Oct i frame lands 6.6 px from the click
        # -- the same 6.6 px whether the FWHM guess is 5, 14 or 17.5.
        # Scaling with the stamp keeps the caller in charge: a small
        # "detection stamp" means refine in place, a large tracking stamp
        # means a large drift is allowed.
        max_offset = max(SEED_OFFSET_FRACTION * float(rad), 2.0 * MIN_STELLAR_FWHM)
        if math.hypot(fx - cx, fy - cy) <= max_offset:
            return (fx, fy), fit, stamp_s

    # No centre-of-mass fallback. It returns a position with no width, no
    # amplitude and no shape, so there is nothing to validate -- and it was the
    # sole reason blank noise and single hot pixels still came back as
    # detections after the Gaussian checks were added. Measured on the four
    # cases the checks were written for, the Gaussian decides every one
    # correctly on its own:
    #   pure noise    fail (axis ratio 7.2)      hot pixel  fail (fwhm 0.4 px)
    #   star at 10x   pass (fwhm 15.1, 15.4)     star at 2x fail (amp < 3 sigma)
    # A star the Gaussian cannot fit at any of the three stamp radii is not one
    # to hang photometry on.
    return None, None, None


def find_star_in_frame(frame, expected_xy, stamp_radius=DEFAULT_STAMP_RADIUS,
                       fwhm_guess=DEFAULT_FWHM_GUESS, expand_steps=(0,10,20),
                       max_search_radius=200, min_snr=DEFAULT_MIN_SNR):
    """
    Locate the star around expected_xy using progressively larger local stamps.
    Returns (found_xy (full-frame coords), measured_parameters, stamp_smoothed)
    or (None, None, None).

    "or (None, None, None)" used to be unreachable in practice: every fit was
    accepted on `fit is not None`, so a blank field returned a position anyway.
    Now each candidate has to look like a star before it is returned.
    """
    cx, cy = expected_xy

    # Search area. `expand_steps` is the caller's decision and is now obeyed:
    # a caller passing (0,) means ONE stamp and no growing.
    #
    # There used to be an unconditional extra rung at stamp_radius + 100 after
    # the loop, so the bulk worker's "strict, no jumping" refinement -- which
    # passes expand_steps=(0,) precisely to stay put -- quietly searched a
    # 261x261 stamp anyway. Measured on the GRB 22-Oct i frame: a candidate
    # 7 px from the click came back "refined" to 104 px away, on a brighter
    # star entirely.
    radii = [stamp_radius]
    for step in expand_steps:
        if step > 0:
            new_r = min(max_search_radius, stamp_radius + step)
            if new_r not in radii:
                radii.append(new_r)

    for rad in radii:
        found, params, stamp_s = _try_stamp(frame, cx, cy, rad, fwhm_guess,
                                            min_snr=min_snr)
        if found is not None:
            return found, params, stamp_s
    return None, None, None


def find_star_adaptive(frame, expected_xy, previous_detected_xy=None,
                       stamp_radius=DEFAULT_STAMP_RADIUS,
                       fwhm_guess=DEFAULT_FWHM_GUESS, max_search_radius=200,
                       min_snr=DEFAULT_MIN_SNR):
    """
    Adaptive source detection that uses frame-to-frame position tracking.

    If previous_detected_xy is provided (from previous frame), predicts new
    position based on drift and searches around that. Falls back to fixed-
    position search.

    Returns (found_xy, measured_parameters, stamp_smoothed) or (None, None, None).
    """
    cx, cy = expected_xy

    # Step 1: If we have previous position, search there. (The old code
    # computed drift = prev - expected and then searched at
    # expected + drift, which is prev exactly; same behaviour, less noise.)
    if previous_detected_xy is not None:
        px, py = previous_detected_xy
        search_radii = [
            stamp_radius,
            min(max_search_radius, stamp_radius + 50),
            min(max_search_radius, stamp_radius + 150),
        ]
        for rad in search_radii:
            found, params, stamp_s = _try_stamp(frame, px, py, rad, fwhm_guess,
                                                min_snr=min_snr)
            if found is not None:
                return found, params, stamp_s

    # Step 2: Fall back to standard fixed-position search
    return find_star_in_frame(frame, expected_xy, stamp_radius=stamp_radius,
                              fwhm_guess=fwhm_guess,
                              max_search_radius=max_search_radius,
                              min_snr=min_snr)

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

def _coarse_offset(src_pts, ref_pts, bin_size=8.0, max_shift=600.0):
    """
    Vote for the frame-to-frame shift using EVERY pair of sources.

    This exists because nearest-neighbour matching in raw coordinates cannot
    find the shift it is supposed to measure: it only pairs stars already
    within `match_radius` of each other, so beyond ~10 px the true
    counterparts are all rejected and the median is taken over chance
    coincidences. Measured before this was added, on a synthetic 150-star
    field, a true 60 px shift was reported as dx=-1.5, dy=-1.6 with 2
    "inliers" and no error raised. INO340 frames drift ~350 px a night.

    The offset between the same star in two frames is the same vector for
    every star, so the true shift is a sharp peak in the 2-D histogram of all
    N*M pairwise offsets while false pairs spread out. Returns (dx, dy), or
    None when no peak stands out.
    """
    if src_pts.size == 0 or ref_pts.size == 0:
        return None

    # All pairwise offsets, restricted to physically plausible shifts.
    dxy = (ref_pts[None, :, :] - src_pts[:, None, :]).reshape(-1, 2)
    keep = (np.abs(dxy[:, 0]) <= max_shift) & (np.abs(dxy[:, 1]) <= max_shift)
    dxy = dxy[keep]
    if dxy.shape[0] < 3:
        return None

    nb = max(1, int(np.ceil(2.0 * max_shift / bin_size)))
    rng = [[-max_shift, max_shift], [-max_shift, max_shift]]
    hist, xe, ye = np.histogram2d(dxy[:, 0], dxy[:, 1], bins=nb, range=rng)
    ix, iy = np.unravel_index(int(np.argmax(hist)), hist.shape)
    if hist[ix, iy] < 3:
        return None

    peak = np.array([0.5 * (xe[ix] + xe[ix + 1]), 0.5 * (ye[iy] + ye[iy + 1])])
    # The peak bin only localises the shift to +/- bin_size; take the median of
    # the offsets around it so the coarse estimate is good to ~1 px.
    near = dxy[np.all(np.abs(dxy - peak) <= bin_size, axis=1)]
    if near.shape[0] < 3:
        return None
    return np.median(near, axis=0)


def _match_near_offset(src_pts, ref_pts, offset, match_radius=8.0):
    """Pair src to ref after applying `offset`, keeping pairs within match_radius."""
    shifted = src_pts + np.asarray(offset, dtype=float)
    dists, idxs = KDTree(ref_pts).query(shifted, k=1)
    dists = np.atleast_1d(dists)
    idxs = np.atleast_1d(idxs)
    keep = dists <= match_radius
    if not np.any(keep):
        return np.empty((0, 2)), np.empty((0, 2))
    return src_pts[keep], ref_pts[idxs[keep]]


def _estimate_one(src, dst, similarity):
    """Fit either a similarity or a pure translation to matched point pairs."""
    if similarity:
        t = SimilarityTransform()
        if not t.estimate(src, dst):
            raise ValueError("similarity fit did not converge")
        return t
    return SimilarityTransform(translation=np.median(dst - src, axis=0))


def _fit_transform(src_matched, ref_matched, max_rms=2.0, min_inliers=4,
                   min_for_similarity=8):
    """
    Fit the frame-to-frame transform, using only as many free parameters as the
    matched stars can actually constrain.

    A similarity (translation + rotation + scale, 4 parameters) is what the
    INO340 field needs -- it rotates ~1.8 deg per night, which is 63 px at the
    corner of a 4096 frame that a translation-only fit smears rather than
    removes. But some fields simply do not contain enough stars to constrain a
    rotation: the GRB 251013C g-band Low frames yield 45 detections of which
    only ~5 are real stars (fluxes 20874, 10033, 7520, 3331, 2039, then a cliff
    to 345). Fitting a rotation to five points there would invent one. So below
    `min_for_similarity` pairs this drops to translation only and says so in
    `info['model']`, rather than either failing or over-fitting.

    Returns (tform, n_inliers, rms, model). Raises ValueError when the result
    fails the acceptance criteria -- the whole point being that a bad alignment
    must FAIL rather than return a plausible-looking wrong answer.
    """
    n0 = int(src_matched.shape[0])
    if n0 < min_inliers:
        raise ValueError(f"only {n0} matched sources (need {min_inliers})")

    similarity = n0 >= min_for_similarity
    src, dst = src_matched, ref_matched
    tform = None
    for _ in range(3):
        tform = _estimate_one(src, dst, similarity)
        resid = np.linalg.norm(tform(src) - dst, axis=1)
        # Reject on a ROBUST scale, not on the rms. With a handful of stars a
        # single mismatched pair inflates the rms enough to keep itself: on the
        # GRB 251013C frame at 00:54:39 the residuals were
        # [0.23 0.54 0.99 1.16 1.38 7.17], rms 3.05, and a 3*rms cut sat at
        # 9.2 px -- so the outlier survived and the frame was rejected for a
        # rms IT had caused. median + 3*1.4826*MAD puts the cut at 3.0 px and
        # drops it.
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        keep = resid <= max(med + 3.0 * 1.4826 * mad, 1.0)
        if keep.sum() < min_inliers or keep.all():
            break
        src, dst = src[keep], dst[keep]
        similarity = src.shape[0] >= min_for_similarity

    resid = np.linalg.norm(tform(src) - dst, axis=1)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    n = int(src.shape[0])
    if n < min_inliers:
        raise ValueError(f"only {n} inliers survived (need {min_inliers})")
    if rms > max_rms:
        raise ValueError(f"residual rms {rms:.2f} px exceeds {max_rms:.2f} px")
    return tform, n, rms, ("similarity" if similarity else "translation")


def estimate_similarity(data, ref_data, fwhm=8.0, threshold_sigma=5.0,
                        max_sources=500, match_radius=8.0, max_shift=600.0,
                        min_inliers=4, min_for_similarity=8, max_rms=2.0):
    """
    Solve for the transform taking `data` onto `ref_data`.

    Three stages, each of which fails loudly:
      1. detect sources in both frames,
      2. vote for the coarse shift over all pairwise offsets,
      3. match around that shift and fit a transform.

    Returns (tform, info); info carries model, n_sources_src/ref, n_matched,
    n_inliers, rms, dx, dy, rotation_deg and scale as NUMBERS, so a caller can
    record or threshold them instead of parsing a method string.
    """
    src_pts = _detect_sources_global(data, fwhm=fwhm,
                                     threshold_sigma=threshold_sigma,
                                     max_sources=max_sources)
    ref_pts = _detect_sources_global(ref_data, fwhm=fwhm,
                                     threshold_sigma=threshold_sigma,
                                     max_sources=max_sources)
    if src_pts.shape[0] < min_inliers or ref_pts.shape[0] < min_inliers:
        raise ValueError(f"too few sources detected "
                         f"(src {src_pts.shape[0]}, ref {ref_pts.shape[0]})")

    offset = _coarse_offset(src_pts, ref_pts, max_shift=max_shift)
    if offset is None:
        raise ValueError("no coarse shift peak found")

    src_m, ref_m = _match_near_offset(src_pts, ref_pts, offset,
                                      match_radius=match_radius)
    tform, n_inliers, rms, model = _fit_transform(
        src_m, ref_m, max_rms=max_rms, min_inliers=min_inliers,
        min_for_similarity=min_for_similarity)

    info = dict(model=model,
                n_sources_src=int(src_pts.shape[0]),
                n_sources_ref=int(ref_pts.shape[0]),
                n_matched=int(src_m.shape[0]),
                n_inliers=n_inliers, rms=rms,
                coarse_dx=float(offset[0]), coarse_dy=float(offset[1]),
                dx=float(tform.translation[0]), dy=float(tform.translation[1]),
                rotation_deg=float(np.degrees(tform.rotation)),
                scale=float(tform.scale))
    return tform, info


def ransac_align(data, ref_data, fwhm=8.0, threshold_sigma=5.0, max_sources=500,
                 match_radius=8.0, max_shift=600.0, min_inliers=4,
                 min_for_similarity=8, max_rms=2.0):
    """
    Register `data` onto `ref_data` and return (warped, method_string).

    Raises ValueError when the transform cannot be trusted, so the caller falls
    through to another method instead of silently measuring a frame that was
    never aligned.
    """
    tform, info = estimate_similarity(
        data, ref_data, fwhm=fwhm, threshold_sigma=threshold_sigma,
        max_sources=max_sources, match_radius=match_radius,
        max_shift=max_shift, min_inliers=min_inliers,
        min_for_similarity=min_for_similarity, max_rms=max_rms)

    warped = warp(data, inverse_map=tform.inverse, output_shape=ref_data.shape,
                  preserve_range=True, cval=np.nan).astype(np.float32)
    method = (f"{info['model']} dx={info['dx']:.2f},dy={info['dy']:.2f},"
              f"rot={info['rotation_deg']:.3f}deg,inliers={info['n_inliers']},"
              f"rms={info['rms']:.2f}px")
    return warped, method


# ---------------- field transform: where is a star in another frame? ----------------

def field_stars(image, fwhm, threshold_sigma=5.0, max_sources=300):
    """
    Positions (full-resolution pixels, brightest first) of the stars that pin
    down a frame's geometry, as an (N, 2) array of (x, y).

    Detected on a block-averaged copy, b = fwhm // 3, so the binned PSF is
    still ~3 px wide: a full-resolution DAOStarFinder pass with a 12 px kernel
    took 5.9 s on a 4096^2 PSR J2215 frame, the 4x4-binned one 0.15 s, and the
    transform fitted from it left 0.33-0.42 px rms over ~270 stars. Positions
    only seed the transform; nothing is measured on the binned copy.
    """
    img = np.asarray(image, dtype=float)
    b = max(1, int(float(fwhm) // 3))
    ny, nx = img.shape
    small = img[:ny // b * b, :nx // b * b].reshape(ny // b, b, nx // b, b).mean(axis=(1, 3))
    _, med, std = sigma_clipped_stats(small, sigma=3.0, maxiters=5)
    if not (np.isfinite(std) and std > 0):
        return np.empty((0, 2))
    srcs = DAOStarFinder(fwhm=float(fwhm) / b, threshold=float(threshold_sigma) * std,
                         exclude_border=True)(small - med)
    if srcs is None or len(srcs) == 0:
        return np.empty((0, 2))
    order = np.argsort(-np.asarray(srcs['flux'], dtype=float))[:int(max_sources)]
    x = np.asarray(srcs['xcentroid'], dtype=float)[order]
    y = np.asarray(srcs['ycentroid'], dtype=float)[order]
    if b > 1:
        x, y = (x + 0.5) * b - 0.5, (y + 0.5) * b - 0.5
    return np.column_stack((x, y))


def _vote_offset(src_pts, ref_pts, tol=8.0, max_shift=600.0, min_votes=3):
    """
    The shift most pairwise offsets agree on, counted within `tol` of each
    candidate rather than in fixed bins. _coarse_offset's histogram can split
    the true pairs over two neighbouring bins, and with only five shared stars
    (SA 92 2x2, frames taken after 23:42) no bin then reached three votes.
    Returns (dx, dy) or None.
    """
    if src_pts.size == 0 or ref_pts.size == 0:
        return None
    dxy = (ref_pts[None, :, :] - src_pts[:, None, :]).reshape(-1, 2)
    dxy = dxy[(np.abs(dxy[:, 0]) <= max_shift) & (np.abs(dxy[:, 1]) <= max_shift)]
    if dxy.shape[0] < min_votes:
        return None
    counts = KDTree(dxy).query_ball_point(dxy, r=tol, return_length=True)
    k = int(np.argmax(counts))
    if counts[k] < min_votes:
        return None
    near = dxy[np.hypot(*(dxy - dxy[k]).T) <= tol]
    return np.median(near, axis=0)


def _rematch_and_fit(src_pts, ref_pts, guess, match_radius=8.0, max_rms=2.0,
                     min_inliers=4, min_for_similarity=5):
    """
    Pair every frame star with the nearest reference star under `guess`
    (each reference star claimed once, by its nearest), fit, and repeat three
    times so a rotation the guess got slightly wrong is pulled in. A guess
    that leaves too few pairs within match_radius is retried with a first
    match two and three times wider. Raises ValueError like _fit_transform.

    max_rms judges the converged (third) refit only. Applied to every refit, it
    rejected PSR J2215 High_30: the vote guess is a pure translation, its first
    refit (rotation not yet pulled in) left 2.78 px, the third would have left
    0.71 px over 154 stars, and the frame fell back to "last position".
    """
    tree = KDTree(ref_pts)
    last = None
    for first_radius in (match_radius, 2 * match_radius, 3 * match_radius):
        tform = guess
        try:
            for it in range(3):
                d, j = tree.query(tform(src_pts))
                keep = np.zeros(len(src_pts), dtype=bool)
                claimed = set()
                for k in np.argsort(d):
                    if d[k] <= (first_radius if it == 0 else match_radius) and j[k] not in claimed:
                        claimed.add(j[k])
                        keep[k] = True
                tform, n, rms, model = _fit_transform(
                    src_pts[keep], ref_pts[j[keep]], max_rms=(max_rms if it == 2 else np.inf),
                    min_inliers=min_inliers, min_for_similarity=min_for_similarity)
            return tform, n, rms, model
        except ValueError as e:
            last = e
    raise last


def field_transform(src_pts, ref_pts, match_radius=8.0, max_rms=2.0,
                    min_inliers=4, min_for_similarity=5):
    """
    Transform taking a frame's field_stars onto the reference frame's, so
    that `tform.inverse(ref_xy)` is where a reference position lies in the
    frame. Returns (tform, info); raises ValueError naming why each starting
    guess failed.

    Two starting guesses, then the same refit (_rematch_and_fit):

    * astroalign's triangle matching of the 50 brightest stars, which does
      not care about rotation. Over a whole PSR J2215 night (drift up to
      300 px, rotation up to 1.6 deg) estimate_similarity's vote-then-match
      lost 6 of 13 frames (rms 2.1-4.6 px: the match after a translation
      vote misses stars a rotation has moved); with this guess all 13 fit
      at 0.39-0.87 px.
    * a translation vote (_vote_offset), for fields too sparse for triangles.

    A similarity is fitted from five pairs up, not eight as in
    _fit_transform's default: these fields really do rotate. On GRB 251013C
    Low (7 shared stars) a translation left 1.46 px rms and a similarity
    0.37 px (rotation 0.06 deg in 37 min); on SA 92 2x2 (5 stars) 1.5-2.0 px
    against 0.35-0.46 px, and only the similarity matched the frames taken
    2.5 h later (rotation 0.5 deg).
    """
    src_pts = np.asarray(src_pts, dtype=float).reshape(-1, 2)
    ref_pts = np.asarray(ref_pts, dtype=float).reshape(-1, 2)
    if len(src_pts) < min_inliers or len(ref_pts) < min_inliers:
        raise ValueError(f"too few stars (frame {len(src_pts)}, reference {len(ref_pts)})")
    import astroalign as aa
    why, guesses = [], []
    try:
        T, _ = aa.find_transform(src_pts[:50], ref_pts[:50])
        guesses.append(("triangles", SimilarityTransform(matrix=np.asarray(T.params))))
    except Exception as e:
        why.append(f"triangles: {type(e).__name__}")
    off = _vote_offset(src_pts, ref_pts, tol=match_radius)
    if off is not None:
        guesses.append(("vote", SimilarityTransform(translation=off)))
    else:
        why.append("vote: no shift that three pairs agree on")
    for name, guess in guesses:
        try:
            tform, n, rms, model = _rematch_and_fit(
                src_pts, ref_pts, guess, match_radius=match_radius, max_rms=max_rms,
                min_inliers=min_inliers, min_for_similarity=min_for_similarity)
        except ValueError as e:
            why.append(f"{name}: {e}")
            continue
        return tform, dict(guess=name, model=model, n_inliers=n, rms=rms,
                           dx=float(tform.translation[0]), dy=float(tform.translation[1]),
                           rotation_deg=float(np.degrees(tform.rotation)),
                           scale=float(tform.scale))
    raise ValueError("; ".join(why))


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
