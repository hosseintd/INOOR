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
from .camera_config import DEFAULT_MODE, ADC_CEILING_ADU

# constants
DEFAULT_FWHM_GUESS = 15
DEFAULT_STAMP_RADIUS = 90
# mag_err = MAG_ERR_FACTOR / SNR; exact value is 2.5/ln(10) = 1.0857...
MAG_ERR_FACTOR = 2.5 / math.log(10)

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

def _crop_around(image, center_xy, half_size):
    """
    Cut a square region of half-width `half_size` (pixels) around center_xy.

    Returns (crop, cx_local, cy_local). Photometry on the crop is IDENTICAL to
    photometry on the full frame as long as every aperture used stays inside
    the crop: apertures that overrun an edge are summed over their in-bounds
    part either way, and CircularAperture.area is analytic (pi*r**2), not
    data-dependent. Cropping exists purely so we stop touching all 16.7M
    pixels of a 4096^2 KL4040 frame to measure a star that occupies ~1e4 of
    them.
    """
    ny, nx = image.shape
    h = int(math.ceil(half_size)) + 2  # +2 px guard band for interpolation edges
    cx, cy = float(center_xy[0]), float(center_xy[1])
    x0 = max(0, int(math.floor(cx)) - h)
    x1 = min(nx, int(math.ceil(cx)) + h + 1)
    y0 = max(0, int(math.floor(cy)) - h)
    y1 = min(ny, int(math.ceil(cy)) + h + 1)
    if x1 <= x0 or y1 <= y0:
        return None, cx, cy
    return image[y0:y1, x0:x1], cx - x0, cy - y0


# Sky estimator shared by every annulus INOOR measures through: the aperture
# photometry, the SNR curve that picks the radius, and the curve of growth.
SKY_CLIP_SIGMA = 3.0
SKY_CLIP_MAXITERS = 10


def clipped_sky_mean(values, sigma=SKY_CLIP_SIGMA, maxiters=SKY_CLIP_MAXITERS):
    """
    Sky level from annulus pixels: the mean after iterative sigma clipping
    about the median (astropy's sigma_clip defaults: centre = median, width =
    standard deviation, symmetric limits), repeated until nothing more is
    rejected. Returns (mean, n_used); (nan, 0) when no finite pixel is left.

    A plain mean lets one cosmic ray, or a neighbour's core, raise the sky
    under the whole aperture. On PSR J2215 frame 86 a faint star whose 5x
    brighter neighbour sits 35 px away -- inside 2r-3r at r = 12 px -- read
    0.80 mag too faint with the plain mean; clipping rejected 190 of 2265
    pixels and recovered it, while isolated stars moved by 0.7-2.3 mmag. Not
    a median: calibrated frames are stored as integers, where a median sits on
    an integer level, and a median sky was 25% worse on the PSR series.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    for _ in range(int(maxiters)):
        if v.size == 0:
            break
        keep = np.abs(v - np.median(v)) <= sigma * v.std()
        if keep.all():
            break
        v = v[keep]
    if v.size == 0:
        return float('nan'), 0
    return float(v.mean()), int(v.size)


def _annulus_skies(arr, x, y, r_in, r_out):
    """
    clipped_sky_mean over many annuli centred on (x, y) in `arr`, one per
    (r_in[i], r_out[i]). A pixel belongs to an annulus when its centre does,
    r_in <= r < r_out. Pixels are sorted by distance once, so each annulus is
    a slice: the 200 annuli of an SNR curve cost one sort, not 200 masks.
    Returns a list of (mean, n_used).
    """
    ny, nx = arr.shape
    yy, xx = np.indices((ny, nx))
    rr = np.hypot(xx - float(x), yy - float(y)).ravel()
    order = np.argsort(rr, kind='stable')
    rs = rr[order]
    vs = np.asarray(arr, dtype=float).ravel()[order]
    out = []
    for a, b in zip(np.atleast_1d(r_in), np.atleast_1d(r_out)):
        i0 = int(np.searchsorted(rs, float(a), side='left'))
        i1 = int(np.searchsorted(rs, float(b), side='left'))
        out.append(clipped_sky_mean(vs[i0:i1]))
    return out


def annulus_sky(image, center_xy, r_in, r_out):
    """Sky per pixel in one annulus, as (mean, n_used); see clipped_sky_mean.
    Pixels off the image edge are simply absent, so an annulus running off
    the frame is averaged over the pixels it has."""
    crop, cx, cy = _crop_around(image, center_xy, float(r_out) + 1.0)
    if crop is None:
        return float('nan'), 0
    return _annulus_skies(crop, cx, cy, [r_in], [r_out])[0]


def compute_snr_vs_radius(image, center_xy, fwhm=12.0, radii=None,
                          read_noise=DEFAULT_MODE.read_noise, dark_noise=DEFAULT_MODE.dark_noise,
                          gain=DEFAULT_MODE.gain, sample_max_pixels=200000,
                          inner_coef=2.0, outer_coef=3.0):
    """
    Compute SNR vs aperture radius around center_xy.
    Returns radii array, snr array, best_radius, mag_error.

    The sky annulus is [inner_coef*r, outer_coef*r] -- the SAME one
    perform_aperture_photometry will use. It used to be [r, max(r+1, 1.5r)],
    so the radius was optimised against a background model that the actual
    measurement never used, and at r ~ FWHM that inner edge sat inside the
    star's own wings: the sky came out too high, the net flux too low, and the
    SNR curve peaked early. On the GRB 22-Oct i-band frames the optimum moved
    from ~4 px to a radius comparable with the PSF once the two agreed. The sky
    in each annulus is the measurement's own estimator, clipped_sky_mean.

    sample_max_pixels is accepted for older callers and no longer used.
    """
    if radii is None:
        max_r = max(40, 5 * fwhm)
        radii = np.linspace(1, max_r, 200)

    radii = np.asarray(radii, dtype=float)

    # Work on a crop large enough to contain the largest annulus we evaluate
    # (r_out = outer_coef * r). The max(60, 5*fwhm) floor is left over from the
    # HWHM blend, which is gone; it never binds for the default radius grid and
    # is kept only so results stay bit-identical for callers passing short grids.
    prof_max_radius = max(60, 5 * fwhm)
    if radii.size:
        needed = max(float(np.max(radii)) * float(outer_coef) + 1.0, prof_max_radius)
    else:
        needed = prof_max_radius
    crop, x, y = _crop_around(image, center_xy, needed)
    if crop is None:
        crop, x, y = image, float(center_xy[0]), float(center_xy[1])
    arr = crop

    # Evaluate every radius in ONE aperture_photometry call. photutils accepts a
    # list of apertures and reuses the same bounding-box/overlap machinery, so
    # the per-radius sums are bit-identical to calling it once per radius --
    # it just stops paying the ~200x fixed per-call overhead (Table
    # construction, input validation, WCS handling).
    n_r = radii.size
    aps = [CircularAperture((x, y), r=float(r)) for r in radii]
    try:
        tab = aperture_photometry(arr, aps)
        ap_sums = [float(tab[f'aperture_sum_{i}'][0]) for i in range(n_r)]
    except Exception:
        ap_sums = None
    # SAME annulus rule and SAME sky estimator as the measurement itself.
    r_ins = radii * float(inner_coef)
    skies = _annulus_skies(arr, x, y, r_ins,
                           np.maximum(r_ins + 1.0, radii * float(outer_coef)))

    snrs = []
    for i, r in enumerate(radii):
        ap = aps[i]
        try:
            if ap_sums is None:
                raise RuntimeError("batched aperture photometry failed")
            source_flux = ap_sums[i]
            bkg_per_pix = skies[i][0]
            n_ap = ap.area if hasattr(ap,'area') else math.pi*(r**2)
            total_bkg = bkg_per_pix * n_ap
            net_signal_counts = (source_flux - total_bkg)
            # Standard CCD/CMOS aperture-photometry noise equation (Newberry 1991;
            # Mighell 1999; as implemented by AstroImageJ). net_signal stays in ADU
            # (no gain division): the noise_sq below is already expressed in the
            # matching ADU^2 domain (source_flux/gain is the Poisson variance of
            # every photon in the aperture -- source+sky -- converted from
            # electrons^2 to ADU^2; read_noise/dark_noise are already in
            # electrons, so they need gain**2, not gain**1, for the same
            # conversion). Dividing net_signal by gain again would be a spurious
            # extra factor of gain in the denominator's units.
            net_signal = net_signal_counts
            noise_sq = (source_flux / gain) + (((read_noise**2) + dark_noise) * n_ap) / (gain**2)
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

    # The recommended radius IS the SNR peak. It used to be (3*HWHM + peak)/2,
    # an ad-hoc blend with no published basis. On 139 PSR J2215 frames the
    # plain peak, taken as k = r_peak / frame FWHM and shared by all stars,
    # lands within 1-3% of the time-series optimum in both gain modes.
    snr_at_best = float(snrs[idx_max])
    mag_error = MAG_ERR_FACTOR / snr_at_best if snr_at_best > 0 else np.inf
    return np.asarray(radii), snrs, r_best_snr, mag_error

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
                                read_noise=DEFAULT_MODE.read_noise, dark_noise=DEFAULT_MODE.dark_noise,
                                gain=DEFAULT_MODE.gain,
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
        
    try:
        phot = aperture_photometry(image, ap)
        source_flux = float(phot['aperture_sum'][0])
        # Same sky estimator as the SNR curve that chose the radius
        # (clipped_sky_mean). NaN when the annulus holds no finite pixel, which
        # makes the result a non-detection below instead of a zero sky.
        bkg_mean, _ = annulus_sky(image, center_xy, r_in, r_out)
        ap_area = ap.area if hasattr(ap,'area') else math.pi * aperture_radius**2
        total_bkg = bkg_mean * ap_area
        net_counts = source_flux - total_bkg
        # Defensive check: exptime must be positive and reasonable (>= 1e-6 s = 1 microsecond)
        # Very small exposure times (bias frames at 30µs) are treated as invalid
        if exptime is None or exptime <= 0 or not np.isfinite(exptime):
            instr_mag = None
            mag = None
        elif net_counts <= 0 or not np.isfinite(net_counts):
            # Not a detection. This used to be clamped with
            # max(1e-9, net_counts/exptime), which turns any non-detection into
            # a magnitude of about +22.5 -- a real-looking number that lands on
            # a light curve indistinguishable from a measurement.
            instr_mag = None
            mag = None
        else:
            instr_mag = -2.5 * math.log10(net_counts / exptime)
            mag = instr_mag + zeropoint
        n_ap = ap_area
        # Standard CCD/CMOS aperture-photometry noise equation (Newberry 1991;
        # Mighell 1999; as implemented by AstroImageJ). net_signal stays in ADU
        # (no gain division): noise_sq below is already expressed in the matching
        # ADU^2 domain (source_flux/gain is the Poisson variance of every photon
        # in the aperture -- source+sky -- converted from electrons^2 to ADU^2;
        # read_noise/dark_noise are already in electrons, so they need gain**2,
        # not gain**1, for the same conversion). Dividing net_signal by gain
        # again would be a spurious extra factor of gain in the denominator.
        noise_sq = (source_flux / gain) + (((read_noise**2) + dark_noise) * n_ap) / (gain**2)
        noise = math.sqrt(noise_sq) if noise_sq > 0 else 1e-12
        net_signal = net_counts
        snr = net_signal / noise if noise > 0 else 0.0
        mag_err = MAG_ERR_FACTOR / snr if snr > 0 else np.inf
        return dict(instr_mag=instr_mag, mag=mag, mag_err=mag_err, flux=net_counts,
                    bkg_mean=bkg_mean, aperture_area=ap_area, snr=snr)
    except Exception:
        # 'mag' must be present: the success branch has it and callers index it
        # unconditionally, so omitting it raised KeyError('mag') out of a Qt slot
        # with no handler, and surfaced in the bulk worker as the useless message
        # "Processing failed: 'mag'".
        return dict(instr_mag=None, mag=None, mag_err=np.inf, flux=0.0,
                    bkg_mean=0.0, aperture_area=0.0, snr=0.0,
                    aperture_correction=0.0)

def profile_max_radius(r_used):
    """
    How far out the plotted radial profile extends, as a function of the
    aperture actually used.

    Defined in ONE place because three call sites need to agree: the
    single-click DetectionWorker, the live re-measure (_measure_frame) and the
    bulk worker. They previously did not: the single-click path passed a
    hardcoded placeholder radius of 10.0 (its pre-SNR guess) instead of the
    final aperture, so its profile always spanned 30 px while the other two
    spanned ~3*r_best. Same star, same frame, visibly different plot.
    """
    return max(30.0, float(r_used) * 3.0)


# ---------------------------------------------------------------------------
# Aperture-radius policies
#
# Which radius you measure through is not a free choice: the reported magnitude
# carries an aperture-loss term -2.5*log10(E(r)), where E(r) is the fraction of
# the star's light inside r. That term only cancels when the radius is the SAME
# for the measurements being compared. Measured on 139 frames of PSR J2215+5135
# (report: 2025works/INOOR_aperture_error_model):
#
#   * Two stars in one frame measured at r=20 and r=27 px differ by 62 mmag of
#     pure aperture loss -- far larger than their 3-12 mmag error bars. So a
#     zeropoint calibrated on one star and applied to another is only valid if
#     both used the same radius.
#   * For ONE star across frames, a radius fixed in pixels tracks the seeing:
#     r=20 px injected 17.8 mmag rms correlated at +0.98 with seeing, which
#     looks exactly like variability. A radius that scales with the seeing keeps
#     E(r) constant and removes it.
#   * POLICY_SNR is brightness-dependent (its median radius spanned 2.12 px
#     across three stars of the same field) and its optimum jitters by ~1.9 px
#     because the SNR curve is flat at maximum. Both feed straight into the
#     aperture-loss term, and made INOOR's error bars look optimistic by 1.6x.
#   * POLICY_GROWTH is nearly brightness-independent (0.16 px spread across the
#     same three stars) but, measured on each star's own noisy profile, still
#     mismatches between stars by ~0.35 px. POLICY_FRAME_FWHM removes even that
#     by deriving the radius from the frame instead of from the star, so two
#     independent runs on two different stars necessarily agree.
POLICY_SNR = "snr"                  # per-star SNR optimum   (best single measurement)
POLICY_GROWTH = "growth"            # radius enclosing a target fraction of THIS star
POLICY_FRAME_FWHM = "frame_fwhm"    # r = k * FWHM of the FRAME  (best for series)
POLICY_FIXED = "fixed"              # one radius, given by the caller

APERTURE_POLICIES = (POLICY_SNR, POLICY_GROWTH, POLICY_FRAME_FWHM, POLICY_FIXED)


def curve_of_growth(image, center_xy, radii=None, fwhm=12.0,
                    inner_coef=2.0, outer_coef=3.0):
    """
    Background-subtracted enclosed-flux fraction E(r), normalised at the
    outermost radius. Returns (radii, fraction); fraction is forced monotonic
    non-decreasing so that inverting it for a target fraction is well defined
    even when photon noise makes the raw curve wobble.
    """
    if radii is None:
        radii = np.arange(2.0, max(30.0, 3.0 * float(fwhm)) + 1.0, 1.0)
    radii = np.asarray(radii, dtype=float)
    r_out = float(radii[-1])

    crop, cx, cy = _crop_around(image, center_xy, outer_coef * r_out)
    if crop is None:
        return radii, np.full(radii.size, np.nan)

    arr = np.asarray(crop, dtype=float)
    aps = [CircularAperture((cx, cy), r=float(r)) for r in radii]
    try:
        tab = aperture_photometry(arr, aps)
        ap_sums = np.array([float(tab[f'aperture_sum_{i}'][0]) for i in range(radii.size)])
    except Exception:
        return radii, np.full(radii.size, np.nan)

    # The photometry's own sky estimator (clipped_sky_mean), so the radius this
    # curve chooses and the aperture correction it gives describe the same sky.
    bkg, _ = _annulus_skies(arr, cx, cy, [inner_coef * r_out], [outer_coef * r_out])[0]
    net = ap_sums - bkg * np.array([ap.area for ap in aps])

    total = net[-1]
    if not np.isfinite(total) or total <= 0:
        return radii, np.full(radii.size, np.nan)
    frac = np.maximum.accumulate(net / total)
    return radii, frac


def growth_radius(image, center_xy, target_fraction=0.90, fwhm=12.0,
                  radii=None, inner_coef=2.0, outer_coef=3.0):
    """
    Radius at which the curve of growth reaches `target_fraction`. Returns None
    when the curve could not be measured, so the caller keeps its fallback.
    """
    radii, frac = curve_of_growth(image, center_xy, radii=radii, fwhm=fwhm,
                                  inner_coef=inner_coef, outer_coef=outer_coef)
    if not np.all(np.isfinite(frac)):
        return None
    t = float(target_fraction)
    if t <= frac[0]:
        return float(radii[0])
    if t >= frac[-1]:
        return float(radii[-1])
    return float(np.interp(t, frac, radii))


def _center_box(arr, box):
    """Central `box`-sized square of `arr` (the whole array if it is smaller)."""
    ny, nx = arr.shape
    if box is None or (ny <= box and nx <= box):
        return arr
    y0 = max(0, ny // 2 - box // 2)
    x0 = max(0, nx // 2 - box // 2)
    return arr[y0:y0 + box, x0:x0 + box]


def _block_mean(arr, b):
    """Mean of each b x b block of `arr`; rows/columns that do not fill a block are dropped."""
    ny, nx = arr.shape
    return arr[:ny // b * b, :nx // b * b].reshape(ny // b, b, nx // b, b).mean(axis=(1, 3))


def _detection_binning(kernel, grow):
    """
    Block size of the image one rung of estimate_frame_fwhm DETECTS on (the
    widths are always measured at full resolution). `grow` = sample box /
    first sample box.

    DAOStarFinder convolves directly, so a detection costs pixels x kernel
    area. On the sparse GRB 251013C Low-gain frame 1 one call took 28 s, 21 s
    of it in the 48 px rung (4.3 s on the 1024 px box, 16.7 s on the 2048 px
    box), with the GUI frozen. Two limits keep every detection cheap:

    * a kernel of 32 px or more detects on a (kernel // 16)-binned copy, which
      keeps the binned kernel 16-32 px wide -- still far wider than
      DAOStarFinder needs. The 24 and 12 px rungs of the default guess (12)
      stay at full resolution, so what a dense field returns at the default
      guess does not change.
    * a grown box is binned by kernel // 3, clipped to 2..`grow`, so it costs
      about what the first box did while the binned kernel stays ~3 px or
      wider (for kernels of 6 px or more): centroids stay good to a fraction
      of a full-resolution pixel.
    """
    b = max(1, int(kernel // 16))
    if grow > 1:
        b = max(b, min(int(grow), max(2, int(kernel // 3))))
    return b


def _fwhm_one_kernel(arr, med, std, kernel, threshold_sigma, max_stars, det=None):
    """
    Median stellar FWHM in `arr` using one DAOStarFinder kernel size.
    Returns (fwhm, n_stars_used, robust_sigma_of_the_widths) or (None, 0, None).

    det: optional (image, median, sigma, b) -- detect on that b x b block mean
    of `arr`, with the kernel scaled by 1/b, instead of on `arr` itself.
    """
    dimg, dmed, dstd, b = det if det is not None else (arr, med, std, 1)
    try:
        srcs = DAOStarFinder(fwhm=float(kernel) / b,
                             threshold=float(threshold_sigma) * dstd,
                             exclude_border=True)(dimg - dmed)
    except Exception:
        return None, 0, None
    if srcs is None or len(srcs) == 0:
        return None, 0, None

    order = np.argsort(-np.asarray(srcs['flux'], dtype=float))[:int(max_stars)]
    vals = []
    for i in order:
        cx = float(srcs['xcentroid'][i]); cy = float(srcs['ycentroid'][i])
        if b > 1:
            # Block centre back in full-resolution pixels. Only when binned:
            # (x + 0.5) * 1 - 0.5 is not always exactly x in floating point.
            cx, cy = (cx + 0.5) * b - 0.5, (cy + 0.5) * b - 0.5
        # Measure FWHM directly off the radial profile rather than converting a
        # half-light radius: real PSFs have Moffat wings, and the Gaussian
        # r_half -> FWHM conversion overestimated the width by ~40% on INO340
        # frames, which then pushed the aperture far past anything useful.
        # Only the half-power point is wanted, which sits near FWHM/2, so a
        # profile a couple of kernel-widths across is ample. Scaling this with
        # the kernel (3x) made the large rungs of the ladder dominate the cost:
        # a 48 px kernel built a 289x289 crop and a full sqrt over it per star.
        rp_r, rp_p = compute_radial_profile(
            arr, (cx, cy), max_radius=min(60.0, max(25.0, 1.5 * float(kernel))))
        hwhm = calculate_hwhm(rp_r, rp_p)
        if hwhm is not None and hwhm > 0:
            vals.append(2.0 * float(hwhm))
    if not vals:
        return None, 0, None

    v = np.asarray(vals, dtype=float)       # brightest (DAOStarFinder flux) first
    # Cosmic rays and hot pixels are narrower than any star. The MAD clip below
    # copes with a few of them, but on a sparse field they can be half the
    # sample and then they own the median: on GRB 251013C Low frame 1 the 24 px
    # rung found 3 stars (19.5-20.4 px) and 3 cosmic rays (2.4-4.9 px) and
    # returned 12.19, the middle of the gap; the 12 px rung (2 stars, 3 cosmic
    # rays) returned 3.24. Stars head the flux ranking -- DAOStarFinder flux
    # sums the kernel footprint, which a star fills and a cosmic ray does not
    # (2550-17200 against 150-460 there) -- so a width below half the median
    # of the brightest third is an artefact. They are dropped only when they
    # are at least half the sample, where the median breaks; below that the
    # clip handles them, and dense fields are left exactly as they were.
    narrow = v < 0.5 * float(np.median(v[:int(np.ceil(v.size / 3.0))]))
    if 2 * int(narrow.sum()) >= v.size:
        v = v[~narrow]
    # One clip pass: a cosmic ray or a blend that slipped through the flux cut
    # would otherwise drag the median of a small sample.
    if v.size >= 5:
        m = np.median(v)
        mad = np.median(np.abs(v - m))
        if mad > 0:
            keep = np.abs(v - m) <= 3.0 * 1.4826 * mad
            if keep.sum() >= 3:
                v = v[keep]
    # Star-to-star scatter of the widths kept (robust sigma), for the PSF check.
    spread = float(1.4826 * np.median(np.abs(v - np.median(v))))
    return float(np.median(v)), int(v.size), spread


def estimate_frame_fwhm(image, fwhm_guess=12.0, threshold_sigma=8.0,
                        max_stars=25, sample_box=1024, min_stars=5,
                        return_spread=False):
    """
    Robust FWHM of the FRAME, from many stars rather than one.

    This is what makes a radius reproducible across separate runs: two stars
    measured in two independent multi-photometry runs get the same radius only
    if the radius is a property of the frame. Returns None when it cannot be
    measured.

    Three things keep the answer from depending on what the user typed into
    "Fit FWHM", which is only ever a starting guess:

    * The detection kernel is swept over a ladder (2g, g, 4g) and the MEDIAN of
      the answers is returned. A kernel MUCH SMALLER than the true PSF detects
      cosmic rays, hot pixels and noise-dominated faint sources whose measured
      width is ~3 px: on INO340 g-band frames a single kernel at guess=6
      returned 3.5 px instead of 14; the ladder returns 14.0. The third rung is
      only paid for when the first two disagree, so the normal case costs two
      detections rather than three.
    * Wider kernels are not immune. On a sparse field they pick up cosmic rays
      too (GRB 251013C Low frame 1, true FWHM ~20 px: guess 12 returned 12.19,
      guess 12.19 returned 20.04), and a kernel several times the PSF finds
      nothing else. So a rung drops sources narrower than half its brightest
      stars once they are half its sample, and rungs that disagree are weighted
      by the number of stars they measured.
    * The sample box grows (1024 -> 2048 -> whole frame) until at least
      `min_stars` usable stars are found, so a sparse field is measured from the
      whole frame instead of failing or running on three sources. A grown box,
      and any kernel of 32 px or more, is searched on a block-averaged copy
      (_detection_binning), so no detection costs much more than the first.

    return_spread=True returns (fwhm, spread) instead: spread is the robust
    sigma of the single-star widths in the rung that measured the most stars,
    or None when that rung has fewer than `min_stars` -- too few to say how
    much stars of this frame differ (the PSF check then stays unchecked).
    """
    arr_full = np.asarray(image, dtype=float)
    ny, nx = arr_full.shape
    big = max(ny, nx)

    boxes = []
    b = int(sample_box) if sample_box else big
    while True:
        boxes.append(min(b, big))
        if b >= big:
            break
        b *= 2

    g = max(3.0, float(fwhm_guess))
    # Widest rung first: it is the one that cannot be fooled by sharp artefacts.
    kernels = [min(2.0 * g, 64.0), g, min(4.0 * g, 64.0)]

    best = spread = None
    for box in boxes:
        arr = _center_box(arr_full, box)
        grow = max(1, box // boxes[0])
        med = std = None
        if grow == 1:
            # A grown box detects only on binned copies (_detection_binning
            # never returns 1 there), so it needs no full-resolution sigma.
            try:
                _, med, std = sigma_clipped_stats(arr, sigma=3.0, maxiters=5)
            except Exception:
                continue
        binned = {}
        vals, ns, n_max, agreed, box_spread = [], [], 0, False, None
        for j, kern in enumerate(kernels):
            blk = _detection_binning(kern, grow)
            det = None
            if blk > 1:
                if blk not in binned:
                    small = _block_mean(arr, blk)
                    _, bmed, bstd = sigma_clipped_stats(small, sigma=3.0, maxiters=5)
                    binned[blk] = (small, bmed, bstd, blk)
                det = binned[blk]
            v, n, s = _fwhm_one_kernel(arr, med, std, kern, threshold_sigma, max_stars, det)
            if v is not None:
                vals.append(v)
                ns.append(n)
                if n > n_max:
                    box_spread = s if n >= int(min_stars) else None
                n_max = max(n_max, n)
            # Two independent kernels agreeing is already enough evidence; the
            # third rung exists to break a tie, not to be paid for every frame.
            if j == 1 and len(vals) == 2 and abs(vals[0] - vals[1]) <= 0.25 * max(vals):
                agreed = True
                break
        if not vals:
            continue
        if agreed:
            best = float(np.median(vals))
        else:
            # Rungs that disagree are not equally good evidence. A kernel much
            # wider than the PSF finds no stars, only a stray cosmic ray, and a
            # plain median of two rungs is their midpoint: on a synthetic sparse
            # frame (12 px PSF, guess 20) a 40 px rung's single 3.0 px source and
            # a 20 px rung's six 12.15 px stars gave 7.58. Count each rung once
            # per star it measured (with one star each, this is the old median).
            best = float(np.median(np.repeat(vals, ns)))
        spread = box_spread
        if n_max >= int(min_stars):
            return (best, spread) if return_spread else best
    return (best, spread) if return_spread else best


def frame_growth_curve(image, fwhm, radii=None, threshold_sigma=20.0,
                       max_stars=15, sample_box=1024, inner_coef=2.0,
                       outer_coef=3.0, min_stars=4):
    """
    Median curve of growth E(r) of the FRAME, from bright isolated stars.

    Used for the aperture correction: a magnitude measured through radius r is
    fainter than the star's total magnitude by -2.5*log10(E(r)). On INO340
    g-band frames E(1.0*FWHM) ~= 0.78, i.e. a 267 mmag offset, so a magnitude
    reported at a small radius is not on the same scale as a catalogue total
    magnitude. That offset largely cancels in a zeropoint transfer only when
    both stars have the same E(r); correcting for it explicitly makes the
    reported magnitude comparable to a catalogue directly.

    Returns (radii, fraction) or (None, None) when it cannot be measured.
    """
    arr_full = np.asarray(image, dtype=float)
    ny, nx = arr_full.shape
    big = max(ny, nx)
    fw = max(1.0, float(fwhm))
    if radii is None:
        radii = np.arange(2.0, 5.0 * fw + 1.0, 1.0)
    radii = np.asarray(radii, dtype=float)

    boxes = []
    b = int(sample_box) if sample_box else big
    while True:
        boxes.append(min(b, big))
        if b >= big:
            break
        b *= 2

    margin = outer_coef * float(radii[-1]) + 2.0
    for box in boxes:
        arr = _center_box(arr_full, box)
        if min(arr.shape) < 4 * margin:
            continue
        try:
            _, med, std = sigma_clipped_stats(arr, sigma=3.0, maxiters=5)
            srcs = DAOStarFinder(fwhm=fw, threshold=float(threshold_sigma) * std,
                                 exclude_border=True)(arr - med)
        except Exception:
            continue
        if srcs is None or len(srcs) == 0:
            continue

        xs = np.asarray(srcs['xcentroid'], dtype=float)
        ys = np.asarray(srcs['ycentroid'], dtype=float)
        curves = []
        for i in np.argsort(-np.asarray(srcs['flux'], dtype=float)):
            cx, cy = xs[i], ys[i]
            if not (margin < cx < arr.shape[1] - margin
                    and margin < cy < arr.shape[0] - margin):
                continue
            # Isolation: a neighbour inside the sky annulus biases the
            # normalisation and produces E(r) > 1, which would corrupt the
            # median far more than dropping the star costs.
            d = np.hypot(xs - cx, ys - cy)
            d[d < 1e-6] = np.inf
            if d.min() < 4.0 * fw:
                continue
            _, frac = curve_of_growth(arr, (cx, cy), radii=radii, fwhm=fw,
                                      inner_coef=inner_coef, outer_coef=outer_coef)
            if not np.all(np.isfinite(frac)):
                continue
            # A usable curve of growth has FLATTENED by ~3 FWHM. One that is
            # still climbing out at 5 FWHM is not measuring this star's wings:
            # it is a faint neighbour drifting into the aperture, or a sky
            # gradient across the wide outer annulus. Those curves normalise on
            # the wrong total and, left in, dragged the frame median at 3 FWHM
            # from 0.98 down to 0.77 on INO340 frames.
            if frac[int(np.argmin(np.abs(radii - 3.0 * fw)))] < 0.90:
                continue
            # The opposite failure: the curve rose above its own total, i.e. the
            # net flux FELL further out (sky over-subtracted, or a neighbour in
            # the far annulus). An enclosed fraction above 1 is impossible. On
            # PSR J2215 frame High_1 the plain-mean sky let 13 such curves in (up
            # to E = 8.9), giving E(1.05 FWHM) = 0.855 against 0.794 from clean
            # stars; with the clipped sky 7 of 15 still reached 1.08-1.19.
            if float(np.max(frac)) > 1.02:
                continue
            curves.append(frac)
            if len(curves) >= int(max_stars):
                break
        if len(curves) >= int(min_stars):
            return radii, np.median(np.vstack(curves), axis=0)
    return None, None


def aperture_correction(cog_radii, cog_frac, r):
    """
    Magnitude correction (<= 0) taking a magnitude measured through radius `r`
    to a total magnitude: m_total = m_aper + aperture_correction(...).
    Returns None when the curve is unusable. It used to return 0.0, which a
    caller cannot tell from "no correction needed": a run labelled "+apcorr"
    then carried uncorrected magnitudes on every frame with fewer than four
    usable stars, and nothing said so.
    """
    if cog_radii is None or cog_frac is None:
        return None
    try:
        e = float(np.interp(float(r), np.asarray(cog_radii, dtype=float),
                            np.asarray(cog_frac, dtype=float)))
    except Exception:
        return None
    if not np.isfinite(e) or e <= 0.0 or e > 1.5:
        return None
    return float(2.5 * math.log10(e))


# Default FWHM factor k = 1.0, the same as the GUI, the bulk worker and the
# dialogs; these two functions alone said 1.5, which only scripts calling them
# without fwhm_factor ever saw. Every GUI path passes the value explicitly.
def resolve_aperture_radius(image, center_xy, *, policy=POLICY_SNR,
                            r_best=None, growth_fraction=0.90,
                            fwhm_factor=1.0, frame_fwhm=None, fwhm=12.0,
                            fixed_radius=None, inner_coef=2.0, outer_coef=3.0):
    """
    Turn an aperture policy into a radius in pixels. Falls back to `r_best`
    (the SNR optimum) whenever the requested policy cannot be evaluated, so a
    measurement is never silently skipped.
    """
    fallback = float(r_best) if r_best is not None else float(fwhm)

    if policy == POLICY_FIXED:
        return float(fixed_radius) if fixed_radius is not None else fallback

    if policy == POLICY_GROWTH:
        r = growth_radius(image, center_xy, target_fraction=growth_fraction,
                          fwhm=fwhm, inner_coef=inner_coef, outer_coef=outer_coef)
        return float(r) if r is not None else fallback

    if policy == POLICY_FRAME_FWHM:
        f = frame_fwhm if frame_fwhm is not None else estimate_frame_fwhm(
            image, fwhm_guess=fwhm)
        if f is not None and f > 0:
            return float(fwhm_factor) * float(f)
        return fallback

    return fallback


def peak_status(image, center_xy, radius, linear_limit=None):
    """
    Brightest pixel inside the aperture and what it means: 'saturated' at the
    ADC ceiling, 'nonlinear' above the mode's measured linear limit, else 'ok'.
    Returns (peak_adu, status); (None, 'ok') when the aperture holds no data.

    Nothing checked this before: a PSR J2215 star with a 3954 ADU raw peak
    (416 pixels above 3500) came out at SNR 1507 with a 0.7 mmag error bar.
    Where no limit was measured (Low, Merge, binned) only the ceiling counts,
    and a pixel saturated in the raw read can sit a little below it after
    bias, dark and flat; those cases are not caught.
    """
    crop, cx, cy = _crop_around(image, center_xy, float(radius) + 1.0)
    if crop is None:
        return None, 'ok'
    yy, xx = np.indices(crop.shape)
    vals = crop[((xx - cx) ** 2 + (yy - cy) ** 2 <= float(radius) ** 2) & np.isfinite(crop)]
    if vals.size == 0:
        return None, 'ok'
    peak = float(vals.max())
    if peak >= ADC_CEILING_ADU:
        return peak, 'saturated'
    if linear_limit is not None and peak > float(linear_limit):
        return peak, 'nonlinear'
    return peak, 'ok'


def measure_star(image, center_xy, *,
                 fwhm=12.0,
                 aperture_override=None,
                 aperture_policy=POLICY_SNR,
                 growth_fraction=0.90, fwhm_factor=1.0, frame_fwhm=None,
                 inner_coef=2.0, outer_coef=3.0,
                 read_noise=DEFAULT_MODE.read_noise,
                 dark_noise=DEFAULT_MODE.dark_noise,
                 gain=DEFAULT_MODE.gain,
                 zeropoint=0.0, exptime=None,
                 radii=None, n_radii=200,
                 snr_curve=None, want_growth=False,
                 aperture_correct=False, frame_cog=None,
                 linear_limit=None, frame_fwhm_spread=None):
    """
    THE measurement routine: SNR-vs-radius -> aperture choice -> radial
    profile -> aperture photometry, for a star already centred at center_xy.

    Every caller (single click, live parameter re-measure, bulk photometry)
    goes through here so that identical inputs give bit-identical curves,
    r_best and magnitudes. Before this existed the three paths each rebuilt
    the sequence slightly differently and disagreed on both the plotted
    curves and the chosen aperture.

    aperture_override : use this radius for the reported magnitude instead of
                        the SNR-optimised r_best (the "fixed aperture" mode).
                        Takes precedence over aperture_policy, so a caller that
                        has already worked out one shared radius for the whole
                        frame can simply pass it here.
    aperture_policy   : how to pick the radius when no override is given; see
                        APERTURE_POLICIES and the note above them. The SNR curve
                        is computed regardless, for the plot and for r_best.
    frame_fwhm        : pre-computed FWHM of this frame, so a caller measuring
                        many stars in one frame pays for it only once.
    snr_curve         : optional pre-computed (radii, snrs, r_best) to reuse
                        instead of recomputing (live-update "fast" path).
    aperture_correct  : add the frame's aperture correction so the reported
                        magnitude is a TOTAL magnitude rather than the flux
                        inside r_used. Needed to compare against a catalogue;
                        harmless but unnecessary for differential work, where
                        the term cancels as long as the radius is shared.
    frame_cog         : pre-computed (radii, fraction) from frame_growth_curve,
                        so a caller measuring many stars in one frame pays for
                        it only once. Measured here when omitted.
    linear_limit      : the frame's camera-mode linear limit (calibrated ADU);
                        phot['peak_adu'] and phot['saturation'] report the
                        brightest aperture pixel against it (see peak_status).
    frame_fwhm_spread : robust sigma of single-star widths in this frame
                        (estimate_frame_fwhm(return_spread=True)); under the
                        frame-FWHM rule phot['psf_check'] is 'broad' when the
                        star is wider than frame_fwhm + 3 * spread, else 'ok',
                        and 'unchecked' without a spread or under another rule.

    Returns dict(radii, snrs, r_best, r_used, rp_radii, rp_profile, phot).
    """
    if snr_curve is not None and snr_curve[0] is not None and snr_curve[1] is not None:
        c_radii, c_snrs, c_r_best = snr_curve
        radii_out = np.asarray(c_radii)
        snrs_out = np.asarray(c_snrs)
        r_best = float(c_r_best) if c_r_best is not None else float(radii_out[0])
    else:
        if radii is None:
            radii = np.linspace(1, max(40, 5 * fwhm), int(n_radii))
        radii_out, snrs_out, r_best, _ = compute_snr_vs_radius(
            image, center_xy, fwhm=fwhm, radii=radii,
            read_noise=read_noise, dark_noise=dark_noise, gain=gain,
            inner_coef=inner_coef, outer_coef=outer_coef)
        r_best = float(r_best)

    if aperture_override is not None:
        r_used = float(aperture_override)
    else:
        r_used = resolve_aperture_radius(
            image, center_xy, policy=aperture_policy, r_best=r_best,
            growth_fraction=growth_fraction, fwhm_factor=fwhm_factor,
            frame_fwhm=frame_fwhm, fwhm=fwhm,
            inner_coef=inner_coef, outer_coef=outer_coef)

    # The growth curve is what the GUI plots instead of the SNR curve when the
    # growth policy is active, so hand it back rather than making the caller
    # measure the star a second time.
    cog_radii = cog_frac = None
    if want_growth or aperture_policy == POLICY_GROWTH:
        cog_radii, cog_frac = curve_of_growth(
            image, center_xy, fwhm=fwhm,
            inner_coef=inner_coef, outer_coef=outer_coef)

    rp_radii, rp_profile = compute_radial_profile(
        image, center_xy, max_radius=profile_max_radius(r_used))

    phot = perform_aperture_photometry(
        image, center_xy, r_used,
        inner_coef=inner_coef, outer_coef=outer_coef,
        read_noise=read_noise, dark_noise=dark_noise, gain=gain,
        zeropoint=zeropoint, exptime=exptime)

    # Aperture correction. Applied to the magnitudes only: the flux, the SNR and
    # therefore the error bar all still describe what was actually measured
    # inside r_used, and the correction is a scale factor common to every star
    # in the frame, so it adds no noise of its own worth propagating here.
    # The status says which it is, because a frame whose curve of growth cannot
    # be built keeps its aperture magnitude ('unavailable'), and that must not
    # pass for a total magnitude in a run that asked for one.
    ap_corr = 0.0
    ap_corr_status = 'off'
    if aperture_correct:
        if frame_cog is None:
            frame_cog = frame_growth_curve(
                image, frame_fwhm if frame_fwhm else fwhm,
                inner_coef=inner_coef, outer_coef=outer_coef)
        corr = aperture_correction(frame_cog[0], frame_cog[1], r_used)
        if corr is None:
            ap_corr_status = 'unavailable'
        else:
            ap_corr, ap_corr_status = corr, 'applied'
            for key in ('instr_mag', 'mag'):
                if phot.get(key) is not None:
                    phot[key] = float(phot[key]) + ap_corr
    phot['aperture_correction'] = ap_corr
    phot['aperture_correction_status'] = ap_corr_status
    phot['peak_adu'], phot['saturation'] = peak_status(image, center_xy, r_used, linear_limit)

    # r = k * FWHM_frame cancels the aperture loss only for stars with the
    # frame's PSF. A blend, a galaxy or a broader PSF loses more light outside
    # r and reads too faint. The bar is the frame's own star-to-star scatter:
    # on PSR J2215 High_80 243 isolated stars sat at 1.012 +- 0.044 of the
    # frame FWHM (corners 1.027), saturated ones at a median 2.0.
    phot['star_fwhm'], phot['psf_check'] = None, 'unchecked'
    if (aperture_override is None and aperture_policy == POLICY_FRAME_FWHM
            and frame_fwhm and frame_fwhm_spread is not None):
        hwhm = calculate_hwhm(rp_radii, rp_profile)
        if hwhm is not None and hwhm > 0:
            phot['star_fwhm'] = 2.0 * float(hwhm)
            phot['psf_check'] = ('broad' if phot['star_fwhm'] > float(frame_fwhm)
                                 + 3.0 * float(frame_fwhm_spread) else 'ok')

    return dict(radii=radii_out, snrs=snrs_out, r_best=r_best, r_used=r_used,
                rp_radii=rp_radii, rp_profile=rp_profile, phot=phot,
                cog_radii=cog_radii, cog_frac=cog_frac,
                frame_fwhm=frame_fwhm, frame_cog=frame_cog,
                aperture_correction=ap_corr,
                aperture_correction_status=ap_corr_status,
                aperture_policy=(POLICY_FIXED if aperture_override is not None
                                 else aperture_policy))


def compute_radial_profile(image, center_xy, max_radius):
    """
    Compute radial mean profile in 1-pixel-thick annuli up to max_radius.
    Returns (radii_array, profile_array)
    """
    cx, cy = float(center_xy[0]), float(center_xy[1])
    if max_radius <= 0:
        return np.array([]), np.array([])

    radii = np.arange(1, int(max_radius) + 1)
    if radii.size == 0:
        return radii, np.asarray([], dtype='float32')

    # Only pixels within max_radius+0.5 of the centre can land in any annulus,
    # so crop instead of building np.indices() over the whole frame. On a
    # 4096^2 KL4040 frame the old version allocated two 134 MB int64 index
    # arrays plus a 134 MB radius array and then ran one full-frame boolean
    # mask PER annulus; results here are identical, just without touching the
    # 16.7M pixels that cannot contribute.
    crop, cx_l, cy_l = _crop_around(image, (cx, cy), float(int(max_radius)) + 1.0)
    if crop is None or crop.size == 0:
        return radii, np.full(radii.size, np.nan, dtype='float32')

    ys, xs = np.ogrid[0:crop.shape[0], 0:crop.shape[1]]
    r = np.sqrt((xs - cx_l) ** 2 + (ys - cy_l) ** 2)

    # Vectorised equivalent of the per-annulus mask loop: bin index i holds the
    # pixels with i-0.5 <= r < i+0.5. floor(r+0.5) reproduces that half-open
    # interval exactly; np.rint would NOT, because it rounds halves to even
    # (rint(0.5)==0, rint(1.5)==2) and would misplace exact .5 boundaries.
    bins = np.floor(r + 0.5).astype(np.intp)
    vals = crop
    good = np.isfinite(vals) & (bins >= 1) & (bins <= radii[-1])
    if not np.any(good):
        return radii, np.full(radii.size, np.nan, dtype='float32')

    b = bins[good]
    v = vals[good].astype(np.float64, copy=False)
    counts = np.bincount(b, minlength=radii[-1] + 1)
    sums = np.bincount(b, weights=v, minlength=radii[-1] + 1)

    with np.errstate(invalid='ignore', divide='ignore'):
        prof = sums[1:radii[-1] + 1] / counts[1:radii[-1] + 1]
    prof[counts[1:radii[-1] + 1] == 0] = np.nan
    return radii, np.asarray(prof, dtype='float32')

def pick_best_source_crowded(sources, crop_origin, target_xy, fwhm=8.0):
    """
    Pick the source the user meant when they clicked at `target_xy`.

    Distance decides. Brightness only breaks a tie among sources that are
    essentially on top of each other (within half a FWHM of the closest one),
    where "which of these is the star" is a real question and the brighter
    answer is the better guess.

    It used to be a weighted score -- 40% distance, 35% flux, 15% roundness,
    10% sharpness -- with the distance term normalised by the LARGEST distance
    in the crop. That makes the distance penalty depend on the crop size rather
    than on the sky, so a click next to a faint star beside a bright one landed
    on the bright one. Commit 4c94990 tried switching to pure nearest and was
    reverted in f025105, presumably because pure nearest happily returns a noise
    peak; keeping the brightness tie-break is what makes nearest safe.

    Returns: (full_coords, crop_coords, source_index, score)
    where score is the distance in pixels from the click (lower is better).
    """
    if sources is None or len(sources) == 0:
        return None, None, None, 0.0

    try:
        xcol = np.asarray(sources['xcentroid'], dtype=float)
        ycol = np.asarray(sources['ycentroid'], dtype=float)
    except Exception:
        return None, None, None, 0.0
    if xcol.size == 0:
        return None, None, None, 0.0

    try:
        flux = np.asarray(sources['flux'], dtype=float)
    except Exception:
        flux = np.zeros_like(xcol)

    x0, y0 = crop_origin
    tx = float(target_xy[0]) - x0
    ty = float(target_xy[1]) - y0
    dist = np.hypot(xcol - tx, ycol - ty)

    # Everything within half a FWHM of the nearest candidate is "the same
    # place"; among those, prefer the brightest.
    tie = max(1.0, 0.5 * float(fwhm))
    close = np.nonzero(dist <= dist.min() + tie)[0]
    best_idx = int(close[np.argmax(flux[close])]) if close.size else int(np.argmin(dist))

    x_crop, y_crop = float(xcol[best_idx]), float(ycol[best_idx])
    return ((x0 + x_crop, y0 + y_crop), (x_crop, y_crop), best_idx,
            float(dist[best_idx]))


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
    
    # DAOStarFinder convolves with a kernel about 2*ceil(1.5*fwhm)+1 across, so
    # a crop smaller than that returns NOTHING however low the threshold is --
    # silently. On the GRB 22-Oct i frames (PSF 17.5 px, kernel ~55 px) the
    # 30 px Detection Stamp gave 0 detections at every threshold down to 0.1
    # sigma, which reads as "there is no source here" rather than "the stamp is
    # too small". Grow the crop to fit the kernel with a margin.
    kernel_box = 2 * int(math.ceil(1.5 * float(fwhm))) + 1
    need_half = int(math.ceil(kernel_box * 1.5))
    if (x1 - x0) < 2 * need_half or (y1 - y0) < 2 * need_half:
        x0 = max(0, x - need_half); x1 = min(nx, x + need_half)
        y0 = max(0, y - need_half); y1 = min(ny, y + need_half)

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
