# ext_coeff.py
"""
Extinction coefficient helper utilities.

Functions:
 - get_datetime_from_header(header): robustly extract date-time string suitable for astropy.Time
 - compute_airmass_for_files(files, indices, ra, dec, latitude, longitude): returns
    per-frame info (time_str, airmass) and only those frames with successfully parsed times
 - fit_mag_vs_airmass(mags, errs, airmass): linear fit (mag = intercept + k * airmass), returns slope k and uncertainties
 - build_results_json(...) : convenience for saving results to JSON
"""

import re
import math
import json
from typing import List, Tuple, Dict, Optional
import numpy as np
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
import astropy.units as u
import os
from utils.exptime_utils import get_exptime_seconds
# Robust date extractor for common header forms.
_iso_like_re = re.compile(r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)')

def _iso_token(val) -> Optional[str]:
    """Pull the leading ISO-ish timestamp out of one header value."""
    if val is None:
        return None
    try:
        s = val.decode('utf-8') if isinstance(val, bytes) else str(val)
    except Exception:
        s = str(val)
    s = s.strip()
    if not s:
        return None
    m = _iso_like_re.search(s)
    if m:
        return m.group(1)
    token = s.split()[0].strip("'\"")
    return token if ('-' in token and ':' in token) else None


# A civil timezone offset is a whole quarter-hour and at most 14 h from UT.
_TZ_STEP_HOURS = 0.25
_TZ_MAX_HOURS = 14.0


def _is_timezone_offset(delta_hours: float) -> bool:
    """True if `delta_hours` looks like two clocks in different timezones."""
    a = abs(delta_hours)
    if a < 1.0 / 60.0 or a > _TZ_MAX_HOURS:
        return False
    return abs(a / _TZ_STEP_HOURS - round(a / _TZ_STEP_HOURS)) < 0.02


def get_datetime_from_header(hdr) -> Optional[str]:
    """
    The UT capture time, as an ISO string.

    KL4040 headers carry BOTH, and they are not the same clock. Verified on
    real frames:

        DATE-OBS = 2025-10-14T00:34:15      (Iran local)
        DATE     = 2025-10-13T21:04:17      (UT)

    -- exactly 3 h 30 m apart, the Iran offset, and note they fall on
    different calendar DAYS. The app's own light-curve dialog already labels
    them "Local (DATE-OBS)" and "UT (DATE)".

    This function used to prefer DATE-OBS and hand it to Time(scale='utc'),
    which put every airmass 3 h 30 m out in hour angle -- 52 degrees of Earth
    rotation -- and made the fitted extinction coefficient meaningless.

    Deciding between them: FITS defines DATE as the file's UT creation time
    and DATE-OBS as the UT start of the observation, so DATE-OBS is normally
    the right key. But when the two differ by a clean civil timezone offset
    they are plainly the same instant on two clocks, and the one the camera
    labelled DATE is the UT one. An arbitrary difference means DATE really is
    a file-creation stamp, and DATE-OBS wins. Returns None when neither key is
    usable.
    """
    if hdr is None:
        return None

    tokens = {k: _iso_token(hdr.get(k)) for k in ('DATE', 'DATE-OBS')}
    tokens = {k: v for k, v in tokens.items() if v}
    if not tokens:
        return None
    if len(tokens) == 1:
        return next(iter(tokens.values()))

    try:
        t_date = Time(tokens['DATE'])
        t_obs = Time(tokens['DATE-OBS'])
    except Exception:
        return tokens['DATE-OBS']

    delta_h = float((t_obs - t_date).to_value('hour'))
    if abs(delta_h) < 1.0 / 60.0:
        return tokens['DATE']
    if _is_timezone_offset(delta_h):
        return tokens['DATE']
    return tokens['DATE-OBS']


def _parse_skycoord(ra_in, dec_in) -> Optional[SkyCoord]:
    """
    Try to create an astropy SkyCoord robustly given ra/dec inputs.
    Accepts decimal degrees ("45.3", "+51.6"), sexagesimal hourangle RA
    ("22:15:30", "22h15m30s"), or a combined free-form string.

    Plain decimal values are ALWAYS interpreted as degrees: trying
    hourangle first (the old behavior) silently misreads e.g. ra="45.3"
    as 45.3 HOURS.
    """
    ra_s = str(ra_in).strip()
    dec_s = str(dec_in).strip()
    # 1) plain decimal -> degrees
    try:
        return SkyCoord(ra=float(ra_s), dec=float(dec_s), unit=(u.deg, u.deg))
    except (ValueError, TypeError):
        pass
    # 2) sexagesimal / explicit hour markers -> hourangle RA, degree Dec
    try:
        if ':' in ra_s or 'h' in ra_s.lower():
            return SkyCoord(ra=ra_s, dec=dec_s, unit=(u.hourangle, u.deg))
    except Exception:
        pass
    # 3) free-form combined string
    try:
        return SkyCoord(f"{ra_s} {dec_s}", frame='icrs')
    except Exception:
        return None

def compute_airmass_for_files(files: List[str],
                              indices: List[int],
                              ra: str, dec: str,
                              latitude: float, longitude: float,
                              elevation_m: float = 0.0,
                              use_exposure_midpoint: bool = True
                              ) -> Tuple[List[dict], List[str]]:
    """
    For selected frames (files[index]), parse header time and compute airmass for given RA/Dec and location.

    Returns:
      per_frame_info: list of dicts with keys:
         - index, filename, time_str, parsed_time_iso, airmass (sec z), zenith_angle_deg
      errors: list of error messages (if any)
    """
    from astropy.io import fits

    results = []
    errors = []
    # build location
    loc = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg, height=elevation_m * u.m)

    sc = _parse_skycoord(ra, dec)
    if sc is None:
        errors.append("Failed to parse RA/Dec. Provide either HH:MM:SS / DD:MM:SS or decimal degrees.")
        return results, errors

    for idx in indices:
        if idx < 0 or idx >= len(files):
            errors.append(f"Index {idx} out of range.")
            continue
        fp = files[idx]
        try:
            hdr = fits.getheader(fp)
        except Exception as e:
            errors.append(f"Failed reading header for {fp}: {e}")
            continue
        dt = get_datetime_from_header(hdr)
        if dt is None:
            errors.append(f"No DATE-OBS-like header found in {os.path.basename(fp)}.")
            continue
        # parse Time robustly
        try:
            # Time(..., format='isot') expects 'YYYY-MM-DDTHH:MM:SS[.sss]'
            t = Time(dt, format='isot', scale='utc')
        except Exception:
            try:
                # fallback: let astropy parse as ISO or try flexible parsing
                t = Time(dt, scale='utc')
            except Exception as e:
                errors.append(f"Failed to parse time '{dt}' in {fp}: {e}")
                continue

        # The header time is the START of the exposure. Airmass changes across
        # a 58 s INO340 frame, and near the horizon it changes fast, so the
        # airmass that belongs with the frame's flux is the one at mid-exposure.
        exptime_s = get_exptime_seconds(hdr, default=None)
        t_start = t
        if use_exposure_midpoint and exptime_s and exptime_s > 0:
            t = t + (0.5 * float(exptime_s)) * u.s
        # altaz frame
        altaz = sc.transform_to(AltAz(obstime=t, location=loc))
        alt = altaz.alt.to(u.deg).value
        zenith = 90.0 - alt
        # Reject frames where the target is at/below the horizon -- sec(z)
        # is meaningless there (negative or diverging).
        if alt <= 0.0:
            errors.append(f"Target below horizon (alt={alt:.1f} deg) in {os.path.basename(fp)}; skipped.")
            continue
        # airmass using simple secant(zenith). For large zenith angles (>75 deg) this is inaccurate, but acceptable here.
        rad = math.radians(zenith)
        # protect against cos = 0
        cosv = math.cos(rad)
        if abs(cosv) < 1e-8:
            airmass = float('inf')
        else:
            airmass = 1.0 / cosv
        results.append({
            'index': idx,
            'filename': os.path.basename(fp),
            'filepath': fp,
            'time_str_header': dt,
            'time_start_iso': t_start.isot,
            'exptime_s': (float(exptime_s) if exptime_s else None),
            'time_iso': t.isot,
            'alt_deg': alt,
            'zenith_deg': round(zenith, 3),
            'airmass': float(airmass)
        })

    return results, errors

def fit_mag_vs_airmass(mags: List[float], errs: Optional[List[float]], airmasses: List[float]):
    """
    Fit a linear relation mag = intercept + k * airmass.
    Returns dict with slope k, intercept, cov matrix, uncertainties.
    If errs is provided and all positive, perform weighted least squares with weights = 1/sigma.
    """
    x = np.asarray(airmasses, dtype=float)
    y = np.asarray(mags, dtype=float)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        raise ValueError("Empty or mismatched input arrays for fit.")
    # determine weights
    w = None
    if errs is not None:
        earr = np.asarray(errs, dtype=float)
        if earr.size == x.size and np.all(np.isfinite(earr)) and np.all(earr > 0):
            # np.polyfit uses w such that it multiplies residuals --> pass w = 1/sigma
            w = 1.0 / earr
    # perform linear fit (degree 1)
    # try to compute covariance (numpy.polyfit with cov=True)
    try:
        if w is None:
            p, cov = np.polyfit(x, y, 1, cov=True)
        else:
            # cov='unscaled': our weights are 1/sigma with sigma = ABSOLUTE
            # photometric uncertainties (mag_err), so the covariance must not
            # be rescaled by reduced chi-square (the cov=True default treats
            # weights as merely relative and would silently rescale k_err).
            p, cov = np.polyfit(x, y, 1, w=w, cov='unscaled')
        # p: [slope, intercept] from polyfit? Actually polyfit returns highest-first: p[0]*x + p[1]
        slope = float(p[0])
        intercept = float(p[1])
        # uncertainties are sqrt of diagonal
        unc = np.sqrt(np.diag(cov)).tolist()
        slope_err, intercept_err = float(unc[0]), float(unc[1])
        out = dict(slope=slope, intercept=intercept, cov=cov.tolist(),
                   slope_err=slope_err, intercept_err=intercept_err,
                   chi2_nu=None, slope_err_scaled=None, intercept_err_scaled=None)
        # How well the line fits the error bars. The errors above trust the
        # error bars; when the points scatter more than they allow (clouds,
        # transparency drift: chi2_nu = 34.5 on one GRB 251013C star, k_err
        # reported 6x too small) the scaled errors, err * sqrt(chi2_nu), are
        # the honest ones. Both are returned; the caller shows both.
        if w is not None and x.size > 2:
            chi2 = float(np.sum(((y - (slope * x + intercept)) * w) ** 2))
            chi2_nu = chi2 / (x.size - 2)
            out.update(chi2_nu=chi2_nu,
                       slope_err_scaled=slope_err * float(np.sqrt(chi2_nu)),
                       intercept_err_scaled=intercept_err * float(np.sqrt(chi2_nu)))
        return out
    except Exception:
        # fallback to manual least-squares
        if w is None:
            A = np.vstack([x, np.ones_like(x)]).T
            coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
            slope = float(coeffs[0]); intercept = float(coeffs[1])
            return dict(slope=slope, intercept=intercept, cov=None, slope_err=float('nan'), intercept_err=float('nan'))
        else:
            # weighted least squares
            W = np.diag(w**2)
            A = np.vstack([x, np.ones_like(x)]).T
            try:
                AtW = A.T @ W
                covm = np.linalg.inv(AtW @ A)
                coeffs = covm @ (AtW @ y)
                slope = float(coeffs[0]); intercept = float(coeffs[1])
                errs = np.sqrt(np.diag(covm))
                return dict(slope=slope, intercept=intercept, cov=covm.tolist(), slope_err=float(errs[0]), intercept_err=float(errs[1]))
            except Exception:
                slope = float(np.nan); intercept = float(np.nan)
                return dict(slope=slope, intercept=intercept, cov=None, slope_err=float('nan'), intercept_err=float('nan'))

def build_results_json(ra, dec, filtername, latitude, longitude, per_frame_results, fit_result) -> dict:
    """
    Build a JSON-serializable dictionary with metadata, per-frame table and fit summary.
    """
    return {
        'ra_input': ra,
        'dec_input': dec,
        'filter': filtername,
        'observatory_latitude': float(latitude),
        'observatory_longitude': float(longitude),
        'per_frame': per_frame_results,
        'fit': {
            'slope_k': fit_result.get('slope'),
            'slope_err': fit_result.get('slope_err'),
            'intercept': fit_result.get('intercept'),
            'intercept_err': fit_result.get('intercept_err'),
            'chi2_nu': fit_result.get('chi2_nu'),
            'slope_err_scaled': fit_result.get('slope_err_scaled'),
            'intercept_err_scaled': fit_result.get('intercept_err_scaled'),
            'cov': fit_result.get('cov')
        }
    }

def save_json(path: str, data: dict):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
