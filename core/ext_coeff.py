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
# Robust date extractor for common header forms.
_iso_like_re = re.compile(r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)')

def get_datetime_from_header(hdr) -> Optional[str]:
    """
    Try several header keys and extract a usable ISO-ish timestamp string.
    Returns string suitable for astropy.Time(format='isot') or None if not found.
    """
    if hdr is None:
        return None
    # candidate keys to try:
    keys = ['DATE']
    for k in keys:
        val = hdr.get(k, None)
        if val is None:
            continue
        # header may be bytes
        try:
            s = val.decode('utf-8') if isinstance(val, bytes) else str(val)
        except Exception:
            s = str(val)
        s = s.strip()
        if not s:
            continue
        # try regex to pull the front ISO-like token
        m = _iso_like_re.search(s)
        if m:
            token = m.group(1)
            return token
        # fallback: split at whitespace and take first token
        token = s.split()[0]
        # remove surrounding quotes
        token = token.strip("'\"")
        # quick check: should contain '-' and ':'
        if '-' in token and ':' in token:
            return token
    return None

def _parse_skycoord(ra_in, dec_in) -> Optional[SkyCoord]:
    """
    Try to create an astropy SkyCoord robustly given ra/dec inputs.
    Accepts strings like "22h15m", "+51d35m", or numeric degrees.
    """
    try:
        # try hourangle for RA, degrees for Dec (common)
        sc = SkyCoord(ra=ra_in, dec=dec_in, unit=(u.hourangle, u.deg))
        return sc
    except Exception:
        pass
    try:
        # try both degrees
        sc = SkyCoord(ra=ra_in, dec=dec_in, unit=(u.deg, u.deg))
        return sc
    except Exception:
        pass
    try:
        # try letting SkyCoord parse strings (one-argument style)
        sc = SkyCoord(f"{ra_in} {dec_in}", frame='icrs')
        return sc
    except Exception:
        return None

def compute_airmass_for_files(files: List[str],
                              indices: List[int],
                              ra: str, dec: str,
                              latitude: float, longitude: float,
                              elevation_m: float = 0.0) -> Tuple[List[dict], List[str]]:
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
            errors.append(f"No DATE-OBS-like header found in {fp}.")
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
        # altaz frame
        altaz = sc.transform_to(AltAz(obstime=t, location=loc))
        alt = altaz.alt.to(u.deg).value
        zenith = 90.0 - alt
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
            p, cov = np.polyfit(x, y, 1, w=w, cov=True)
        # p: [slope, intercept] from polyfit? Actually polyfit returns highest-first: p[0]*x + p[1]
        slope = float(p[0])
        intercept = float(p[1])
        # uncertainties are sqrt of diagonal
        unc = np.sqrt(np.diag(cov)).tolist()
        slope_err, intercept_err = float(unc[0]), float(unc[1])
        return dict(slope=slope, intercept=intercept, cov=cov.tolist(), slope_err=slope_err, intercept_err=intercept_err)
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
            'cov': fit_result.get('cov')
        }
    }

def save_json(path: str, data: dict):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
