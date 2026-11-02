"""
Exposure time detection and formatting utilities.

THE single implementation for EXPTIME parsing in INOOR. Do not reimplement
this logic elsewhere: two subtly different heuristics running in different
pipeline stages can silently disagree about the exposure time of the SAME
file (this actually happened between calibration and bulk photometry).

The FLI Kepler cameras record EXPTIME in 10-microsecond ticks with a header
comment like "Duration of exposure in 10us periods". Unit detection uses:
  1. the header comment of the matched keyword (authoritative), then
  2. a magnitude heuristic: values > 1e5 are assumed to be 10us ticks
     (1e5 ticks = 1 s; a genuine 100,000-second exposure is implausible).
"""

# Keyword candidates, in priority order (union of all spellings seen in the
# codebase's data sources).
_EXPTIME_KEYS = ('EXPTIME', 'EXP_TIME', 'EXPOSURE', 'EXPT', 'TEXPTIME', 'EXPOS')

# Ticks are 10 microseconds: seconds = ticks * 1e-5
_TICKS_TO_SECONDS = 1e-5
_TICKS_HEURISTIC_THRESHOLD = 1e5


def _comment_indicates_10us(header, key):
    """True if the header comment for `key` mentions 10-microsecond units."""
    try:
        comment = str(header.comments[key])
    except Exception:
        return False
    comment_lower = comment.lower()
    return any(x in comment_lower for x in ('10us', '10µs', '10 us', '10 µs'))


def _find_exptime_key(header):
    """Return the first EXPTIME-like keyword present with a numeric value."""
    for key in _EXPTIME_KEYS:
        if key in header:
            try:
                float(header[key])
                return key
            except (TypeError, ValueError):
                continue
    return None


def get_exptime_seconds(header, default=1.0):
    """
    Extract exposure time in SECONDS from a FITS header.

    Tries EXPTIME/EXP_TIME/EXPOSURE/EXPT/TEXPTIME/EXPOS in order. Converts
    from 10-microsecond ticks when the keyword's comment says so, or when the
    raw value exceeds 1e5 (magnitude heuristic).

    Args:
        header: FITS header object or None
        default: value returned when no usable keyword is found. Pass
                 default=None when the caller needs to distinguish
                 "missing" from a real exposure time.

    Returns:
        float exposure time in seconds, or `default`.
    """
    if header is None:
        return default

    key = _find_exptime_key(header)
    if key is None:
        return default

    val = float(header[key])
    if _comment_indicates_10us(header, key) or val > _TICKS_HEURISTIC_THRESHOLD:
        return round(val * _TICKS_TO_SECONDS, 6)
    return round(val, 6)


def _format_seconds(exptime_s):
    """Human formatting: µs below 1 ms, ms below 1 s, else seconds."""
    if exptime_s < 0.001:
        return f"{exptime_s * 1e6:.0f}µs"
    elif exptime_s < 1.0:
        return f"{exptime_s * 1000:.2f}ms"
    return f"{exptime_s:.4g}s"


def format_exptime(header):
    """
    Extract and format exposure time from a FITS header for display
    (e.g. "30µs", "1.50ms", "93.99s"). Unit detection is shared with
    get_exptime_seconds.
    """
    if header is None:
        return "1.0s"
    try:
        exptime_s = get_exptime_seconds(header, default=None)
        if exptime_s is None:
            return "N/A"
        return _format_seconds(exptime_s)
    except Exception:
        return "N/A"


def format_exptime_from_raw(raw_value, is_10us_units=True):
    """
    Format exposure time from a raw numeric value with unit specification.
    Works with raw values instead of headers.
    
    Args:
        raw_value: Numeric exposure time value
        is_10us_units: If True, assumes value is in 10µs units and converts to seconds.
                       If False, assumes value is already in seconds.
                       Default: True (since most CMOS sensor files use 10µs units)
    
    Returns:
        str: Formatted exposure time string (e.g., "30µs", "1.50ms", "93.99s")
    """
    try:
        val = float(raw_value)
        exptime_s = val * _TICKS_TO_SECONDS if is_10us_units else val
        return _format_seconds(exptime_s)
    except Exception:
        return "N/A"
