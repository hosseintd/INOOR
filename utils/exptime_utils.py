"""
Exposure time detection and formatting utilities.
Handles various FITS header formats for exposure time and converts to appropriate units.
"""

def _detect_unit_from_header(header):
    """
    Helper function to detect if EXPTIME is in 10µs units from header comment.
    
    Args:
        header: FITS header object or None
        
    Returns:
        bool: True if "10us" or "10µs" found in EXPTIME comment, False otherwise
    """
    if header is None:
        return False
    
    try:
        # Try to get the EXPTIME comment using bracket notation
        comment = ""
        if hasattr(header, 'comments'):
            try:
                comment = str(header.comments['EXPTIME'])
            except (KeyError, TypeError):
                # If bracket notation fails, try other methods
                try:
                    # Some headers might have comments as a dict-like with get()
                    if hasattr(header.comments, 'get'):
                        comment = str(header.comments.get('EXPTIME', ''))
                except:
                    pass
        
        # Check for various forms of 10µs unit specification (case-insensitive)
        comment_lower = comment.lower()
        if any(x in comment_lower for x in ['10us', '10µs', '10 us', '10 µs', '10us periods']):
            print(f"[DEBUG] Detected 10µs units from comment: {comment}")
            return True
    except Exception as e:
        print(f"[DEBUG] Error detecting unit from header: {e}")
        pass
    
    return False


def get_exptime_seconds(header):
    """
    Extract exposure time in seconds from FITS header.
    Detects unit from header comment: checks for "10us" or "10µs" keywords.
    If comment indicates 10µs units, converts to seconds.
    If no comment or comment doesn't mention 10µs, assumes value is already in seconds.
    
    Args:
        header: FITS header object or None
        
    Returns:
        float: Exposure time in seconds
    """
    if header is None:
        return 1.0
    
    et = header.get('EXPTIME', header.get('EXP_TIME', header.get('EXPOSURE', 1.0)))
    try:
        val = float(et)
        
        # Use helper to detect unit from header comment
        is_10us_units = _detect_unit_from_header(header)
        
        # Convert if in 10µs units, otherwise assume already in seconds
        if is_10us_units:
            return val * 0.00001  # Convert from 10µs to seconds
        else:
            return val
    except:
        return 1.0


def format_exptime(header):
    """
    Extract and format exposure time from FITS header with appropriate units.
    
    Detects unit from header comment: checks for "10us" or "10µs" keywords.
    If in 10µs units, converts to seconds for internal storage.
    Then formats for display:
    - Small values (< 1 second): shows in microseconds (µs) or milliseconds (ms)
    - Large values (≥ 1 second): shows in seconds (s)
    
    Args:
        header: FITS header object or None
        
    Returns:
        str: Formatted exposure time string (e.g., "30µs", "1.50ms", "93.99s")
    """
    if header is None:
        return "1.0s"
    
    et = header.get('EXPTIME', header.get('EXP_TIME', header.get('EXPOSURE', 1.0)))
    try:
        val = float(et)
        
        # Use helper to detect unit from header comment
        is_10us_units = _detect_unit_from_header(header)
        
        # Convert if in 10µs units
        if is_10us_units:
            exptime_s = val * 0.00001  # Convert from 10µs to seconds
        else:
            exptime_s = val  # Already in seconds
        
        # Format based on magnitude
        if exptime_s < 0.001:  # Less than 1ms
            exptime_us = exptime_s * 1e6
            return f"{exptime_us:.0f}µs"
        elif exptime_s < 1.0:  # Less than 1s
            exptime_ms = exptime_s * 1000
            return f"{exptime_ms:.2f}ms"
        else:
            return f"{exptime_s:.4g}s"
    except:
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
        
        # Convert if in 10µs units
        if is_10us_units:
            exptime_s = val * 0.00001  # Convert from 10µs to seconds
        else:
            exptime_s = val  # Already in seconds
        
        # Format based on magnitude
        if exptime_s < 0.001:  # Less than 1ms
            exptime_us = exptime_s * 1e6
            return f"{exptime_us:.0f}µs"
        elif exptime_s < 1.0:  # Less than 1s
            exptime_ms = exptime_s * 1000
            return f"{exptime_ms:.2f}ms"
        else:
            return f"{exptime_s:.4g}s"
    except:
        return "N/A"
