import os
import sys

# Default configuration values and suggested astrometry scale presets (arcmin)
# API key is user-provided, no default. Get yours at: https://nova.astrometry.net/api_help
ASTROMETRY_API_KEY = os.environ.get("ASTROMETRY_API_KEY", "")
MAST_API_KEY = os.environ.get("MAST_API_KEY", "")

# Default astrometry scale presets (units: arcmin width)
SCALE_PRESETS = {
    "default": (6.0, 10800.0),        # 0.1 deg - 180 deg in arcmin
    "wide_field": (60.0, 10800.0),    # 1 deg - 180 deg
    "very_wide": (600.0, 10800.0),    # 10 deg - 180 deg
    "tiny": (2.0, 10.0),              # 2 - 10 arcmin
    "custom": (2.0, 10.0)
}

# Recommended worker timeout for astrometry solves (seconds)
ASTROMETRY_TIMEOUT = 900

# Catalog defaults
DEFAULT_CATALOGS = ["GAIA", "Pan-STARRS", "SDSS", "USNO"]


def get_project_root():
    """Return the project root directory.
    When frozen (PyInstaller), returns the exe's directory.
    When running as script, returns the project root (2 levels up from utils/)."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def get_export_dir():
    """Return the absolute path to the export directory, creating it if needed."""
    export_dir = os.path.join(get_project_root(), "Output", "Astrometry", "exports")
    os.makedirs(export_dir, exist_ok=True)
    return export_dir


# Keep EXPORT_DIR as a lazy-evaluated constant for backward compat
# NOTE: Do NOT create directories at import time — it causes stray folders
EXPORT_DIR = os.path.join("Output", "Astrometry", "exports")
