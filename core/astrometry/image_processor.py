"""
FITS image utilities: load, zscale, display scaling, simple manipulations.
"""
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS

class ImageProcessor:
    def __init__(self, path):
        self.path = path
        self.hdul = fits.open(path, ignore_missing_end=True, ignore_missing_simple=True)
        # assume primary HDU contains the image; simple fallback to first image HDU
        self.data = None
        self.header = None
        for h in self.hdul:
            if getattr(h, 'data', None) is not None:
                self.data = h.data
                self.header = h.header
                break
        if self.data is None:
            # fallback to first HDU
            self.data = self.hdul[0].data
            self.header = self.hdul[0].header
        # normalize to 2D if necessary (e.g., 3D cubes)
        if self.data is not None and self.data.ndim > 2:
            # pick first plane
            self.data = self.data[0]
        self.wcs = None
        try:
            self.wcs = WCS(self.header)
        except Exception:
            self.wcs = None

    def get_zscale_limits(self):
        try:
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(self.data)
            return float(vmin), float(vmax)
        except Exception:
            # fallback
            return float(np.nanmin(self.data)), float(np.nanmax(self.data))

    def get_display_image(self, vmin=None, vmax=None):
        """
        Return a normalized 2D numpy array suitable for plotting.
        - vmin/vmax in raw units (not normalized)
        - If None then ZScale is used.
        """
        vmin_auto, vmax_auto = self.get_zscale_limits()
        vmin = vmin if vmin is not None else vmin_auto
        vmax = vmax if vmax is not None else vmax_auto
        disp = self.data.astype(float).copy()
        disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            disp = np.clip(disp, vmin, vmax)
        except Exception:
            pass
        if vmax > vmin:
            disp = (disp - vmin) / (vmax - vmin)
        else:
            # avoid division by zero
            disp = disp - np.nanmin(disp)
            mx = np.nanmax(disp)
            if mx > 0:
                disp = disp / mx
        return disp

    def close(self):
        try:
            self.hdul.close()
        except Exception:
            pass
