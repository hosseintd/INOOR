import numpy as np
from astropy.visualization import ZScaleInterval
import time

class ImageDisplayCache:
    """
    Caches downsampled versions of large FITS images to speed up UI rendering
    without affecting the precision of scientific calculations.
    """
    def __init__(self, max_display_dim=1536):
        self.max_display_dim = max_display_dim
        self.cache = {} # path -> {display_data, vmin, vmax, scale, original_shape}
        self.max_cache_size = 10 # Number of frames to keep in RAM

    def get_display_data(self, path, original_data):
        if path in self.cache:
            return self.cache[path]

        # Calculate downsampling factor
        ny, nx = original_data.shape
        max_dim = max(ny, nx)
        scale = 1
        if max_dim > self.max_display_dim:
            scale = int(np.ceil(max_dim / self.max_display_dim))
            # Use slicing for fast downsampling
            display_data = original_data[::scale, ::scale].copy()
        else:
            display_data = original_data.copy()

        # Calculate ZScale on the downsampled image (much faster)
        try:
            # Save random state for determinism
            state = np.random.get_state()
            try:
                np.random.seed(42)
                interval = ZScaleInterval()
                vmin, vmax = interval.get_limits(display_data)
            finally:
                np.random.set_state(state)
            
            if vmin is None or vmax is None:
                vmin, vmax = np.nanmin(display_data), np.nanmax(display_data)
                
        except Exception:
            vmin, vmax = np.nanmin(display_data), np.nanmax(display_data)

        entry = {
            'display_data': display_data,
            'vmin': float(vmin),
            'vmax': float(vmax),
            'scale': scale,
            'original_shape': (ny, nx)
        }

        # Manage cache size
        if len(self.cache) >= self.max_cache_size:
            # Simple "pop first" strategy
            first_key = next(iter(self.cache))
            self.cache.pop(first_key)

        self.cache[path] = entry
        return entry

    def clear(self):
        self.cache.clear()
