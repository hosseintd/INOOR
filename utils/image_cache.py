import threading

import numpy as np
from astropy.visualization import ZScaleInterval


class ImageDisplayCache:
    """
    Caches downsampled versions of large FITS images to speed up UI rendering
    without affecting the precision of scientific calculations.

    Thread safety: get_display_data is called both from the GUI thread and
    from InitialLoadWorker (a QThread), so cache reads/writes are guarded by
    a lock. Note: the previous implementation mutated the GLOBAL numpy RNG
    state (get_state/seed(42)/set_state) around the ZScale call -- a race
    condition when two threads hit it concurrently, and pointless anyway
    because ZScaleInterval subsamples deterministically without np.random.
    """
    def __init__(self, max_display_dim=1536):
        self.max_display_dim = max_display_dim
        self.cache = {}  # path -> {display_data, vmin, vmax, scale, original_shape}
        self.max_cache_size = 10  # Number of frames to keep in RAM
        self._lock = threading.Lock()

    def get_display_data(self, path, original_data):
        with self._lock:
            if path in self.cache:
                return self.cache[path]

        # Compute outside the lock (can be slow for 4k frames); a rare
        # concurrent duplicate computation is harmless.
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
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(display_data)
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

        with self._lock:
            # Manage cache size
            if len(self.cache) >= self.max_cache_size:
                # Simple "pop first" strategy
                first_key = next(iter(self.cache))
                self.cache.pop(first_key)
            self.cache[path] = entry

        return entry

    def clear(self):
        with self._lock:
            self.cache.clear()
