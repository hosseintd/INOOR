"""
Simple transient detection via image subtraction and catalog cross-match.
"""
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from astropy.table import Table

class TransientDetector:
    def __init__(self, science_image_processor):
        self.science = science_image_processor

    def subtract_reference(self, reference_path, smooth_sigma=1.0):
        """
        Load the reference FITS, roughly align shapes (simple trim) and subtract.
        """
        with fits.open(reference_path, ignore_missing_end=True) as rh:
            ref = None
            for h in rh:
                if getattr(h, 'data', None) is not None:
                    ref = h.data
                    break
            if ref is None:
                ref = rh[0].data
        sci = self.science.data.astype(float)
        ref = ref.astype(float)
        # align shapes by trimming to overlap
        minshape = (min(ref.shape[0], sci.shape[0]), min(ref.shape[1], sci.shape[1]))
        ref_c = ref[:minshape[0], :minshape[1]]
        sci_c = sci[:minshape[0], :minshape[1]]
        # subtract medians
        ref_s = ref_c - np.median(ref_c)
        sci_s = sci_c - np.median(sci_c)
        diff = sci_s - gaussian_filter(ref_s, sigma=smooth_sigma)
        return diff

    def detect_sources_threshold(self, diff_image, threshold_sigma=5.0, min_area=1):
        """
        Simple threshold detection.
        """
        std = np.nanstd(diff_image)
        if std == 0 or np.isnan(std):
            std = 1.0
        thresh = threshold_sigma * std
        mask = diff_image > thresh
        y, x = np.nonzero(mask)
        flux = diff_image[y, x]
        tbl = Table()
        tbl['x'] = x
        tbl['y'] = y
        tbl['flux'] = flux
        return tbl
