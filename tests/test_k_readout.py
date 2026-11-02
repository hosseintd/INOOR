"""
The SNR plot labels the SNR peak with k = r_peak / FWHM_frame when the frame
FWHM is known, so the one-frame k recipe can be read off the plot.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from matplotlib.figure import Figure

from utils.gui_helpers import plot_profile_and_snr


def _texts(r_best, r_used, frame_fwhm):
    fp, fs = Figure(), Figure()
    r = np.linspace(1, 60, 60)
    plot_profile_and_snr(fp, fs, r, np.exp(-r / 10), r, 50 * np.exp(-((r - r_best) / 20) ** 2),
                         r_best, r_used=r_used, frame_fwhm=frame_fwhm)
    return [t.get_text() for t in fs.axes[0].texts]


def test_peak_apart_from_the_radius_carries_k():
    texts = _texts(15.0, 12.0, 12.5)
    assert " peak 15.0, k=1.20" in texts and " r=12.0" in texts


def test_radius_on_the_peak_carries_k():
    assert " r=15.0, k=1.20" in _texts(15.0, 15.0, 12.5)


def test_no_frame_fwhm_no_k():
    assert not any("k=" in t for t in _texts(15.0, 12.0, None))
