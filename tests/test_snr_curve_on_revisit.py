"""
Revisiting a frame whose SNR curve was cleared must rebuild the SAME curve.

Changing Fit FWHM clears every frame's curve; on_selection_changed then
rebuilt it with the High-mode noise model, fwhm 12 and the default 2r-3r
annulus, whatever the frame's mode and the user's annulus were. On a Low
frame that is a different curve and a different r_best from the
measurement shown next to it.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits
from PyQt5 import QtWidgets

from core import photometry_core as pc
from core.camera_config import KL4040_LOW
from models.multi_photometry_model import MultiPhotometryModel
from views.multi_photometry_view import MultiPhotometryView
from controllers.multi_photometry_controller import MultiPhotometryController


@pytest.fixture(scope="module")
def app():
    existing = QtWidgets.QApplication.instance()
    yield existing or QtWidgets.QApplication([])


def test_rebuilt_curve_uses_the_frames_mode_and_annulus(app, tmp_path):
    rng = np.random.default_rng(20261009)
    yy, xx = np.mgrid[0:401, 0:401]
    img = 8.0 + 60.0 * np.exp(-((xx - 200) ** 2 + (yy - 200) ** 2) / (2 * 5.0 ** 2))
    img = (img + rng.normal(0.0, 2.0, img.shape)).astype(np.float32)
    path = tmp_path / "frame_Low_1.fits"
    hdr = fits.Header()
    hdr["IMGTYPE"] = 3            # KL4040 Low channel
    fits.writeto(path, img, hdr)

    model = MultiPhotometryModel()
    view = MultiPhotometryView()
    ctrl = MultiPhotometryController(model, view)
    model.inner_coef, model.outer_coef = 3.0, 5.0
    model.fwhm = 10.0
    model.add_files([str(path)])
    d = model._frame_data[0]
    d.selected_source = (200.0, 200.0)
    d.aperture_radius = 10.0
    d.radii = d.snrs = d.r_best = None      # what a Fit FWHM change leaves

    ctrl.on_selection_changed(0)

    radii, snrs, r_best, _ = pc.compute_snr_vs_radius(
        img, (200.0, 200.0), fwhm=10.0,
        gain=KL4040_LOW.gain, read_noise=KL4040_LOW.read_noise,
        dark_noise=KL4040_LOW.dark_noise, inner_coef=3.0, outer_coef=5.0)
    assert np.allclose(d.radii, radii)
    assert np.allclose(d.snrs, snrs)
    assert d.r_best == pytest.approx(r_best)
