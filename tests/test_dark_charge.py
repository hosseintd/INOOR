"""
The dark charge in the noise model is dark current x exposure, not a constant.

A fixed 4.15 e- was charged to every frame: about right for 86 s, but a 58 s
frame collects 2.8 e- and a 300 s frame 14.4 e- at the measured 0.048 e-/pix/s
(-10 C). The paper's Table 1 already describes it as a rate.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits
from PyQt5 import QtWidgets

from core import camera_config as cc
from core.multi_photometry_worker import BulkPhotometryWorker


@pytest.fixture(scope="module")
def app():
    existing = QtWidgets.QApplication.instance()
    yield existing or QtWidgets.QApplication([])


def test_dark_charge_scales_with_exposure():
    high = cc.KL4040_HIGH
    assert high.dark_charge(58.14) == pytest.approx(0.048 * 58.14)
    assert high.dark_charge(300.0) == pytest.approx(14.4)
    # Unknown exposure: the documented fallback, never zero.
    assert high.dark_charge(None) == high.dark_noise
    assert high.dark_charge(0.0) == high.dark_noise
    # Both channels read the same thermal electrons.
    assert cc.KL4040_LOW.dark_current == high.dark_current == cc.KL4040_MERGE.dark_current


def _frame(path, exptime):
    yy, xx = np.mgrid[0:201, 0:201]
    img = 40.0 + 500.0 * np.exp(-((xx - 100) ** 2 + (yy - 100) ** 2) / (2 * 4.0 ** 2))
    img += np.random.default_rng(3).normal(0.0, 3.0, img.shape)
    hdr = fits.Header()
    hdr["EXPTIME"] = exptime
    hdr["IMGTYPE"] = 1                       # KL4040 High channel
    fits.writeto(path, img.astype(np.float32), hdr)


def test_bulk_worker_charges_this_frames_exposure(tmp_path):
    p = str(tmp_path / "frame_High_1.fits")
    _frame(p, 100.0)
    worker = BulkPhotometryWorker([p], 0, (100.0, 100.0), fwhm=9.4,
                                  detection_stamp_size=30, aperture_override=8.0)
    got = []
    worker.done.connect(got.append)
    worker.run()
    (res,) = got[0]
    assert res["success"], res.get("msg")
    assert res["dark_noise_used"] == pytest.approx(4.8)


def test_interactive_path_charges_this_frames_exposure(app, tmp_path):
    from models.multi_photometry_model import MultiPhotometryModel
    from views.multi_photometry_view import MultiPhotometryView
    from controllers.multi_photometry_controller import MultiPhotometryController

    p = str(tmp_path / "frame_High_1.fits")
    _frame(p, 200.0)
    model = MultiPhotometryModel()
    ctrl = MultiPhotometryController(model, MultiPhotometryView())
    model.add_files([p])
    model._frame_data[0].exptime = 200.0
    gain, rn, dark, mode = ctrl._frame_camera_params(0)
    assert mode == "High"
    assert dark == pytest.approx(0.048 * 200.0)
