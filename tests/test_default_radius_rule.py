"""
The Multi Photometry tab defaults to the shared frame-FWHM radius rule, and
the view shows it before any frame is loaded -- otherwise the first edit of
any other parameter copied the view's first combo entry ("Max SNR") into
the model.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5 import QtWidgets


@pytest.fixture(scope="module")
def app():
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_default_rule_is_frame_fwhm_before_any_frame(app):
    import controllers.multi_photometry_controller as mpc
    from models.multi_photometry_model import MultiPhotometryModel
    from views.multi_photometry_view import MultiPhotometryView

    model = MultiPhotometryModel()
    assert model.aperture_policy == "frame_fwhm"
    view = MultiPhotometryView()
    mpc.MultiPhotometryController(model, view)
    assert view.aperture_policy() == "frame_fwhm"

    view.spin_inner.setValue(2.5)                  # any edit, still no frame loaded
    view.spin_inner.editingFinished.emit()
    assert model.aperture_policy == "frame_fwhm"


def test_zeropoints_warn_under_a_per_star_rule(app, tmp_path, monkeypatch):
    import numpy as np
    from astropy.io import fits
    import controllers.multi_photometry_controller as mpc
    from models.multi_photometry_model import MultiPhotometryModel
    from views.multi_photometry_view import MultiPhotometryView

    p = tmp_path / "z_High_1.fits"
    fits.writeto(p, np.zeros((64, 64), np.float32), fits.Header({"EXPTIME": 60.0}))
    model = MultiPhotometryModel()
    ctrl = mpc.MultiPhotometryController(model, MultiPhotometryView())
    model.add_files([str(p)])
    warned = []
    monkeypatch.setattr(mpc.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warned.append(a[2]) or mpc.QMessageBox.No))
    monkeypatch.setattr(mpc.QMessageBox, "information",
                        staticmethod(lambda *a, **k: mpc.QMessageBox.Cancel))

    model.aperture_policy = "snr"
    ctrl.calibrate_zeropoints()
    assert len(warned) == 1 and "own radius" in warned[0]
    assert not ctrl.zeropoint_calibration_mode

    model.aperture_policy = "frame_fwhm"
    ctrl.calibrate_zeropoints()
    assert len(warned) == 1                         # no warning for the shared rule
