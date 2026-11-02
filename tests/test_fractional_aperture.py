"""
A manual aperture radius may carry two decimals (e.g. 7.25 px).

The Aperture (px) box used to be a QSpinBox and the controller cast its value
with int(), so a radius could only be a whole number of pixels even though
the photometry underneath already works with a float radius. These tests
drive the real widgets and check that the decimals survive every hop: box ->
model -> box, and into the measurement itself.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits
from PyQt5 import QtWidgets

from core import photometry_core as pc
from models.multi_photometry_model import MultiPhotometryModel
from views.multi_photometry_view import MultiPhotometryView
from controllers.multi_photometry_controller import MultiPhotometryController


@pytest.fixture(scope="module")
def app():
    existing = QtWidgets.QApplication.instance()
    yield existing or QtWidgets.QApplication([])


def test_view_box_accepts_two_decimals(app):
    view = MultiPhotometryView()
    view.spin_aperture.setValue(10.25)
    assert view.spin_aperture.value() == pytest.approx(10.25)
    # A third decimal is rounded to the box's two, not silently truncated to int.
    view.spin_aperture.setValue(7.126)
    assert view.spin_aperture.value() == pytest.approx(7.13)


def test_controller_keeps_the_decimals(app):
    model = MultiPhotometryModel()
    view = MultiPhotometryView()
    ctrl = MultiPhotometryController(model, view)

    view.spin_aperture.setValue(7.25)
    ctrl.update_params_from_view("aperture")
    assert model.aperture == pytest.approx(7.25)
    assert model.fixed_aperture, "typing a radius still means 'use this radius'"

    # And back from the model into a fresh view (e.g. after a theme/tab rebuild).
    view2 = MultiPhotometryView()
    view2.update_params(model)
    assert view2.spin_aperture.value() == pytest.approx(7.25)


def test_a_typed_fractional_radius_survives(app):
    """Typing "7.25" must give 7.25, with a frame loaded.

    Every change re-measures and update_params() rewrites the box, so with
    per-keystroke tracking the box turned "7" into "7.00" and the "." that
    followed was rejected: the radius stuck at 7.00.
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    model = MultiPhotometryModel()
    view = MultiPhotometryView()
    ctrl = MultiPhotometryController(model, view)
    # A current frame is what makes show_current_data -> update_params run.
    # The file does not exist; the controller logs the load error and goes on.
    model.add_files([os.path.join(os.path.dirname(__file__), "no_such_frame.fits")])

    box = view.spin_aperture
    box.setFocus()
    QTest.keyClick(box, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(box, "7.25")
    QTest.keyClick(box, Qt.Key_Return)
    assert box.value() == pytest.approx(7.25)
    assert model.aperture == pytest.approx(7.25)
    assert model.fixed_aperture


def test_measure_star_uses_the_fractional_radius():
    yy, xx = np.mgrid[0:101, 0:101]
    img = 100.0 + 5000.0 * np.exp(-((xx - 50) ** 2 + (yy - 50) ** 2) / (2 * 3.0 ** 2))
    m = pc.measure_star(img, (50.0, 50.0), fwhm=7.0, aperture_override=7.25,
                        zeropoint=0.0, exptime=1.0)
    assert m["r_used"] == pytest.approx(7.25)
    # photutils' exact overlap: the aperture area follows the float radius,
    # not the nearest integer (pi*7^2 = 153.9, pi*7.25^2 = 165.1).
    assert m["phot"]["aperture_area"] == pytest.approx(np.pi * 7.25 ** 2, rel=1e-3)


def test_calibration_review_dialog_keeps_the_decimals(app, tmp_path):
    from views.components.calibration_review_dialog import CalibrationReviewDialog

    path = str(tmp_path / "frame.fits")
    fits.writeto(path, np.full((64, 64), 100.0, dtype=np.float32))
    history = [{"name": "Ref", "results": {}}]
    dlg = CalibrationReviewDialog(None, [path], history,
                                  params={"aperture": 12.75, "fixed_aperture": True})
    assert dlg.spin_aperture.value() == pytest.approx(12.75)
    assert dlg._policy_label() == "fixed 12.75 px"
