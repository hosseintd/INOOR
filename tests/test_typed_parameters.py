"""
A decimal typed into a Multi Photometry parameter box must arrive intact.

Every parameter change ends in show_current_data -> update_params, which
calls setValue on every box in the panel, and QAbstractSpinBox reformats its
text on setValue even when the value is unchanged. With per-keystroke
tracking, typing "1.07" emitted 1.0 after the first digit, the box became
"1.00", and the "." was rejected -- so no decimal could be typed at all.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest

from models.multi_photometry_model import MultiPhotometryModel
from views.multi_photometry_view import MultiPhotometryView
from controllers.multi_photometry_controller import MultiPhotometryController


@pytest.fixture(scope="module")
def app():
    existing = QtWidgets.QApplication.instance()
    yield existing or QtWidgets.QApplication([])


def _tab_with_a_frame():
    model = MultiPhotometryModel()
    view = MultiPhotometryView()
    ctrl = MultiPhotometryController(model, view)
    # A current frame makes update_params run; the file need not exist.
    model.add_files([os.path.join(os.path.dirname(__file__), "no_such_frame.fits")])
    return model, view, ctrl


def _type_into(box, text):
    box.setFocus()
    QTest.keyClick(box, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(box, text)
    QTest.keyClick(box, Qt.Key_Return)


def test_fwhm_factor_keeps_its_decimals(app):
    model, view, ctrl = _tab_with_a_frame()
    view.cmb_ap_policy.setCurrentIndex(view.cmb_ap_policy.findData("frame_fwhm"))
    _type_into(view.spin_ap_param, "1.07")
    assert model.fwhm_factor == pytest.approx(1.07)


@pytest.mark.parametrize("box_name, text, attr", [
    ("spin_fwhm", "12.19", "fwhm"),
    ("spin_thresh", "2.5", "threshold"),
    ("spin_inner", "2.5", "inner_coef"),
    ("spin_outer", "3.5", "outer_coef"),
])
def test_typed_decimals_reach_the_model(app, box_name, text, attr):
    model, view, ctrl = _tab_with_a_frame()
    _type_into(getattr(view, box_name), text)
    assert getattr(model, attr) == pytest.approx(float(text))
