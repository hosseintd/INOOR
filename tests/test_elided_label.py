"""
A long frame name must not set the Multi Photometry tab's minimum width.

A plain QLabel's minimum width is its whole text: one 85-character KL4040
filename made the window at least 1397 px wide, cutting off the right panel
on a 1280 px half screen.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5 import QtWidgets

from views.components.elided_label import ElidedLabel
from views.multi_photometry_view import MultiPhotometryView

LONG_NAME = ("Calibrated_grb251013c_g_2025_10_14_1x1_exp00.01.00.000_000001_"
             "Low_1_20251014_003415.fit")


@pytest.fixture(scope="module")
def app():
    existing = QtWidgets.QApplication.instance()
    yield existing or QtWidgets.QApplication([])


def test_long_filename_does_not_widen_the_tab(app):
    view = MultiPhotometryView()
    before = view.minimumSizeHint().width()
    view.lbl_fname.setText(LONG_NAME * 3)
    assert view.minimumSizeHint().width() == before


def test_full_name_stays_readable(app):
    lbl = ElidedLabel()
    lbl.setText(LONG_NAME)
    lbl.resize(200, 20)
    assert lbl.fullText() == LONG_NAME
    assert lbl.toolTip() == LONG_NAME
    shown = lbl.text()
    assert shown != LONG_NAME and "…" in shown
    # Middle elision keeps both ends: the target and the frame number/time.
    assert shown.startswith("Cal") and shown.endswith(".fit")
