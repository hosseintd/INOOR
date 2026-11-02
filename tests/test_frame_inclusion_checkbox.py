"""
A ticked box means "use this frame".

The calibration tree used the opposite convention: every frame arrived
UNticked, and you ticked one to CONDEMN it. That reads backwards -- a tick
normally means "included" -- so the natural gesture silently threw frames out
of the master.

These tests drive the real QTreeWidget through CalibrationView.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5 import QtCore, QtWidgets

from views.calibration_view import CalibrationView


@pytest.fixture(scope="module")
def app():
    existing = QtWidgets.QApplication.instance()
    yield existing or QtWidgets.QApplication([])


class _Set:
    """Stand-in for models.file_set.FileSet."""
    def __init__(self, set_type, files, bad_indices=None):
        self.set_type = set_type
        self.files = list(files)
        self.bad_indices = set(bad_indices or ())
        self.method = 'median'
        self.do_sigma_clip = False
        self.sigma_lower = 3.0
        self.sigma_upper = 3.0


def _children(view):
    parent = view.tree_data.topLevelItem(0)
    return [parent.child(i) for i in range(parent.childCount())]


def test_new_frames_arrive_included(app):
    view = CalibrationView()
    view.update_tree([_Set('Dark', ['a.fits', 'b.fits', 'c.fits'])])

    states = [c.checkState(0) for c in _children(view)]
    assert states == [QtCore.Qt.Checked] * 3, (
        "a freshly loaded frame takes part in calibration, so it must be ticked")


def test_excluded_frames_are_unticked_and_red(app):
    view = CalibrationView()
    view.update_tree([_Set('Dark', ['a.fits', 'b.fits', 'c.fits'], bad_indices={1})])

    kids = _children(view)
    assert kids[0].checkState(0) == QtCore.Qt.Checked
    assert kids[1].checkState(0) == QtCore.Qt.Unchecked
    assert kids[2].checkState(0) == QtCore.Qt.Checked

    red = kids[1].foreground(0).color().name()
    normal = kids[0].foreground(0).color().name()
    assert red != normal, "an excluded frame must be visually distinct"


def test_unticking_reports_the_frame_as_excluded(app):
    view = CalibrationView()
    view.update_tree([_Set('Flat', ['a.fits', 'b.fits'])])

    seen = []
    view.bad_frame_toggled.connect(lambda t, i, bad: seen.append((t, i, bad)))

    _children(view)[1].setCheckState(0, QtCore.Qt.Unchecked)
    assert seen == [('Flat', 1, True)], (
        f"unticking must exclude the frame; got {seen}")

    seen.clear()
    _children(view)[1].setCheckState(0, QtCore.Qt.Checked)
    assert seen == [('Flat', 1, False)], (
        f"re-ticking must bring the frame back; got {seen}")


def test_the_header_says_what_the_box_means(app):
    view = CalibrationView()
    assert "Use" in view.tree_data.headerItem().text(0)
