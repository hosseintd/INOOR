"""
The info box of the Astrometry tab describes the frame on screen. Opening a frame used to leave the
previous frame's PS1 magnitudes there, and a PS1 answer still on its way would have landed under the
new frame. No network: the cone search is a slow stub, the worker and its thread are the real ones.
"""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from PyQt5 import QtWidgets


@pytest.fixture(scope="module")
def app():
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _answer(ra, dec, radius_deg, **kw):
    time.sleep(0.3)
    return Table({'objID': [1], 'raMean': [ra], 'decMean': [dec], 'nDetections': [20],
                  'gMeanPSFMag': [17.805], 'rMeanPSFMag': [17.305], 'iMeanPSFMag': [17.090],
                  'zMeanPSFMag': [16.993], 'yMeanPSFMag': [16.902]})


def _offline(*a, **kw):
    time.sleep(0.3)
    raise RuntimeError("the query failed")


def _tab(tmp_path, monkeypatch, ps1):
    import controllers.astrometry_controller as ac
    import controllers.astrometry_workers as aw
    from models.astrometry_model import AstrometryModel
    from views.astrometry_view import AstrometryView

    paths = {}
    for name in ("frame_A", "frame_B"):
        paths[name] = str(tmp_path / f"{name}.fits")
        fits.writeto(paths[name], np.zeros((32, 32), np.float32))
    monkeypatch.setattr(aw, "ps1cone", ps1)
    model, view = AstrometryModel(), AstrometryView()
    ctrl = ac.AstrometryController(model, view)

    def open_frame(name):
        monkeypatch.setattr(ac.QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **kw: (paths[name], "")))
        ctrl.on_open_fits()

    open_frame("frame_A")
    view.viewer.last_picked_world = SkyCoord(333.88 * u.deg, 51.60 * u.deg)
    return view, ctrl, open_frame


def _wait_for(app, worker):
    t0 = time.time()
    while worker.isRunning() and time.time() - t0 < 10:
        app.processEvents()
        time.sleep(0.01)
    for _ in range(20):          # deliver the queued signal
        app.processEvents()


def test_opening_a_frame_clears_the_box(app, tmp_path, monkeypatch):
    view, ctrl, open_frame = _tab(tmp_path, monkeypatch, _answer)
    view.info_box.setText("--- PS1 Match ---\ng: 17.805")
    open_frame("frame_B")
    assert view.info_box.toPlainText() == ""


def test_the_answer_is_shown_when_the_frame_is_unchanged(app, tmp_path, monkeypatch):
    view, ctrl, _ = _tab(tmp_path, monkeypatch, _answer)
    ctrl.on_query_ps1()
    _wait_for(app, ctrl.cat_worker)
    assert "g: 17.805" in view.info_box.toPlainText()


def test_a_late_answer_does_not_land_under_the_new_frame(app, tmp_path, monkeypatch):
    view, ctrl, open_frame = _tab(tmp_path, monkeypatch, _answer)
    ctrl.on_query_ps1()
    worker = ctrl.cat_worker
    open_frame("frame_B")
    _wait_for(app, worker)
    assert view.info_box.toPlainText() == ""


def test_a_late_error_does_not_land_under_the_new_frame(app, tmp_path, monkeypatch):
    view, ctrl, open_frame = _tab(tmp_path, monkeypatch, _offline)
    ctrl.on_query_ps1()
    worker = ctrl.cat_worker
    open_frame("frame_B")
    _wait_for(app, worker)
    assert view.info_box.toPlainText() == ""


def test_an_error_is_shown_when_the_frame_is_unchanged(app, tmp_path, monkeypatch):
    view, ctrl, _ = _tab(tmp_path, monkeypatch, _offline)
    ctrl.on_query_ps1()
    _wait_for(app, ctrl.cat_worker)
    assert "Query error: the query failed" in view.info_box.toPlainText()
