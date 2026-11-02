"""
Cancel during a plate solve.

Cancel cannot stop astrometry.net, and the worker waits for the job anyway. Its result used to
arrive minutes later and be applied as if nothing had happened: a "Success" box, and the viewer
switched to the cancelled frame's solution while the model still held the frame the operator had
opened since. No network here: the remote solve is a short stand-in.
"""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits
from PyQt5 import QtWidgets


@pytest.fixture(scope="module")
def app():
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _tab(tmp_path, monkeypatch):
    import controllers.astrometry_controller as ac
    import core.astrometry.astrometry_solver as sm
    from models.astrometry_model import AstrometryModel
    from views.astrometry_view import AstrometryView

    paths = {}
    for name, value in (("frame_A", 0.0), ("frame_B", 1.0), ("solved_frame_A", 2.0)):
        paths[name] = str(tmp_path / f"{name}.fits")
        fits.writeto(paths[name], np.full((32, 32), value, np.float32))

    def remote_solve(self, fits_path, **kw):
        time.sleep(0.5)
        return {"solved_fits": paths["solved_frame_A"], "submission_id": 1, "job_id": 2,
                "rdls_path": None, "axy_path": None, "downsample_factor": 1}

    boxes = []
    monkeypatch.setattr(sm.AstrometrySolver, "solve_from_file", remote_solve)
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(ac.QMessageBox, kind,
                            staticmethod(lambda *a, _k=kind: boxes.append((_k, a[1]))))
    model, view = AstrometryModel(), AstrometryView()
    ctrl = ac.AstrometryController(model, view)
    model.set_current_file(paths["frame_A"])
    view.viewer.load_fits(paths["frame_A"])
    return model, view, ctrl, paths, boxes


def _wait_for(app, worker):
    t0 = time.time()
    while worker.isRunning() and time.time() - t0 < 10:
        app.processEvents()
        time.sleep(0.01)
    for _ in range(20):          # deliver the queued finished/error signal
        app.processEvents()


def test_a_cancelled_solve_does_not_replace_the_frame_opened_since(app, tmp_path, monkeypatch):
    model, view, ctrl, paths, boxes = _tab(tmp_path, monkeypatch)
    ctrl.on_solve_plate({"api_key": "x", "downsample": 1})
    worker = ctrl.worker
    # the real button: QProgressDialog.cancel() called directly does not emit canceled()
    ctrl.progress.findChild(QtWidgets.QPushButton).click()
    model.set_current_file(paths["frame_B"])
    view.viewer.load_fits(paths["frame_B"])
    _wait_for(app, worker)

    assert view.viewer.title.text() == "frame_B.fits"
    assert model.solved_file is None
    assert not any(title == "Success" for _, title in boxes)
    assert "cancelled" in view.info_box.toPlainText().lower()


def test_a_solve_that_is_not_cancelled_is_still_shown(app, tmp_path, monkeypatch):
    model, view, ctrl, paths, boxes = _tab(tmp_path, monkeypatch)
    ctrl.on_solve_plate({"api_key": "x", "downsample": 1})
    _wait_for(app, ctrl.worker)

    assert view.viewer.title.text() == "solved_frame_A.fits"
    assert model.solved_file == paths["solved_frame_A"]
    assert ("information", "Success") in boxes
    # QProgressDialog.close() emits canceled(); closing it after a solve is not a Cancel
    assert "cancelled" not in view.info_box.toPlainText().lower()
    assert not ctrl.worker.cancelled


def test_success_names_the_whole_path_of_the_solved_file(app, tmp_path, monkeypatch):
    import controllers.astrometry_controller as ac

    model, view, ctrl, paths, _ = _tab(tmp_path, monkeypatch)
    texts = []
    monkeypatch.setattr(ac.QMessageBox, "information",
                        staticmethod(lambda parent, title, text: texts.append((title, text))))
    ctrl.on_solve_plate({"api_key": "x", "downsample": 1})
    _wait_for(app, ctrl.worker)

    assert [t for t, _ in texts] == ["Success"]
    assert os.path.abspath(paths["solved_frame_A"]) in texts[0][1]
