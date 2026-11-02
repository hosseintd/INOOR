"""
The question asked when Calibration starts without Flat frames.

The answer No used to close the question and do nothing, with no message, so START CALIBRATION
looked broken. The real controller runs here; only the dialogs and the worker are stand-ins.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits
from PyQt5 import QtWidgets


@pytest.fixture(scope="module")
def app():
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Signal:
    def connect(self, slot):
        pass


class _FakeWorker:
    made = []

    def __init__(self, sets, out_dir, **kw):
        self.kw = kw
        self.progress = _Signal()
        self.finished = _Signal()
        _FakeWorker.made.append(self)

    def requestInterruption(self):
        pass

    def isRunning(self):
        return False

    def start(self):
        pass


def _tab(tmp_path, monkeypatch, answer, set_types):
    import controllers.calibration_controller as cc
    from models.file_set import FileSet
    from models.session_model import SessionModel
    from views.calibration_view import CalibrationView

    _FakeWorker.made = []
    asked, shown = [], []
    monkeypatch.setattr(cc, "CalibrationWorker", _FakeWorker)
    monkeypatch.setattr(cc.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **kw: str(tmp_path)))
    monkeypatch.setattr(cc.CalibrationSummaryDialog, "exec_",
                        lambda self: QtWidgets.QDialog.Accepted)
    monkeypatch.setattr(cc.QMessageBox, "question",
                        staticmethod(lambda parent, title, text, *a: (asked.append((title, text)), answer)[1]))
    monkeypatch.setattr(cc.QMessageBox, "information",
                        staticmethod(lambda parent, title, text, *a: shown.append((title, text))))
    model = SessionModel()
    ctrl = cc.CalibrationController(model, CalibrationView())
    for set_type in set_types:
        path = tmp_path / f"{set_type}.fits"
        fits.writeto(path, np.full((32, 32), 100.0, np.float32), overwrite=True)
        model.add_set(FileSet(set_type=set_type, files=[str(path)]))
    return ctrl, asked, shown


def test_no_starts_nothing_and_says_so(app, tmp_path, monkeypatch):
    ctrl, asked, shown = _tab(tmp_path, monkeypatch, QtWidgets.QMessageBox.No, ["Light", "Dark", "Bias"])
    ctrl.start_calibration()
    assert len(asked) == 1
    assert _FakeWorker.made == []
    assert [title for title, _ in shown] == ["Calibration Not Started"]
    assert "Load Flat frames" in shown[0][1]
    assert ctrl.view.btn_start.isEnabled()


def test_yes_runs_with_the_auto_flat_consent(app, tmp_path, monkeypatch):
    ctrl, asked, shown = _tab(tmp_path, monkeypatch, QtWidgets.QMessageBox.Yes, ["Light", "Dark", "Bias"])
    ctrl.start_calibration()
    assert len(asked) == 1
    assert shown == []
    assert len(_FakeWorker.made) == 1
    assert _FakeWorker.made[0].kw["auto_flat"] is True


def test_the_question_names_what_each_answer_does(app, tmp_path, monkeypatch):
    ctrl, asked, _ = _tab(tmp_path, monkeypatch, QtWidgets.QMessageBox.No, ["Light", "Dark", "Bias"])
    ctrl.start_calibration()
    _, text = asked[0]
    assert "Yes: each Light is divided by an auto-flat made from that frame itself." in text
    assert "No: calibration does not start." in text


def test_flats_loaded_means_no_question_and_no_auto_flat(app, tmp_path, monkeypatch):
    ctrl, asked, shown = _tab(tmp_path, monkeypatch, QtWidgets.QMessageBox.No,
                              ["Light", "Dark", "Bias", "Flat"])
    ctrl.start_calibration()
    assert asked == []
    assert shown == []
    assert _FakeWorker.made[0].kw["auto_flat"] is False
