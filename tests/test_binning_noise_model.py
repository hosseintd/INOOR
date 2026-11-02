"""
Binned frames get the noise parameters measured for their binning.

The KL4040's 2x2 output is 0.400 (High) / 0.473 (Low) x the sum of four
pixels, so its effective gain is ~2.5x the unbinned one: 2x2 data reduced with
the 1x1 values had error bars wrong by about that factor. No other binning has
been measured, and for those INOOR must say so rather than use 1x1 values.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits
from PyQt5 import QtWidgets

from core import camera_config as cc


@pytest.fixture(scope="module")
def app():
    existing = QtWidgets.QApplication.instance()
    yield existing or QtWidgets.QApplication([])


def _hdr(**kw):
    h = fits.Header()
    for k, v in kw.items():
        h[k] = v
    return h


def test_2x2_frames_get_the_measured_2x2_values():
    high = cc.detect_mode_or_none(_hdr(IMGTYPE=1, XBINNING=2, YBINNING=2), "x.fit")
    assert high is cc.KL4040_HIGH_2X2
    assert (high.gain, high.read_noise) == (2.314, 9.28)
    assert cc.detect_mode_or_none(_hdr(IMGTYPE=3, XBINNING=2, YBINNING=2), "x.fit") is cc.KL4040_LOW_2X2
    assert cc.detect_mode_or_none(_hdr(ISMRGFRM=True, XBINNING=2, YBINNING=2), None) is cc.KL4040_MERGE_2X2
    # A binned pixel collects the dark current of four.
    assert high.dark_charge(100.0) == pytest.approx(4 * 0.048 * 100.0)


def test_binning_read_from_the_filename_when_the_header_is_silent():
    assert cc.detect_mode_or_none(
        None, "Dark_2026_08_03_2x2_exp00.00.01.000_High_1.fit") is cc.KL4040_HIGH_2X2
    assert cc.detect_mode_or_none(None, "g_1x1_exp00.01.00_Low_1.fit") is cc.KL4040_LOW


def test_unmeasured_binning_is_not_given_1x1_values():
    assert cc.detect_mode_or_none(_hdr(IMGTYPE=1, XBINNING=3, YBINNING=3), "a.fit") is None
    assert cc.detect_mode_or_none(_hdr(IMGTYPE=1, XBINNING=1, YBINNING=2), "b.fit") is None
    # 1x1 stays exactly as before.
    assert cc.detect_mode_or_none(_hdr(IMGTYPE=1, XBINNING=1, YBINNING=1), "c.fit") is cc.KL4040_HIGH


def _frame(path, xbin):
    yy, xx = np.mgrid[0:201, 0:201]
    img = 40.0 + 500.0 * np.exp(-((xx - 100) ** 2 + (yy - 100) ** 2) / (2 * 4.0 ** 2))
    img += np.random.default_rng(5).normal(0.0, 3.0, img.shape)
    fits.writeto(path, img.astype(np.float32),
                 _hdr(EXPTIME=60.0, IMGTYPE=1, XBINNING=xbin, YBINNING=xbin))


def test_bulk_run_uses_2x2_values_and_flags_3x3(app, tmp_path, monkeypatch):
    import controllers.multi_photometry_controller as mpc
    from core.multi_photometry_worker import BulkPhotometryWorker
    from models.multi_photometry_model import MultiPhotometryModel
    from views.multi_photometry_view import MultiPhotometryView

    p2, p3 = str(tmp_path / "f_High_1.fits"), str(tmp_path / "f_High_2.fits")
    _frame(p2, 2)
    _frame(p3, 3)
    worker = BulkPhotometryWorker([p2, p3], 0, (100.0, 100.0), fwhm=9.4,
                                  detection_stamp_size=30, aperture_override=8.0)
    got = []
    worker.done.connect(got.append)
    worker.run()
    r2, r3 = got[0]
    assert r2["success"] and r3["success"]
    assert r2["camera_mode"] == "High 2x2" and r2["gain_used"] == pytest.approx(2.314)
    assert r3["camera_mode"] is None and tuple(r3["binning"]) == (3, 3)

    model = MultiPhotometryModel()
    ctrl = mpc.MultiPhotometryController(model, MultiPhotometryView())
    model.add_files([p2, p3])
    shown = []
    monkeypatch.setattr(mpc.QMessageBox, "warning",
                        staticmethod(lambda parent, title, text, *a: shown.append(text)))
    monkeypatch.setattr(mpc.QMessageBox, "information",
                        staticmethod(lambda parent, title, text, *a: shown.append(text)))
    ctrl.progress_dlg = QtWidgets.QProgressDialog()
    ctrl.on_bulk_done(got[0])
    assert shown and "3x3" in shown[0] and "not calibrated" in shown[0]
