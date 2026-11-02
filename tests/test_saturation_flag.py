"""
Stars beyond the KL4040's linear range, or clipped at the 12-bit ceiling,
are flagged instead of being reported with a tiny error bar.

The only measured linear limit is High 1x1: 3500 ADU per read on calibrated
frames (on-sky, 2026-08-25). Other modes are checked against the ceiling only.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from astropy.io import fits

from core import camera_config as cc
from core import photometry_core as pc

SIGMA = 12.0 / 2.3548


def _star(peak, clip=None, shape=(201, 201)):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    img = 40.0 + peak * np.exp(-((xx - 100) ** 2 + (yy - 100) ** 2) / (2 * SIGMA ** 2))
    return np.minimum(img, clip) if clip is not None else img


def test_limits_are_only_the_measured_ones():
    assert cc.KL4040_HIGH.linear_limit_adu == 3500.0
    for m in (cc.KL4040_LOW, cc.KL4040_MERGE, cc.KL4040_HIGH_2X2, cc.KL4040_LOW_2X2):
        assert m.linear_limit_adu is None
    assert cc.ADC_CEILING_ADU == 4095


def test_peak_status_levels():
    assert pc.peak_status(_star(2800), (100, 100), 12, 3500)[1] == 'ok'
    assert pc.peak_status(_star(3700), (100, 100), 12, 3500)[1] == 'nonlinear'
    assert pc.peak_status(_star(3700), (100, 100), 12, None)[1] == 'ok'
    peak, status = pc.peak_status(_star(9000, clip=4095), (100, 100), 12, None)
    assert status == 'saturated' and peak == 4095


def test_measure_star_reports_it():
    m = pc.measure_star(_star(3700), (100.0, 100.0), aperture_override=12.0,
                        linear_limit=3500.0, n_radii=20)
    assert m['phot']['saturation'] == 'nonlinear'
    assert abs(m['phot']['peak_adu'] - 3740.0) < 1.0


def test_bulk_uses_the_frame_modes_limit(tmp_path):
    from core.multi_photometry_worker import BulkPhotometryWorker
    paths = []
    for name, imgtype in (("b_High_1.fits", 1), ("b_Low_1.fits", 3)):
        hdr = fits.Header()
        hdr["EXPTIME"] = 60.0
        hdr["IMGTYPE"] = imgtype                  # 1 = High, 3 = Low
        fits.writeto(tmp_path / name, _star(3700).astype(np.float32), hdr)
        paths.append(str(tmp_path / name))
    for p, expected in zip(paths, ('nonlinear', 'ok')):
        w = BulkPhotometryWorker([p], 0, (100.0, 100.0), fwhm=12.0, aperture_override=12.0,
                                 max_radii_samples=20)
        got = []
        w.done.connect(got.append)
        w.run()
        assert got[0][0]['aperture_result']['saturation'] == expected, p


def test_bulk_summary_and_csv_name_the_flagged_frames(tmp_path, monkeypatch):
    import csv
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import controllers.multi_photometry_controller as mpc
    from core.multi_photometry_worker import BulkPhotometryWorker
    from models.multi_photometry_model import MultiPhotometryModel
    from views.multi_photometry_view import MultiPhotometryView

    hdr = fits.Header()
    hdr["EXPTIME"] = 60.0
    hdr["IMGTYPE"] = 1
    p = str(tmp_path / "s_High_1.fits")
    fits.writeto(p, _star(3700).astype(np.float32), hdr)
    w = BulkPhotometryWorker([p], 0, (100.0, 100.0), fwhm=12.0, aperture_override=12.0,
                             max_radii_samples=20)
    got = []
    w.done.connect(got.append)
    w.run()

    model = MultiPhotometryModel()
    ctrl = mpc.MultiPhotometryController(model, MultiPhotometryView())
    model.add_files([p])
    shown = []
    monkeypatch.setattr(mpc.QMessageBox, "warning",
                        staticmethod(lambda parent, title, text, *a: shown.append(text)))
    monkeypatch.setattr(mpc.QMessageBox, "information",
                        staticmethod(lambda parent, title, text, *a: shown.append("INFO " + text)))
    ctrl.progress_dlg = QtWidgets.QProgressDialog()
    ctrl.on_bulk_done(got[0])
    assert shown and "beyond the linear range on 1 frame" in shown[0]
    assert "s_High_1.fits: NON-LINEAR: peak 3740 ADU" in shown[0]

    out = tmp_path / "out.csv"
    monkeypatch.setattr(mpc.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "CSV (*.csv)")))
    ctrl.save_csv()
    rows = list(csv.DictReader(open(out)))
    assert rows[0]["Saturation"] == "nonlinear" and abs(float(rows[0]["PeakADU"]) - 3740) < 1
