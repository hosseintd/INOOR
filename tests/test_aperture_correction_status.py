"""
The aperture correction must say when it could not be applied, and must not
be built from impossible curves of growth.

- When a frame's curve of growth cannot be measured the correction used to
  become 0.0 silently, so a run labelled "+apcorr" mixed total and aperture
  magnitudes with nothing to tell them apart.
- frame_growth_curve accepted curves whose enclosed fraction rose above 1
  (up to 8.9 on PSR J2215 frame High_1), which pulled E(1.05 FWHM) from 0.794
  to 0.855 -- an ~80 mmag error in the correction.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits
from PyQt5 import QtWidgets

from core import photometry_core as pc

FWHM = 4.0
SIG = FWHM / 2.3548


def _field(n_side=4, size=1024, seed=7):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    img = 50.0 + rng.normal(0.0, 3.0, (size, size))
    pos = [(150 + i * 230.0, 150 + j * 230.0) for i in range(n_side) for j in range(n_side)]
    for x, y in pos:
        img += 3000.0 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * SIG ** 2))
    return img, pos


def test_curves_above_one_are_not_used(monkeypatch):
    img, _ = _field()
    real = pc.curve_of_growth
    calls = {"n": 0}

    def half_of_them_broken(*a, **k):
        r, f = real(*a, **k)
        calls["n"] += 1
        if calls["n"] % 2:                 # every other star: E rises to 1.3 and falls back
            f = f.copy()
            f[len(f) // 2:] = np.maximum(f[len(f) // 2:], 1.3)
        return r, f

    monkeypatch.setattr(pc, "curve_of_growth", half_of_them_broken)
    radii, frac = pc.frame_growth_curve(img, FWHM)
    assert frac is not None
    assert float(np.max(frac)) <= 1.02


def test_unusable_curve_gives_none_not_zero():
    assert pc.aperture_correction(None, None, 5.0) is None
    r = np.arange(1.0, 20.0)
    assert pc.aperture_correction(r, np.full(r.size, np.nan), 5.0) is None
    good = pc.aperture_correction(r, np.linspace(0.3, 1.0, r.size), 5.0)
    assert good is not None and good < 0


def test_measure_star_reports_the_status():
    img, pos = _field()
    x, y = pos[5]
    off = pc.measure_star(img, (x, y), fwhm=FWHM, aperture_override=4.0, exptime=1.0)
    assert off["phot"]["aperture_correction_status"] == "off"

    missing = pc.measure_star(img, (x, y), fwhm=FWHM, aperture_override=4.0, exptime=1.0,
                              aperture_correct=True, frame_cog=(None, None))
    assert missing["phot"]["aperture_correction_status"] == "unavailable"
    assert missing["phot"]["instr_mag"] == pytest.approx(off["phot"]["instr_mag"])

    applied = pc.measure_star(img, (x, y), fwhm=FWHM, aperture_override=4.0, exptime=1.0,
                              aperture_correct=True)
    assert applied["phot"]["aperture_correction_status"] == "applied"
    assert applied["phot"]["instr_mag"] < off["phot"]["instr_mag"]


@pytest.fixture(scope="module")
def app():
    existing = QtWidgets.QApplication.instance()
    yield existing or QtWidgets.QApplication([])


def test_bulk_summary_and_csv_name_the_uncorrected_frames(app, tmp_path, monkeypatch):
    import controllers.multi_photometry_controller as mpc
    from models.multi_photometry_model import MultiPhotometryModel
    from views.multi_photometry_view import MultiPhotometryView

    model = MultiPhotometryModel()
    ctrl = mpc.MultiPhotometryController(model, MultiPhotometryView())
    model.add_files(["/d/a.fits", "/d/b.fits"])
    model.aperture_correct = True

    def res(i, name, status, corr):
        return dict(index=i, success=True, file=name, picked_full=(10.0, 10.0),
                    radii=[1.0], snrs=[1.0], r_best=1.0, aperture_used=1.0,
                    mag=-5.0, mag_err=0.01, flux=100.0, snr=100.0, exptime=1.0,
                    aperture_result={'flux': 100.0, 'instr_mag': -5.0 + corr,
                                     'aperture_correction': corr,
                                     'aperture_correction_status': status})

    shown = []
    for kind in ("warning", "information"):
        monkeypatch.setattr(mpc.QMessageBox, kind,
                            staticmethod(lambda parent, title, text, *a: shown.append(text)))
    ctrl.progress_dlg = QtWidgets.QProgressDialog()
    ctrl.on_bulk_done([res(0, "a.fits", "applied", -0.27), res(1, "b.fits", "unavailable", 0.0)])
    assert "b.fits" in shown[0] and "NOT total magnitudes" in shown[0]
    assert "a.fits" not in shown[0].split("Aperture correction")[1]

    out = tmp_path / "out.csv"
    monkeypatch.setattr(mpc.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "")))
    ctrl.save_csv()
    import csv
    rows = list(csv.DictReader(out.read_text().splitlines()))
    assert [r["ApertureCorrection"] for r in rows] == ["on", "unavailable"]
