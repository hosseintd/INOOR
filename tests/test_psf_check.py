"""
Under the frame-FWHM rule a star much wider than the frame's PSF (blend,
galaxy, broader PSF) loses more light outside r = k * FWHM_frame. It is
flagged when wider than FWHM_frame + 3 x the frame's own star-to-star scatter.
"""
import numpy as np

from core import photometry_core as pc


def _frame(target_fwhm, n=30, size=1024, seed=5):
    rng = np.random.default_rng(seed)
    img = 100.0 + rng.normal(0.0, 5.0, (size, size))
    yy, xx = np.mgrid[0:size, 0:size]
    pts = rng.uniform(80, size - 80, (n, 2))
    pts = pts[np.hypot(pts[:, 0] - 512, pts[:, 1] - 512) > 120]
    def add(x, y, fwhm, amp):
        s = fwhm / 2.3548
        sl = (slice(int(y) - 60, int(y) + 61), slice(int(x) - 60, int(x) + 61))
        img[sl] += amp * np.exp(-((xx[sl] - x) ** 2 + (yy[sl] - y) ** 2) / (2 * s ** 2))
    for x, y in pts:
        add(x, y, 12.0, rng.uniform(800, 2500))
    add(512.0, 512.0, target_fwhm, 1500.0)
    return img


def test_spread_does_not_change_the_fwhm():
    img = _frame(12.0)
    plain = pc.estimate_frame_fwhm(img, fwhm_guess=12.0)
    fwhm, spread = pc.estimate_frame_fwhm(img, fwhm_guess=12.0, return_spread=True)
    assert fwhm == plain and abs(fwhm - 12.0) < 0.6
    assert spread is not None and spread < 0.6


def test_too_few_stars_give_no_spread():
    img = _frame(12.0, n=3)
    fwhm, spread = pc.estimate_frame_fwhm(img, fwhm_guess=12.0, return_spread=True)
    assert spread is None


def _check(img, spread):
    fwhm, s = pc.estimate_frame_fwhm(img, fwhm_guess=12.0, return_spread=True)
    m = pc.measure_star(img, (512.0, 512.0), fwhm=12.0, aperture_policy=pc.POLICY_FRAME_FWHM,
                        frame_fwhm=fwhm, frame_fwhm_spread=(s if spread else None), n_radii=20)
    return m['phot']


def test_broad_star_is_flagged_and_a_normal_one_is_not():
    assert _check(_frame(18.0), True)['psf_check'] == 'broad'
    assert _check(_frame(12.0), True)['psf_check'] == 'ok'
    assert _check(_frame(18.0), False)['psf_check'] == 'unchecked'


def test_bulk_summary_and_csv_name_a_broad_star(tmp_path, monkeypatch):
    import csv, os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from astropy.io import fits
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import controllers.multi_photometry_controller as mpc
    from core.multi_photometry_worker import BulkPhotometryWorker
    from models.multi_photometry_model import MultiPhotometryModel
    from views.multi_photometry_view import MultiPhotometryView

    hdr = fits.Header()
    hdr["EXPTIME"] = 60.0
    hdr["IMGTYPE"] = 1
    p = str(tmp_path / "f_High_1.fits")
    fits.writeto(p, _frame(18.0).astype(np.float32), hdr)
    w = BulkPhotometryWorker([p], 0, (512.0, 512.0), fwhm=12.0, aperture_policy="frame_fwhm",
                             max_radii_samples=20)
    got = []
    w.done.connect(got.append)
    w.run()
    assert got[0][0]['aperture_result']['psf_check'] == 'broad'

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
    assert shown and "wider than the frame's PSF on 1 frame" in shown[0]

    out = tmp_path / "out.csv"
    monkeypatch.setattr(mpc.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "CSV (*.csv)")))
    ctrl.save_csv()
    row = list(csv.DictReader(open(out)))[0]
    assert row["PSFCheck"] == "broad" and float(row["StarFWHM"]) > 15.0
