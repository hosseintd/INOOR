"""
A bulk run must measure the star that was clicked, in every frame.

The old tracker searched a +-100 px stamp for the brightest blob and accepted
it up to 40 px away (100 px after widening), so on PSR J2215 High_80..89 a
faint star 35 px from a 5x brighter one was measured as the bright one in
9 of 10 frames, and every star with a 2x brighter neighbour within ~100 px
it followed on the raw frames was lost from the second frame on.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits

from core.multi_photometry_worker import BulkPhotometryWorker, LAST_POSITION

FWHM = 12.0
SIGMA = FWHM / 2.3548


def _add_star(img, x, y, amp):
    ny, nx = img.shape
    ya, yb = max(0, int(y) - 40), min(ny, int(y) + 41)
    xa, xb = max(0, int(x) - 40), min(nx, int(x) + 41)
    if ya >= yb or xa >= xb:
        return
    yy, xx = np.mgrid[ya:yb, xa:xb]
    img[ya:yb, xa:xb] += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * SIGMA ** 2))


def _write(path, stars, shape, seed):
    rng = np.random.default_rng(seed)
    img = 40.0 + rng.normal(0.0, 8.0, shape)
    for x, y, a in stars:
        _add_star(img, x, y, a)
    hdr = fits.Header()
    hdr["EXPTIME"] = 60.0
    hdr["IMGTYPE"] = 1                       # KL4040 High channel
    fits.writeto(path, img.astype(np.float32), hdr)
    return str(path)


def _move(xy, deg, centre, shift):
    a = np.radians(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (np.asarray(xy, float) - centre) @ R.T + centre + np.asarray(shift, float)


def _run(paths, ref_index, click):
    worker = BulkPhotometryWorker(paths, ref_index, click, fwhm=FWHM, threshold_sigma=3.0,
                                  detection_stamp_size=50,
                                  aperture_override=12.0, max_radii_samples=20)
    got = []
    worker.done.connect(got.append)
    worker.run()                              # synchronously, in this thread
    return got[0]


def _field(n, size, seed):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(60, size - 60, (n, 2))
    return np.column_stack((xy, rng.uniform(300, 1500, n)))


def test_faint_star_beside_a_brighter_one_is_kept(tmp_path):
    faint, bright = np.array([285.0, 262.0]), np.array([250.0, 256.0])   # 35.5 px apart
    field = _field(12, 512, 7)
    field = field[np.hypot(field[:, 0] - 270, field[:, 1] - 260) > 90]
    drift = np.array([2.0, 1.5])
    paths = []
    for i in range(4):
        s = i * drift
        stars = [(*(faint + s), 80.0), (*(bright + s), 400.0)] + [(x + s[0], y + s[1], a) for x, y, a in field]
        paths.append(_write(tmp_path / f"f_High_{i + 1}.fits", stars, (512, 512), i))
    results = _run(paths, 0, tuple(faint))
    assert all(r["success"] for r in results), [r.get("msg") for r in results]
    assert results[0]["picked_full"] == tuple(faint)            # the click, exactly
    for r in results:
        truth = faint + r["index"] * drift
        assert np.hypot(*(np.asarray(r["picked_full"]) - truth)) < 1.0, (r["index"], r["picked_full"], r["method"])


def test_large_drift_and_rotation(tmp_path):
    centre = np.array([512.0, 512.0])
    field = _field(30, 1024, 11)
    target, neighbour = np.array([600.0, 480.0]), np.array([645.0, 470.0])  # 2.5x brighter, 46 px
    paths = []
    for i in range(3):
        deg, shift = 0.8 * i, (70.0 * i, -45.0 * i)
        pts = _move(np.vstack([target, neighbour, field[:, :2]]), deg, centre, shift)
        amps = [300.0, 750.0] + list(field[:, 2])
        paths.append(_write(tmp_path / f"r_High_{i + 1}.fits",
                            [(x, y, a) for (x, y), a in zip(pts, amps)], (1024, 1024), 20 + i))
    results = _run(paths, 0, tuple(target))
    for r in results:
        assert r["success"], r.get("msg")
        truth = _move(target[None, :], 0.8 * r["index"], centre, (70.0 * r["index"], -45.0 * r["index"]))[0]
        assert np.hypot(*(np.asarray(r["picked_full"]) - truth)) < 1.0, (r["index"], r["method"])
        if r["index"]:
            assert r["method"].startswith("field similarity")


def test_star_that_leaves_the_frame_fails_with_a_reason(tmp_path):
    field = _field(15, 512, 3)
    target = np.array([480.0, 250.0])
    paths = []
    for i, dx in enumerate((0.0, 60.0)):                     # target ends up at x = 540
        stars = [(target[0] + dx, target[1], 600.0)] + [(x + dx, y, a) for x, y, a in field]
        paths.append(_write(tmp_path / f"e_High_{i + 1}.fits", stars, (512, 512), 30 + i))
    r0, r1 = _run(paths, 0, tuple(target))
    assert r0["success"]
    assert not r1["success"] and "outside this frame" in r1["msg"]


def test_sparse_field_falls_back_to_the_last_position_and_says_so(tmp_path):
    star = (100.0, 100.0, 500.0)
    paths = [_write(tmp_path / f"s_High_{i + 1}.fits", [(star[0] + i, star[1], star[2])], (201, 201), 40 + i)
             for i in range(2)]
    r0, r1 = _run(paths, 0, (100.0, 100.0))
    assert r1["success"] and r1["method"].startswith(LAST_POSITION)
    assert "no field match" in r1["method"]
    assert np.hypot(r1["picked_full"][0] - 101.0, r1["picked_full"][1] - 100.0) < 1.0


def test_bulk_summary_names_the_unmatched_frames(tmp_path, monkeypatch):
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import controllers.multi_photometry_controller as mpc
    from models.multi_photometry_model import MultiPhotometryModel
    from views.multi_photometry_view import MultiPhotometryView

    star = (100.0, 100.0, 500.0)
    paths = [_write(tmp_path / f"u_High_{i + 1}.fits", [(star[0] + i, star[1], star[2])], (201, 201), 50 + i)
             for i in range(2)]
    results = _run(paths, 0, (100.0, 100.0))
    model = MultiPhotometryModel()
    ctrl = mpc.MultiPhotometryController(model, MultiPhotometryView())
    model.add_files(paths)
    shown = []
    monkeypatch.setattr(mpc.QMessageBox, "warning",
                        staticmethod(lambda parent, title, text, *a: shown.append(text)))
    monkeypatch.setattr(mpc.QMessageBox, "information",
                        staticmethod(lambda parent, title, text, *a: shown.append("INFO " + text)))
    ctrl.progress_dlg = QtWidgets.QProgressDialog()
    ctrl.on_bulk_done(results)
    assert shown and not shown[0].startswith("INFO")
    assert "could not be matched on 1 frame" in shown[0]
    assert "u_High_2.fits" in shown[0] and "u_High_1.fits" not in shown[0]


def test_bulk_results_carry_the_frame_fwhm(tmp_path):
    field = _field(25, 512, 9)
    paths = [_write(tmp_path / f"w_High_{i + 1}.fits", [(x + i, y, a) for x, y, a in field], (512, 512), 60 + i)
             for i in range(2)]
    worker = BulkPhotometryWorker(paths, 0, tuple(field[0, :2]), fwhm=FWHM, aperture_policy="frame_fwhm",
                                  max_radii_samples=20)
    got = []
    worker.done.connect(got.append)
    worker.run()
    for r in got[0]:
        assert r["success"], r.get("msg")
        assert r["frame_fwhm"] is not None and abs(r["frame_fwhm"] - FWHM) < 1.5
