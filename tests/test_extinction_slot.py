"""
The Calc Extinction slot must finish a good fit and report it.

9148adb rewrote the success message to mention "frames excluded for having
no error bar" but never counted them, so every fit that SUCCEEDED raised
NameError('n_no_err') before the result or the plot was shown. Those frames
were also still given a made-up 0.1 mag error, and the elevation the dialog
asks for never reached the airmass calculation.
"""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits
from PyQt5 import QtWidgets

import controllers.multi_photometry_controller as mpc
from models.multi_photometry_model import MultiPhotometryModel
from views.multi_photometry_view import MultiPhotometryView

# GRB 251013C g-band field from INO340, 2025-10-13 UT: the target sets from
# airmass ~1.66 to ~2.0 over these 40 minutes.
RA, DEC = "23:03:16", "-00:12:21"
TIMES_UT = ["2025-10-13T21:04:15", "2025-10-13T21:20:00",
            "2025-10-13T21:30:00", "2025-10-13T21:41:57"]


@pytest.fixture(scope="module")
def app():
    existing = QtWidgets.QApplication.instance()
    yield existing or QtWidgets.QApplication([])


class _Dialog:
    def __init__(self, *a, **k):
        pass

    def exec_(self):
        return QtWidgets.QDialog.Accepted

    def values(self):
        return dict(ra=RA, dec=DEC, flt="g", lat=33.4, lon=51.19, elev=3600.0)


def test_good_fit_is_reported_and_no_error_frames_are_dropped(app, tmp_path, monkeypatch):
    paths = []
    for i, t in enumerate(TIMES_UT):
        p = tmp_path / f"frame_{i}.fits"
        hdr = fits.Header()
        hdr["DATE"] = t
        hdr["EXPTIME"] = 58.14
        fits.writeto(p, np.full((16, 16), 10.0, dtype=np.float32), hdr)
        paths.append(str(p))

    model = MultiPhotometryModel()
    view = MultiPhotometryView()
    ctrl = mpc.MultiPhotometryController(model, view)
    model.add_files(paths)
    for i in range(len(paths)):
        d = model._frame_data[i]
        d.selected_source = (8.0, 8.0)
        d.mag = 20.0 + 0.1 * i
        d.mag_err = 0.01
    model._frame_data[3].mag_err = None          # the one with no error bar

    told = []
    monkeypatch.setattr(mpc, "ExtinctionDialog", _Dialog)
    monkeypatch.setattr(mpc, "ExtinctionPlotDialog",
                        lambda *a, **k: type("P", (), {"exec_": lambda self: None})())
    monkeypatch.setattr(mpc, "get_output_dir", lambda: str(tmp_path))
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(mpc.QMessageBox, kind,
                            staticmethod(lambda parent, title, text, *a, _k=kind:
                                         told.append((_k, title, text))))

    ctrl.calc_extinction()   # raised NameError('n_no_err') before the fix

    success = [t for t in told if t[1] == "Success"]
    assert success, f"no success message; got {told}"
    assert "3 frames" in success[0][2]
    assert "1 frame(s) excluded for having no error bar" in success[0][2]

    out = list((tmp_path / "k_results").glob("extinction_k_*.json"))
    assert len(out) == 1
    res = json.loads(out[0].read_text(encoding="utf-8"))
    assert res["observatory_elevation_m"] == 3600.0
    excluded = [f for f in res["per_frame"] if f.get("excluded")]
    assert [f["index"] for f in excluded] == [3]
    assert all(f.get("mag_err") == 0.01 for f in res["per_frame"] if not f.get("excluded"))
