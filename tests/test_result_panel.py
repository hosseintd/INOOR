"""
The "Current Frame Result" panel of the Multi Photometry tab.
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


def _tab(tmp_path):
    import controllers.multi_photometry_controller as mpc
    from models.multi_photometry_model import MultiPhotometryModel
    from views.multi_photometry_view import MultiPhotometryView

    p = tmp_path / "r_High_1.fits"
    fits.writeto(p, np.zeros((64, 64), np.float32), fits.Header({"EXPTIME": 60.0}))
    model, view = MultiPhotometryModel(), MultiPhotometryView()
    ctrl = mpc.MultiPhotometryController(model, view)
    model.add_files([str(p)])
    return model, view, ctrl, p


def test_clear_all_empties_the_result_panel_and_zp_box(app, tmp_path):
    """Clear All used to leave the last frame's numbers on show until a new
    file was loaded -- which then got ZP 0 while the box still said 21.3."""
    model, view, ctrl, p = _tab(tmp_path)
    model.zeropoints[p.name] = 21.3
    model.update_frame_data(0, mag=13.9, mag_err=0.01, flux=1000.0, snr=50.0)
    ctrl.show_current_data()
    assert view.lbl_mag.text().startswith("13.9") and view.spin_zp.value() == pytest.approx(21.3)

    model.remove_files([0])
    assert view.lbl_mag.text() == "N/A" and view.lbl_snr.text() == "N/A"
    assert view.spin_zp.value() == pytest.approx(model.zeropoint)


def test_mag_label_says_when_a_zeropoint_is_in_it(app, tmp_path):
    """The value is instrumental + ZP; with a ZP it is not an instrumental mag."""
    model, view, ctrl, p = _tab(tmp_path)
    model.update_frame_data(0, mag=-7.39, mag_err=0.01, flux=1000.0, snr=50.0)
    ctrl.show_current_data()
    assert view.lbl_mag_title.text() == "Instr Mag:"
    model.zeropoints[p.name] = 21.32
    model.update_frame_data(0, mag=13.93)
    ctrl.show_current_data()
    assert view.lbl_mag_title.text() == "Mag (ZP):"
