"""
The FoV label of the Astrometry tab gives the frame's width x height. It used to give the diagonal
under the name FoV: 4.67 arcmin for the 3.30 x 3.30 arcmin INO340 field.
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


def _viewer(tmp_path, nx, ny, scale_arcsec=None, rotation_deg=0.0):
    from views.components.astrometry.image_viewer import AstrometryFitsViewer

    hdr = fits.Header()
    if scale_arcsec is not None:
        s = scale_arcsec / 3600.0
        a = np.radians(rotation_deg)
        hdr.update(CTYPE1="RA---TAN", CTYPE2="DEC--TAN", CRVAL1=333.87, CRVAL2=51.6,
                   CRPIX1=nx / 2.0, CRPIX2=ny / 2.0, CD1_1=-s * np.cos(a), CD1_2=s * np.sin(a),
                   CD2_1=s * np.sin(a), CD2_2=s * np.cos(a))
    path = tmp_path / "frame.fits"
    fits.writeto(path, np.zeros((ny, nx), np.float32), header=hdr, overwrite=True)
    viewer = AstrometryFitsViewer()
    viewer.load_fits(str(path))
    return viewer


def test_width_and_height_of_a_non_square_frame(app, tmp_path):
    assert _viewer(tmp_path, 400, 200, 1.0).fov_label.text() == "FoV: 6.67 x 3.33 arcmin"


def test_a_rotated_field_has_the_same_size(app, tmp_path):
    assert _viewer(tmp_path, 400, 200, 1.0, 30.0).fov_label.text() == "FoV: 6.67 x 3.33 arcmin"


def test_the_ino340_field(app, tmp_path):
    assert _viewer(tmp_path, 4096, 4096, 0.0484).fov_label.text() == "FoV: 3.30 x 3.30 arcmin"


def test_a_frame_without_wcs_shows_no_number(app, tmp_path):
    assert "arcmin" not in _viewer(tmp_path, 64, 64).fov_label.text()
