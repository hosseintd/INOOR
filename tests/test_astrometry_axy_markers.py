"""
Red (detected) markers of the Astrometry tab. astrometry.net's axy X, Y are 1-based pixels of the
uploaded, possibly downsampled image; the viewer works in 0-based full-resolution pixels. The
markers used to sit at f*X, (f+1)/2 px off the star (live: 1, 1.5 and 2.5 px for f = 1, 2, 4), and
the RA/Dec shown for a marker and the CSV export carried the same offset.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from PyQt5 import QtWidgets

from core.astrometry.astrometry_solver import _rebin_block_average


@pytest.fixture(scope="module")
def app():
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _viewer(tmp_path, image, f, x_axy, y_axy):
    from views.components.astrometry.image_viewer import AstrometryFitsViewer

    frame = tmp_path / "frame.fits"
    fits.writeto(frame, image.astype(np.float32), overwrite=True)
    axy = tmp_path / "axy.fits"
    Table({"X": np.atleast_1d(x_axy), "Y": np.atleast_1d(y_axy)}).write(axy, overwrite=True)
    viewer = AstrometryFitsViewer()
    viewer.load_fits(str(frame))
    viewer.last_downsample_factor = f
    viewer.load_astrometry_files(None, str(axy))
    return viewer


def _centroid(img):
    yy, xx = np.indices(img.shape)
    return (xx * img).sum() / img.sum(), (yy * img).sum() / img.sum()


@pytest.mark.parametrize("f", [1, 2, 3, 4, 8])
def test_marker_formula(app, tmp_path, f):
    viewer = _viewer(tmp_path, np.zeros((64, 64)), f, 5.0, 7.0)
    assert viewer.catalog_pixcoords_raw["det"] == [(f * 5.0 - (f + 1) / 2.0, f * 7.0 - (f + 1) / 2.0)]


@pytest.mark.parametrize("f", [1, 2, 4])
def test_a_star_measured_on_the_uploaded_image_lands_on_itself(app, tmp_path, f):
    """No formula here: downsample with the solver's own routine, measure as astrometry.net reports
    it (1-based), and the marker must return to the star's true position on the full frame."""
    x0, y0 = 40.3, 23.8
    yy, xx = np.indices((128, 128))
    image = 1000.0 * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 4.0 ** 2))
    down = _rebin_block_average(image, f)
    x_down, y_down = _centroid(down)                 # 0-based in the uploaded image
    viewer = _viewer(tmp_path, image, f, x_down + 1.0, y_down + 1.0)   # the axy file is 1-based
    x, y = viewer.catalog_pixcoords_raw["det"][0]
    assert abs(x - x0) < 1e-3 and abs(y - y0) < 1e-3


def test_export_uses_the_corrected_pixel(app, tmp_path):
    viewer = _viewer(tmp_path, np.zeros((64, 64)), 4, 5.0, 7.0)
    row = viewer.export_annotated_list()[0]
    assert (row["x"], row["y"]) == (17.5, 25.5)
