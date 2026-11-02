"""
Plot elements must follow a theme switch made AFTER they were drawn.

The histogram curve kept the previous theme's colour (teal in the night
theme) until the next frame redrew it.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from matplotlib.colors import to_hex
from PyQt5 import QtWidgets

from views.theme import mpl_colors, theme_manager
from views.components.hist_canvas import MplHistCanvas


@pytest.fixture(scope="module")
def app():
    existing = QtWidgets.QApplication.instance()
    yield existing or QtWidgets.QApplication([])


@pytest.fixture
def restore_theme():
    before = theme_manager.current()
    yield
    theme_manager.apply(before, save=False)


def test_histogram_curve_follows_theme_switch(app, restore_theme):
    theme_manager.apply("dark", save=False)
    canvas = MplHistCanvas()
    edges = np.linspace(0.0, 100.0, 51)
    canvas.show_hist(np.arange(50) + 1.0, edges)
    assert to_hex(canvas.hist_line.get_color()) == to_hex(mpl_colors("dark")["accent"])

    theme_manager.apply("night", save=False)
    assert to_hex(canvas.hist_line.get_color()) == to_hex(mpl_colors("night")["accent"])


def test_empty_histogram_has_no_curve_to_recolour(app, restore_theme):
    canvas = MplHistCanvas()
    canvas.show_hist(None, None)
    assert canvas.hist_line is None
    theme_manager.apply("night", save=False)     # must not raise


def test_aperture_circles_are_red_in_night_theme():
    """The aperture/annulus circles on the image were hard-coded cyan/yellow."""
    from matplotlib.colors import to_rgb
    for key in ("dark", "light"):
        c = mpl_colors(key)
        assert (c["aperture"], c["annulus"]) == ("cyan", "yellow")   # unchanged look
    night = mpl_colors("night")
    for name in ("aperture", "annulus"):
        r, g, b = to_rgb(night[name])
        assert r > 0.9 and r > g and r > b, (name, night[name])     # red-dominated
    assert night["aperture"] != night["annulus"]


def test_exported_snr_graph_prints_the_peak_label_in_ink(app, restore_theme, tmp_path):
    """The SNR peak / k label is drawn in the theme's muted colour, which is
    made to recede on a dark screen; in the white exported figure it was pale."""
    from matplotlib.figure import Figure
    from utils.gui_helpers import plot_profile_and_snr
    from views.theme import PUBLICATION_FG, save_publication_figure

    theme_manager.apply("dark", save=False)
    fig_p, fig_s = Figure(), Figure()
    r = np.linspace(1, 40, 40)
    plot_profile_and_snr(fig_p, fig_s, r, np.exp(-r / 5), r, r * np.exp(-r / 20), 20.0,
                         r_used=13.0, frame_fwhm=13.0)
    ax = fig_s.get_axes()[0]
    peak = [t for t in ax.texts if "peak" in t.get_text()][0]
    screen = to_hex(peak.get_color())
    assert screen == to_hex(mpl_colors("dark")["muted"])

    seen = {}
    real_save = fig_s.savefig
    def spy(*a, **k):
        seen["peak"] = to_hex(peak.get_color())
        return real_save(*a, **k)
    fig_s.savefig = spy
    save_publication_figure(fig_s, str(tmp_path / "snr.png"), dpi=50)
    assert seen["peak"] == to_hex(PUBLICATION_FG)
    assert to_hex(peak.get_color()) == screen, "the screen copy keeps its theme colour"
