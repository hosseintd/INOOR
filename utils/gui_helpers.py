# gui_helpers.py
# Helper widgets and plotting utilities for the photometry GUI.
# Put this file in the same folder as gui_multi_photometry.py

import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Circle
from astropy.visualization import ZScaleInterval
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
import astropy.units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import os

from views.theme import mpl_colors, set_variant, style_axes, style_figure

def get_output_dir():
    """
    Returns the path to the 'Output' directory in that is safe to write to.
    Typically: ~/Documents/INOOR/Output
    """
    # Use user's documents folder to avoid permission issues in Program Files
    docs = os.path.expanduser("~/Documents")
    out_dir = os.path.join(docs, "INOOR", "Output")
    
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating output dir: {e}")
            # Fallback to local if documents fails (unlikely but safe)
            out_dir = os.path.join(os.getcwd(), "Output")
            os.makedirs(out_dir, exist_ok=True)
            
    return out_dir


# class MplHistCanvas(FigureCanvas):
#     """Small histogram canvas used on the right-hand diagnostics panel."""

#     def __init__(self, figsize=(4,2)):
#         # create Figure and keep reference as self.fig (avoids attribute errors)
#         fig = Figure(figsize=figsize)
#         super().__init__(fig)
#         # also keep the same reference name used in older code
#         self.fig = fig
#         self.ax = fig.add_subplot(111)
#         self._style()

#     def _style(self):
#         # style kept minimal so it doesn't depend on external theme code
#         try:
#             self.ax.set_facecolor('#162a2a')
#             self.ax.tick_params(colors='#d9f0ec')
#             for sp in self.ax.spines.values():
#                 sp.set_color('#2e6f6f')
#             # tightly layout figure to avoid clipped labels
#             self.fig.tight_layout()
#         except Exception:
#             pass

#     def show_hist(self, counts, bin_edges):
#         self.ax.clear()
#         self._style()
#         if counts is None or len(counts) == 0:
#             self.ax.text(0.5, 0.5, "No histogram", transform=self.ax.transAxes, ha='center', color='#d9f0ec')
#             self.draw()
#             return
#         centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
#         self.ax.plot(centers, np.where(counts > 0, counts, 1.0))
#         self.ax.set_yscale('log')
#         self.draw()
# DEAD CODE -- nothing imports MplHistCanvas or ZoomImageCanvas from this
# module. The versions actually used are views/components/hist_canvas.py and
# views/components/zoom_image_canvas.py, and those are the ones wired to the
# theme. Editing the colours below changes nothing on screen. Kept only to
# avoid an unrelated deletion in a UI change; safe to remove.
# MplHistCanvas and ZoomImageCanvas used to live here as well. They were
# dead duplicates -- the live, theme-aware versions are
# views/components/hist_canvas.py and views/components/zoom_image_canvas.py --
# and these copies hard-coded pre-theme colours ('#162a2a', '#d9f0ec',
# '#2e6f6f') that could never appear on screen. Nothing imported them.

class ExtinctionDialog(QtWidgets.QDialog):
    """Small dialog to ask RA, Dec, filter and observatory coords for extinction calc."""
    def __init__(self, parent=None, default_lat=33.4, default_lon=51.19,
                 default_elev=3600.0):
        super().__init__(parent=parent)
        self.setWindowTitle("Calculate extinction (k)")
        self.resize(420, 250)
        layout = QtWidgets.QFormLayout(self)

        self.edit_ra = QtWidgets.QLineEdit()
        self.edit_dec = QtWidgets.QLineEdit()
        self.edit_filter = QtWidgets.QLineEdit()
        self.spin_lat = QtWidgets.QDoubleSpinBox(); self.spin_lat.setRange(-90,90); self.spin_lat.setDecimals(6); self.spin_lat.setValue(default_lat)
        self.spin_lon = QtWidgets.QDoubleSpinBox(); self.spin_lon.setRange(-180,180); self.spin_lon.setDecimals(6); self.spin_lon.setValue(default_lon)
        # INO340 sits at ~3600 m. This was never asked for and defaulted to 0,
        # so the site was modelled at sea level.
        self.spin_elev = QtWidgets.QDoubleSpinBox(); self.spin_elev.setRange(-500, 9000); self.spin_elev.setDecimals(0); self.spin_elev.setValue(default_elev)

        layout.addRow("RA (e.g. 22:15:00):", self.edit_ra)
        layout.addRow("Dec (e.g. 51:35:00):", self.edit_dec)
        layout.addRow("Filter (name):", self.edit_filter)
        layout.addRow("Observatory lat (deg):", self.spin_lat)
        layout.addRow("Observatory lon (deg):", self.spin_lon)
        layout.addRow("Observatory elevation (m):", self.spin_elev)

        btns = QtWidgets.QHBoxLayout()
        self.btn_ok = QtWidgets.QPushButton("OK"); self.btn_cancel = QtWidgets.QPushButton("Cancel")
        btns.addStretch(); btns.addWidget(self.btn_ok); btns.addWidget(self.btn_cancel)
        layout.addRow(btns)

        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

    def values(self):
        return dict(ra=self.edit_ra.text().strip(),
                    dec=self.edit_dec.text().strip(),
                    flt=self.edit_filter.text().strip(),
                    lat=float(self.spin_lat.value()),
                    lon=float(self.spin_lon.value()),
                    elev=float(self.spin_elev.value()))


def plot_profile_and_snr(fig_profile, fig_snr, rp_radii, rp_profile, snr_radii, snrs, r_best,
                         r_used=None, frame_fwhm=None):
    """
    Modern plot style for radial profile and SNR curves.

    r_used is the aperture radius actually applied. It is marked and labelled
    the same way as on the curve of growth; the SNR peak gets its own fainter
    line only when a different rule (or a fixed aperture) moved r_used off it.

    With the frame's FWHM the SNR peak is also labelled k = r_peak / FWHM, the
    factor the frame-FWHM rule would need to sit on it (read on a bright,
    isolated star; see the one-frame k recipe).
    """
    c = mpl_colors()
    try:
        fig_profile.clear()
        style_figure(fig_profile)
        axp = fig_profile.add_subplot(111)
        _style_axes(axp, grid=False)

        if rp_radii is not None and rp_profile is not None and len(rp_radii) > 0:
            # series[] is the data-line palette: colour-blind-safe under the
            # dark and light themes, and bright enough to read on the night
            # theme's near-black axes, which c["accent"] is not.
            axp.plot(rp_radii, rp_profile, color=c["series"][1], lw=1.5)
            axp.set_xlabel("Radius (px)", color=c["text"], fontsize=9)
            axp.set_ylabel("Intensity", color=c["text"], fontsize=9)
            axp.grid(True, ls=':', color=c["grid"], alpha=0.5)
        else:
            axp.text(0.5, 0.5, "No profile", transform=axp.transAxes, ha='center', color=c["muted"])
        fig_profile.tight_layout()
    except: pass

    try:
        fig_snr.clear()
        style_figure(fig_snr)
        axs = fig_snr.add_subplot(111)
        _style_axes(axs, grid=False)

        if snr_radii is not None and snrs is not None and len(snr_radii) > 0:
            axs.plot(snr_radii, snrs, color=c["series"][2], lw=1.5)
            # x in data units, y as a fraction of the axes: the label sits at
            # the bottom whatever the SNR scale is.
            xform = axs.get_xaxis_transform()
            if r_used is None and r_best is not None:
                r_used = r_best
            k_peak = ""
            if r_best is not None and frame_fwhm:
                k_peak = f", k={float(r_best) / float(frame_fwhm):.2f}"
            peak_apart = r_best is not None and abs(float(r_best) - float(r_used)) > 0.05
            if peak_apart:
                axs.axvline(float(r_best), color=c["muted"], linestyle=':', lw=1)
                axs.text(float(r_best), 0.90, f" peak {float(r_best):.1f}{k_peak}",
                         transform=xform, color=c["muted"], fontsize=7)
            if r_used is not None:
                axs.axvline(float(r_used), color=c["danger"], linestyle='--', lw=1)
                axs.text(float(r_used), 0.05,
                         f" r={float(r_used):.1f}{'' if peak_apart else k_peak}",
                         transform=xform, color=c["danger"], fontsize=7)
            axs.set_xlabel("Radius (px)", color=c["text"], fontsize=9)
            axs.set_ylabel("SNR", color=c["text"], fontsize=9)
            axs.grid(True, ls=':', color=c["grid"], alpha=0.5)
        else:
            axs.text(0.5, 0.5, "No SNR data", transform=axs.transAxes, ha='center', color=c["muted"])
        fig_snr.tight_layout()
    except: pass


def _style_axes(ax, grid=True):
    style_axes(ax, grid=grid)


def plot_growth_curve(fig, cog_radii, cog_frac, r_used=None, target=None):
    """
    Curve of growth: the fraction of the star's light inside radius r.

    This is the plot that matters when the aperture is chosen by enclosed
    fraction rather than by peak SNR -- unlike the SNR curve it is monotonic,
    so reading a radius off it does not depend on locating a flat maximum.
    """
    c = mpl_colors()
    try:
        fig.clear()
        style_figure(fig)
        ax = fig.add_subplot(111)
        _style_axes(ax)
        if cog_radii is not None and cog_frac is not None and len(cog_radii) > 0:
            ax.plot(cog_radii, np.asarray(cog_frac) * 100.0, color=c["series"][1], lw=1.6)
            if target is not None:
                ax.axhline(float(target) * 100.0, color=c["warning"], ls=':', lw=1)
                ax.text(cog_radii[0], float(target) * 100.0 + 1.5,
                        f"target {float(target)*100:.0f}%", color=c["warning"], fontsize=7)
            if r_used is not None:
                ax.axvline(float(r_used), color=c["danger"], ls='--', lw=1)
                ax.text(float(r_used), 8, f" r={float(r_used):.1f}", color=c["danger"], fontsize=7)
            ax.set_xlabel("Radius (px)", color=c["text"], fontsize=9)
            ax.set_ylabel("Enclosed flux (%)", color=c["text"], fontsize=9)
            ax.set_ylim(0, 105)
        else:
            ax.text(0.5, 0.5, "No growth curve", transform=ax.transAxes,
                    ha='center', color=c["muted"])
        fig.tight_layout()
    except Exception:
        pass


class CollapsibleBox(QtWidgets.QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)

        self.toggle_button = QtWidgets.QToolButton(text=title, checkable=True, checked=False)
        set_variant(self.toggle_button, "section")
        self.toggle_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(QtCore.Qt.RightArrow)
        self.toggle_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.content_area = QtWidgets.QWidget()
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)

        self.toggle_button.clicked.connect(self.on_clicked)

        self.toggle_animation = QtCore.QParallelAnimationGroup(self)
        self.animation = QtCore.QPropertyAnimation(self.content_area, b"maximumHeight")
        self.toggle_animation.addAnimation(self.animation)

    def on_clicked(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        self.animation.setStartValue(0 if checked else self.content_area.layout().sizeHint().height())
        self.animation.setEndValue(self.content_area.layout().sizeHint().height() if checked else 0)
        self.animation.setDuration(300)
        self.toggle_animation.start()

    def set_content_layout(self, layout):
        self.content_area.setLayout(layout)
        # Update animation end value once layout is set
        self.animation.setEndValue(layout.sizeHint().height())
