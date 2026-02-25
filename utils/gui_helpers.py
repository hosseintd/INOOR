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
class MplHistCanvas(FigureCanvas):
    def __init__(self, parent=None, figsize=(4,2)):
        fig = Figure(figsize=figsize, tight_layout=False)
        fig.patch.set_facecolor('#162a2a')
        super().__init__(fig)
        self.fig = fig
        self.ax = fig.add_subplot(111)
        self._style()

    def _style(self):
        self.ax.set_facecolor('#162a2a')
        self.ax.tick_params(axis='both', colors='#d9f0ec', labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color('#2e6f6f')
        self.fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.12)
        self.ax.set_yscale('log')

    def show_hist(self, counts, bin_edges):
        self.ax.clear()
        self._style()
        if counts is None or len(counts) == 0:
            self.ax.text(0.5, 0.5, "No histogram", transform=self.ax.transAxes, ha='center', color='#d9f0ec')
            self.draw()
            return
        counts = np.asarray(counts, dtype='float32')
        bins = np.asarray(bin_edges, dtype='float32')
        centers = 0.5 * (bins[:-1] + bins[1:])
        self.ax.plot(centers, np.where(counts > 0, counts, 1.0), linewidth=0.8, color='#a6f0e6')
        self.ax.set_yscale('log')
        self.draw()

# class ZoomImageCanvas(FigureCanvas):
#     """
#     Image canvas with scroll-zoom and click callback signals.
#     Emits:
#       - click_coords(x, y)
#       - view_changed((xmin,xmax),(ymin,ymax))
#     """
#     click_coords = QtCore.pyqtSignal(float, float)
#     view_changed = QtCore.pyqtSignal(object, object)

#     def __init__(self, figsize=(6,6)):
#         fig = Figure(figsize=figsize)
#         super().__init__(fig)
#         self.figure = fig
#         self.ax = fig.add_subplot(111)
#         self.ax.set_axis_off()
#         self.im = None
#         self._img_shape = None
#         self._zoom_base = 1.2
#         self.mpl_connect('scroll_event', self._on_scroll)
#         self.mpl_connect('button_press_event', self._on_click)

#     def show_image(self, img, vmin=None, vmax=None):
#         self.figure.clear()
#         self.ax = self.figure.add_subplot(111)
#         self.ax.set_axis_off()
#         if img is None:
#             self.im = None
#             self._img_shape = None
#             self.draw()
#             return
#         self._img_shape = img.shape
#         if vmin is None or vmax is None:
#             try:
#                 vmin, vmax = ZScaleInterval().get_limits(img)
#             except Exception:
#                 vmin, vmax = np.nanmin(img), np.nanmax(img)
#         self.im = self.ax.imshow(img, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
#         self.figure.tight_layout()
#         self.draw()

#     def set_clim(self, vmin, vmax):
#         if self.im is None:
#             return
#         try:
#             if not np.isfinite(vmin) or not np.isfinite(vmax):
#                 return
#             if vmin >= vmax:
#                 return
#             self.im.set_clim(vmin, vmax)
#             self.figure.canvas.draw_idle()
#         except Exception:
#             pass

#     def _on_click(self, event):
#         if event.inaxes != self.ax:
#             return
#         if event.button == 1 and event.xdata is not None and event.ydata is not None:
#             self.click_coords.emit(float(event.xdata), float(event.ydata))

#     def _on_scroll(self, event):
#         if self.im is None or event.inaxes != self.ax:
#             return
#         base_scale = self._zoom_base
#         scale_factor = 1/base_scale if event.button == 'up' else base_scale
#         cur_xlim = self.ax.get_xlim(); cur_ylim = self.ax.get_ylim()
#         xdata = event.xdata; ydata = event.ydata
#         if xdata is None or ydata is None:
#             return
#         new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
#         new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
#         relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0]) if (cur_xlim[1] - cur_xlim[0]) != 0 else 0.5
#         rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0]) if (cur_ylim[1] - cur_ylim[0]) != 0 else 0.5
#         new_xmin = xdata - (1.0 - relx) * new_width
#         new_xmax = xdata + (relx) * new_width
#         new_ymin = ydata - (1.0 - rely) * new_height
#         new_ymax = ydata + (rely) * new_height
#         ny, nx = self._img_shape
#         new_xmin = max(0, new_xmin); new_ymin = max(0, new_ymin)
#         new_xmax = min(nx, new_xmax); new_ymax = min(ny, new_ymax)
#         self.ax.set_xlim(new_xmin, new_xmax)
#         self.ax.set_ylim(new_ymin, new_ymax)
#         self.draw_idle()
#         try:
#             self.view_changed.emit(tuple(self.ax.get_xlim()), tuple(self.ax.get_ylim()))
#         except Exception:
#             pass

#     def add_patch(self, patch):
#         if self.ax is None:
#             return
#         self.ax.add_patch(patch)
#         self.draw_idle()

#     def clear_patches(self):
#         if self.ax is None:
#             return
#         for p in list(self.ax.patches):
#             p.remove()
#         self.draw_idle()

# ---------- Zoom canvas with view_changed signal ----------
class ZoomImageCanvas(FigureCanvas):
    click_coords = QtCore.pyqtSignal(float, float)
    view_changed = QtCore.pyqtSignal(object, object)   # (xlim tuple, ylim tuple)

    def __init__(self, parent=None, figsize=(6,6)):
        fig = Figure(figsize=figsize, tight_layout=False)
        fig.patch.set_facecolor('#162a2a')
        super().__init__(fig)
        self.fig = fig
        self.ax = fig.add_subplot(111)
        self.ax.set_axis_off()
        self.im = None
        self.cbar = None
        self._img_shape = None
        self._zoom_base = 1.2
        self.mpl_connect('scroll_event', self._on_scroll)
        self.mpl_connect('button_press_event', self._on_click)

    def show_image(self, img, vmin=None, vmax=None):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        if img is None:
            self.im = None
            self._img_shape = None
            self.draw()
            return
        self._img_shape = img.shape
        if vmin is None or vmax is None:
            vmin, vmax = ZScaleInterval().get_limits(img)
        self.im = self.ax.imshow(img, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes("right", size="3.8%", pad=0.12)
        self.cbar = self.fig.colorbar(self.im, cax=cax)
        try:
            self.cbar.ax.yaxis.set_tick_params(color='white')
            for lab in self.cbar.ax.get_yticklabels():
                lab.set_color('white')
        except Exception:
            pass
        self.fig.subplots_adjust(left=0.02, right=0.88, top=0.98, bottom=0.02)
        self.draw()

    def set_clim(self, vmin, vmax):
        if self.im is None:
            return
        try:
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                return
            if vmin >= vmax:
                return
            self.im.set_clim(vmin, vmax)
            if self.cbar is not None:
                try:
                    self.cbar.update_normal(self.im)
                except Exception:
                    try:
                        self.cbar.set_clim(vmin, vmax)
                    except Exception:
                        pass
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            if event.xdata is None or event.ydata is None:
                return
            self.click_coords.emit(float(event.xdata), float(event.ydata))

    def _on_scroll(self, event):
        if self.im is None or event.inaxes != self.ax:
            return
        base_scale = self._zoom_base
        scale_factor = 1/base_scale if event.button == 'up' else base_scale
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        xdata = event.xdata; ydata = event.ydata
        if xdata is None or ydata is None:
            return
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0]) if (cur_xlim[1] - cur_xlim[0])!=0 else 0.5
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0]) if (cur_ylim[1] - cur_ylim[0])!=0 else 0.5
        new_xmin = xdata - (1.0 - relx) * new_width
        new_xmax = xdata + (relx) * new_width
        new_ymin = ydata - (1.0 - rely) * new_height
        new_ymax = ydata + (rely) * new_height
        ny, nx = self._img_shape
        new_xmin = max(0, new_xmin); new_ymin = max(0, new_ymin)
        new_xmax = min(nx, new_xmax); new_ymax = min(ny, new_ymax)
        self.ax.set_xlim(new_xmin, new_xmax)
        self.ax.set_ylim(new_ymin, new_ymax)
        self.draw_idle()
        # emit view changed for persistence
        try:
            self.view_changed.emit(tuple(self.ax.get_xlim()), tuple(self.ax.get_ylim()))
        except Exception:
            pass

    def add_patch(self, patch):
        if self.ax is None:
            return
        self.ax.add_patch(patch)
        self.draw_idle()

    def clear_patches(self):
        if self.ax is None:
            return
        for p in list(self.ax.patches):
            p.remove()
        self.draw_idle()

class ExtinctionDialog(QtWidgets.QDialog):
    """Small dialog to ask RA, Dec, filter and observatory coords for extinction calc."""
    def __init__(self, parent=None, default_lat=33.4, default_lon=51.19):
        super().__init__(parent=parent)
        self.setWindowTitle("Calculate extinction (k)")
        self.resize(420, 220)
        layout = QtWidgets.QFormLayout(self)

        self.edit_ra = QtWidgets.QLineEdit()
        self.edit_dec = QtWidgets.QLineEdit()
        self.edit_filter = QtWidgets.QLineEdit()
        self.spin_lat = QtWidgets.QDoubleSpinBox(); self.spin_lat.setRange(-90,90); self.spin_lat.setDecimals(6); self.spin_lat.setValue(default_lat)
        self.spin_lon = QtWidgets.QDoubleSpinBox(); self.spin_lon.setRange(-180,180); self.spin_lon.setDecimals(6); self.spin_lon.setValue(default_lon)

        layout.addRow("RA (e.g. 22:15:00):", self.edit_ra)
        layout.addRow("Dec (e.g. 51:35:00):", self.edit_dec)
        layout.addRow("Filter (name):", self.edit_filter)
        layout.addRow("Observatory lat (deg):", self.spin_lat)
        layout.addRow("Observatory lon (deg):", self.spin_lon)

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
                    lon=float(self.spin_lon.value()))


def plot_profile_and_snr(fig_profile, fig_snr, rp_radii, rp_profile, snr_radii, snrs, r_best):
    """Modern plot style for radial profile and SNR curves."""
    try:
        fig_profile.clear()
        axp = fig_profile.add_subplot(111)
        axp.set_facecolor('#162a2a')
        axp.tick_params(colors='#d9f0ec', labelsize=8)
        for sp in axp.spines.values(): sp.set_color('#2e6f6f')
        
        if rp_radii is not None and rp_profile is not None and len(rp_radii) > 0:
            axp.plot(rp_radii, rp_profile, color='#a6f0e6', lw=1.5)
            axp.set_xlabel("Radius (px)", color='#d9f0ec', fontsize=9)
            axp.set_ylabel("Intensity", color='#d9f0ec', fontsize=9)
            axp.grid(True, ls=':', color='#2e6f6f', alpha=0.5)
        else:
            axp.text(0.5, 0.5, "No profile", transform=axp.transAxes, ha='center', color='#d9f0ec')
        fig_profile.tight_layout()
    except: pass

    try:
        fig_snr.clear()
        axs = fig_snr.add_subplot(111)
        axs.set_facecolor('#162a2a')
        axs.tick_params(colors='#d9f0ec', labelsize=8)
        for sp in axs.spines.values(): sp.set_color('#2e6f6f')
        
        if snr_radii is not None and snrs is not None and len(snr_radii) > 0:
            axs.plot(snr_radii, snrs, color='#ffcc00', lw=1.5)
            if r_best is not None:
                axs.axvline(r_best, color='#ff5555', linestyle='--', lw=1, label=f'Peak: {r_best:.1f}')
            axs.set_xlabel("Radius (px)", color='#d9f0ec', fontsize=9)
            axs.set_ylabel("SNR", color='#d9f0ec', fontsize=9)
            axs.grid(True, ls=':', color='#2e6f6f', alpha=0.5)
        else:
            axs.text(0.5, 0.5, "No SNR data", transform=axs.transAxes, ha='center', color='#d9f0ec')
        fig_snr.tight_layout()
    except: pass
class CollapsibleBox(QtWidgets.QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)

        self.toggle_button = QtWidgets.QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setStyleSheet("QToolButton { border: none; font-weight: bold; color: #3df2e5; background: #1a3a3a; text-align: left; padding: 5px; }")
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
