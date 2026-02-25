import sys
import numpy as np
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import ZScaleInterval

class MplImageCanvas(FigureCanvas):
    """
    Matplotlib canvas for showing FITS images with a colorbar.
    Uses cmap='gray'. Colorbar spacing increased and tick labels set to white.
    Provides fast set_clim() updates via set_clim method.
    """
    def __init__(self, parent=None, figsize=(5,5)):
        fig = Figure(figsize=figsize, tight_layout=False)
        fig.patch.set_facecolor('#162a2a')
        super().__init__(fig)
        self.fig = fig
        self.ax = fig.add_subplot(111)
        self.ax.set_axis_off()
        self._im = None
        self._cbar = None
        self.setParent(parent)

    def _style_colorbar(self, cbar):
        try:
            cbar.ax.yaxis.set_tick_params(color='white')
            for label in cbar.ax.get_yticklabels():
                label.set_color('white')
            cbar.ax.yaxis.label.set_color('white')
            cbar.outline.set_edgecolor('#2e6f6f')
        except Exception:
            pass

    def show_image(self, img, vmin=None, vmax=None):
        """
        Show image (numpy 2D). Stores artist at self._im for quick .set_clim updates.
        """
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.ax.set_facecolor('#162a2a')
        self._im = None
        self._cbar = None
        if img is None:
            self.draw()
            return
        
        if vmin is None or vmax is None:
            try:
                vmin, vmax = ZScaleInterval().get_limits(img)
            except Exception:
                vmin, vmax = np.min(img), np.max(img)

        im = self.ax.imshow(img, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        self._im = im
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes("right", size="4%", pad=0.12)
        cbar = self.fig.colorbar(im, cax=cax)
        self._cbar = cbar
        self._style_colorbar(cbar)
        self.fig.subplots_adjust(left=0.02, right=0.90, top=0.98, bottom=0.02)
        self.draw()

    def set_clim(self, vmin, vmax):
        """
        Fast update of displayed colormap limits. Does nothing if there's no image loaded.
        """
        try:
            if self._im is None:
                return
            # guard: ensure finite
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                return
            self._im.set_clim(vmin, vmax)
            # update colorbar limits if exists
            try:
                if self._cbar is not None:
                    self._cbar.set_clim(vmin, vmax)
            except Exception:
                pass
            self.draw()
        except Exception as e:
            print("set_clim error:", e, file=sys.stderr)
