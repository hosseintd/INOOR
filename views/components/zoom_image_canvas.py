import numpy as np
from PyQt5 import QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from astropy.visualization import ZScaleInterval
from mpl_toolkits.axes_grid1 import make_axes_locatable

from views.theme import mpl_colors, style_figure, theme_manager

class ZoomImageCanvas(FigureCanvas):
    """
    Image canvas with scroll-zoom, pan (click-drag potentially), and click signals.
    Colours follow the active theme (views/theme.py).
    Emits:
      - click_coords(x, y)
      - view_changed((xmin, xmax), (ymin, ymax))
    """
    click_coords = QtCore.pyqtSignal(float, float)
    view_changed = QtCore.pyqtSignal(object, object)

    def __init__(self, parent=None, figsize=(6,6)):
        fig = Figure(figsize=figsize, tight_layout=False)
        super().__init__(fig)
        self.fig = fig
        style_figure(fig)
        self.ax = fig.add_subplot(111)
        self.ax.set_axis_off()
        self.im = None
        self.cbar = None
        self._img_shape = None
        self._zoom_base = 1.2

        self.mpl_connect('scroll_event', self._on_scroll)
        self.mpl_connect('button_press_event', self._on_click)
        self.setParent(parent)
        theme_manager.changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, *_):
        c = mpl_colors()
        style_figure(self.fig)
        self.ax.set_facecolor(c["ax_bg"])
        if self.im is not None:
            self.im.set_cmap(c["image_cmap"])
        if self.cbar is not None:
            self._style_colorbar()
        self.draw_idle()

    def _style_colorbar(self):
        try:
            c = mpl_colors()
            self.cbar.ax.yaxis.set_tick_params(color=c["text"])
            for lab in self.cbar.ax.get_yticklabels():
                lab.set_color(c["text"])
            self.cbar.outline.set_edgecolor(c["spine"])
        except Exception:
            pass

    def show_image(self, img, vmin=None, vmax=None, xlim=None, ylim=None, extent=None):
        c = mpl_colors()
        self.fig.clear()
        style_figure(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        if img is None:
            self.im = None
            self._img_shape = None
            self.draw()
            return
            
        # If extent is provided, it defines the full-scale coordinate system [x0, x1, y0, y1]
        # This allows us to pass a downsampled image while maintaining original coordinates.
        if extent is not None:
            self._img_shape = (extent[3], extent[1])
        else:
            self._img_shape = img.shape
            
        if vmin is None or vmax is None:
            try:
                # Optimized: if img is huge, ZScale might still be slow
                vmin, vmax = ZScaleInterval().get_limits(img)
            except Exception:
                vmin, vmax = np.nanmin(img), np.nanmax(img)
                
        self.im = self.ax.imshow(img, origin='lower', cmap=c["image_cmap"], vmin=vmin, vmax=vmax, extent=extent)

        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes("right", size="4%", pad=0.12)
        self.cbar = self.fig.colorbar(self.im, cax=cax)
        self._style_colorbar()


        if xlim: self.ax.set_xlim(xlim)
        if ylim: self.ax.set_ylim(ylim)
        
        self.fig.subplots_adjust(left=0.02, right=0.88, top=0.98, bottom=0.02)
        self.draw()

    def set_clim(self, vmin, vmax):
        if self.im is None: return
        try:
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax: return
            self.im.set_clim(vmin, vmax)
            if self.cbar:
                try: self.cbar.update_normal(self.im)
                except: self.cbar.set_clim(vmin, vmax)
            self.fig.canvas.draw_idle()
        except: pass

    def _on_click(self, event):
        if event.inaxes != self.ax: return
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            self.click_coords.emit(float(event.xdata), float(event.ydata))

    def _on_scroll(self, event):
        if self.im is None or event.inaxes != self.ax: return
        base_scale = self._zoom_base
        scale_factor = 1.0/base_scale if event.button == 'up' else base_scale
        
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None: return
        
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        
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
        self.view_changed.emit((new_xmin, new_xmax), (new_ymin, new_ymax))

    def add_patch(self, patch):
        if self.ax:
            self.ax.add_patch(patch)
            self.draw_idle()

    def clear_patches(self):
        if self.ax:
            for p in list(self.ax.patches): p.remove()
            self.draw_idle()
