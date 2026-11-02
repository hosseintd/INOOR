from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from astropy.visualization import ZScaleInterval

from views.theme import mpl_colors, set_variant, style_figure

class PreviewDialog(QtWidgets.QDialog):
    def __init__(self, image_array, title="Preview", parent=None, zscale=True):
        super().__init__(parent=parent)
        self.setWindowTitle(title)
        self.resize(900, 650)
        layout = QtWidgets.QVBoxLayout(self)

        # Figure setup
        c = mpl_colors()
        fig = Figure(figsize=(7,4), tight_layout=False)
        style_figure(fig)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_facecolor(c["ax_bg"])

        if image_array is not None:
            vmin, vmax = None, None
            if zscale:
                interval = ZScaleInterval()
                try:
                    vmin, vmax = interval.get_limits(image_array)
                except:
                    pass

            im = ax.imshow(image_array, origin='lower', cmap=c["image_cmap"], vmin=vmin, vmax=vmax)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.12)
            cbar = fig.colorbar(im, cax=cax)
            try:
                cbar.ax.yaxis.set_tick_params(color=c["text"])
                for label in cbar.ax.get_yticklabels():
                    label.set_color(c["text"])
                cbar.ax.yaxis.label.set_color(c["text"])
                cbar.outline.set_edgecolor(c["spine"])
            except Exception:
                pass
        
        ax.set_axis_off()
        fig.subplots_adjust(left=0.02, right=0.90, top=0.98, bottom=0.02)
        
        layout.addWidget(canvas, stretch=1)
        
        btns = QtWidgets.QHBoxLayout()
        btns.addStretch()
        self.btn_cancel = QtWidgets.QPushButton("Reject / Close")
        self.btn_confirm = QtWidgets.QPushButton("Confirm / Save")
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_confirm)
        layout.addLayout(btns)
        
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_confirm.clicked.connect(self.accept)
        
        # Styling comes from the application stylesheet (views/theme.py).
        set_variant(self.btn_cancel, "danger")
        set_variant(self.btn_confirm, "primary")
