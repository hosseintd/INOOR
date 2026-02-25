from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from astropy.visualization import ZScaleInterval

class PreviewDialog(QtWidgets.QDialog):
    def __init__(self, image_array, title="Preview", parent=None, zscale=True):
        super().__init__(parent=parent)
        self.setWindowTitle(title)
        self.resize(900, 650)
        layout = QtWidgets.QVBoxLayout(self)
        
        # Figure setup
        fig = Figure(figsize=(7,4), tight_layout=False)
        fig.patch.set_facecolor('#162a2a')
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_facecolor('#162a2a')
        
        if image_array is not None:
            vmin, vmax = None, None
            if zscale:
                interval = ZScaleInterval()
                try:
                    vmin, vmax = interval.get_limits(image_array)
                except:
                    pass
            
            im = ax.imshow(image_array, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.12)
            cbar = fig.colorbar(im, cax=cax)
            try:
                cbar.ax.yaxis.set_tick_params(color='white')
                for label in cbar.ax.get_yticklabels():
                    label.set_color('white')
                cbar.ax.yaxis.label.set_color('white')
                cbar.outline.set_edgecolor('#2e6f6f')
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
        
        # Style
        self.setStyleSheet("""
            QDialog { background: #162a2a; border: 1px solid #2e6f6f; }
            QWidget { background: #162a2a; color: #d9f0ec; font-family: Arial; }
            QPushButton { background: #0f7a73; border-radius: 4px; padding: 10px 20px; color: white; font-weight: bold; }
            QPushButton:pressed { background: #0a524d; }
            QPushButton#btn_cancel { background: #7a1f1f; }
        """)
        self.btn_cancel.setObjectName("btn_cancel")
