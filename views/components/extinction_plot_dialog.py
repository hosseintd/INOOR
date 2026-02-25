import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ExtinctionPlotDialog(QtWidgets.QDialog):
    def __init__(self, parent, airmass, mags, errs, k, intercept, k_err):
        super().__init__(parent)
        self.setWindowTitle("Atmospheric Extinction Fit (k)")
        self.resize(1100, 750)
        
        self.airmass = np.array(airmass)
        self.mags = np.array(mags)
        self.errs = np.array(errs)
        self.k = k
        self.intercept = intercept
        self.k_err = k_err
        
        # UI State
        self.show_legend = True
        self.point_color = "#1f77b4"
        self.line_color = "#e63946"
        
        self._setup_ui()
        self._update_plot()

    def _setup_ui(self):
        self.setStyleSheet("""
            QDialog { background: #162a2a; color: #d9f0ec; }
            QGroupBox { border: 1px solid #2e6f6f; margin-top: 10px; font-weight: bold; }
            QPushButton { background: #0f7a73; color: white; border-radius: 4px; padding: 6px; }
            QLabel { color: #d9f0ec; }
            QCheckBox { color: #d9f0ec; }
        """)
        
        main_layout = QtWidgets.QHBoxLayout(self)
        
        # Left: Plot
        plot_container = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_container)
        self.fig = Figure(figsize=(8, 6), tight_layout=True)
        self.fig.patch.set_facecolor('white')
        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas)
        main_layout.addWidget(plot_container, stretch=1)
        
        # Right: Sidebar
        sidebar = QtWidgets.QWidget()
        sidebar.setFixedWidth(280)
        sb_layout = QtWidgets.QVBoxLayout(sidebar)
        
        # Fit Results Group
        g_results = QtWidgets.QGroupBox("Fit Results")
        rl = QtWidgets.QFormLayout(g_results)
        self.lbl_k = QtWidgets.QLabel(f"{self.k:.5f} \u00b1 {self.k_err:.5f}")
        self.lbl_k.setStyleSheet("color: #00ff00; font-weight: bold;")
        self.lbl_c = QtWidgets.QLabel(f"{self.intercept:.4f}")
        rl.addRow("Extinction (k):", self.lbl_k)
        rl.addRow("Intercept (c):", self.lbl_c)
        sb_layout.addWidget(g_results)
        
        # Plot Options
        g_opts = QtWidgets.QGroupBox("Plot Options")
        ol = QtWidgets.QVBoxLayout(g_opts)
        self.chk_legend = QtWidgets.QCheckBox("Show Fitted Equation")
        self.chk_legend.setChecked(True)
        self.chk_legend.stateChanged.connect(self._update_plot)
        ol.addWidget(self.chk_legend)
        
        self.chk_grid = QtWidgets.QCheckBox("Show Grid")
        self.chk_grid.setChecked(True)
        self.chk_grid.stateChanged.connect(self._update_plot)
        ol.addWidget(self.chk_grid)
        
        self.btn_p_color = QtWidgets.QPushButton("Data Point Color")
        self.btn_p_color.clicked.connect(self._pick_p_color)
        ol.addWidget(self.btn_p_color)
        
        self.btn_l_color = QtWidgets.QPushButton("Fit Line Color")
        self.btn_l_color.clicked.connect(self._pick_l_color)
        ol.addWidget(self.btn_l_color)
        
        sb_layout.addWidget(g_opts)
        
        sb_layout.addStretch()
        
        # Bottom Buttons
        btns = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Save Figure")
        self.btn_save.clicked.connect(self._save_plot)
        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        btns.addWidget(self.btn_save)
        btns.addWidget(self.btn_close)
        sb_layout.addLayout(btns)
        
        main_layout.addWidget(sidebar)

    def _update_plot(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('white')
        
        # Plot measurements
        ax.errorbar(self.airmass, self.mags, yerr=self.errs, fmt='o', 
                    color=self.point_color, label='Measurements', capsize=3)
        
        # Plot fit line
        xs = np.linspace(self.airmass.min() * 0.95, self.airmass.max() * 1.05, 100)
        ys = self.intercept + self.k * xs
        
        equation = f"y = ({self.k:.4f} \u00b1 {self.k_err:.4f})x + {self.intercept:.3f}"
        ax.plot(xs, ys, color=self.line_color, lw=2, label=equation if self.chk_legend.isChecked() else "Linear Fit")
        
        ax.set_xlabel("Airmass (X)", color='black', fontsize=12)
        ax.set_ylabel("Instrumental Magnitude", color='black', fontsize=12)
        ax.set_title("Extinction Coefficient Fit", color='black', fontsize=14)
        ax.invert_yaxis()
        
        if self.chk_grid.isChecked():
            ax.grid(True, ls=':', alpha=0.6)
            
        if self.chk_legend.isChecked():
            ax.legend(loc='best', fontsize=10, frameon=True, framealpha=0.8)
            
        for sp in ax.spines.values(): sp.set_color('black')
        ax.tick_params(colors='black')
        
        self.canvas.draw()

    def _pick_p_color(self):
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.point_color))
        if color.isValid():
            self.point_color = color.name()
            self._update_plot()

    def _pick_l_color(self):
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.line_color))
        if color.isValid():
            self.line_color = color.name()
            self._update_plot()

    def _save_plot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Fit Plot", "extinction_fit.png", "PNG (*.png);;PDF (*.pdf)")
        if path:
            self.fig.savefig(path, dpi=300, bbox_inches='tight')
            QtWidgets.QMessageBox.information(self, "Saved", f"Plot saved to {path}")
