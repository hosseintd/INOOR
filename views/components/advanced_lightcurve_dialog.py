import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class AdvancedLightCurveDialog(QtWidgets.QDialog):
    def __init__(self, parent, rows):
        super().__init__(parent)
        self.setWindowTitle("Advanced Light Curve")
        self.resize(1300, 850)
        
        # Data preparation
        # Data preparation
        self.rows = rows # List of PhotometryRow
        self.valid_rows = [r for r in rows if r.mag is not None]
        
        self.indices = np.array([r.index for r in self.valid_rows])
        self.mags = np.array([r.mag for r in self.valid_rows])
        self.errs = np.array([r.mag_err if r.mag_err else 0.0 for r in self.valid_rows])
        
        # Time Data
        self.ut_times = [r.date_ut for r in self.valid_rows]
        self.obs_times = [r.date_obs for r in self.valid_rows]
        
        # Settings
        self.x_mode = "Index" # "Index", "UT", "Local"
        self.point_color = "#1f77b4"
        self.error_color = "black"
        self.spline_color = "#e63946"
        self.show_spline = True
        
        self._setup_ui()
        if len(self.mags) > 0:
            self._update_axis_defaults()
            self._update_plot()

    def _setup_ui(self):
        # Keep dialog background dark but plot background will be white
        self.setStyleSheet("""
            QDialog { background: #162a2a; color: #d9f0ec; }
            QGroupBox { border: 1px solid #2e6f6f; margin-top: 10px; font-weight: bold; }
            QPushButton { background: #0f7a73; color: white; border-radius: 4px; padding: 6px; }
            QLabel { color: #d9f0ec; }
            QCheckBox { color: #d9f0ec; }
            QDoubleSpinBox, QSpinBox { background: #0e1a1a; color: #d9f0ec; border: 1px solid #2e6f6f; }
        """)
        layout = QtWidgets.QHBoxLayout(self)
        
        # Left: Plot
        plot_container = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_container)
        self.fig = Figure(figsize=(10, 6), tight_layout=True)
        # USER: keep the white background of plot
        self.fig.patch.set_facecolor('white')
        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas)
        layout.addWidget(plot_container, stretch=1)
        
        # Right: Controls
        sidebar_scroll = QtWidgets.QScrollArea()
        sidebar_scroll.setFixedWidth(320)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        
        sidebar = QtWidgets.QWidget()
        self.sidebar = sidebar
        sb_layout = QtWidgets.QVBoxLayout(sidebar)
        
        # 0. X-Axis Mode
        g_xaxis = QtWidgets.QGroupBox("X-Axis Data")
        xl = QtWidgets.QVBoxLayout(g_xaxis)
        self.radio_idx = QtWidgets.QRadioButton("Frame Index")
        self.radio_ut = QtWidgets.QRadioButton("UT (DATE)")
        self.radio_obs = QtWidgets.QRadioButton("Local (DATE-OBS)")
        self.radio_idx.setChecked(True)
        
        xl.addWidget(self.radio_idx)
        xl.addWidget(self.radio_ut)
        xl.addWidget(self.radio_obs)
        
        self.radio_idx.toggled.connect(self._on_xmode_changed)
        self.radio_ut.toggled.connect(self._on_xmode_changed)
        self.radio_obs.toggled.connect(self._on_xmode_changed)
        
        sb_layout.addWidget(g_xaxis)
        
        # 1. Fit Group
        g_fit = QtWidgets.QGroupBox("Spline Fit")
        fl = QtWidgets.QFormLayout(g_fit)
        self.chk_spline = QtWidgets.QCheckBox("Show Spline Curve")
        self.chk_spline.setChecked(True)
        self.chk_spline.stateChanged.connect(self._update_plot)
        fl.addRow(self.chk_spline)
        
        self.spin_s = QtWidgets.QDoubleSpinBox()
        self.spin_s.setRange(0, 1000); self.spin_s.setValue(1.5); self.spin_s.setSingleStep(0.1)
        self.spin_s.valueChanged.connect(self._update_plot)
        fl.addRow("Smoothing (s):", self.spin_s)
        
        self.spin_slw = QtWidgets.QDoubleSpinBox()
        self.spin_slw.setRange(0.5, 10); self.spin_slw.setValue(2.0); self.spin_slw.setSingleStep(0.5)
        self.spin_slw.valueChanged.connect(self._update_plot)
        fl.addRow("Line Width:", self.spin_slw)
        
        self.btn_s_color = QtWidgets.QPushButton("Spline Color")
        self.btn_s_color.clicked.connect(self._pick_s_color)
        fl.addRow(self.btn_s_color)
        sb_layout.addWidget(g_fit)
        
        # 2. Points & Errors Group
        g_style = QtWidgets.QGroupBox("Data Points & Errors")
        gs = QtWidgets.QFormLayout(g_style)
        
        self.btn_p_color = QtWidgets.QPushButton("Point Color")
        self.btn_p_color.clicked.connect(self._pick_p_color)
        gs.addRow(self.btn_p_color)
        
        self.spin_psize = QtWidgets.QSpinBox()
        self.spin_psize.setRange(2, 20); self.spin_psize.setValue(6)
        self.spin_psize.valueChanged.connect(self._update_plot)
        gs.addRow("Point Size:", self.spin_psize)
        
        self.btn_e_color = QtWidgets.QPushButton("Error Bar Color")
        self.btn_e_color.clicked.connect(self._pick_e_color)
        gs.addRow(self.btn_e_color)
        
        self.spin_cap = QtWidgets.QSpinBox()
        self.spin_cap.setRange(0, 15); self.spin_cap.setValue(3)
        self.spin_cap.valueChanged.connect(self._update_plot)
        gs.addRow("Cap Size:", self.spin_cap)
        sb_layout.addWidget(g_style)
        
        # 3. Axis limits Group (Restored)
        g_axis = QtWidgets.QGroupBox("Axis Limits")
        al = QtWidgets.QFormLayout(g_axis)
        self.chk_x_auto = QtWidgets.QCheckBox("Auto X-axis")
        self.chk_x_auto.setChecked(True)
        self.chk_x_auto.stateChanged.connect(self._on_axis_toggle)
        al.addRow(self.chk_x_auto)
        
        self.spin_xmin = QtWidgets.QDoubleSpinBox()
        self.spin_xmin.setRange(-1e6, 1e6); self.spin_xmin.setEnabled(False)
        self.spin_xmin.valueChanged.connect(self._update_plot)
        al.addRow("X Min:", self.spin_xmin)
        
        self.spin_xmax = QtWidgets.QDoubleSpinBox()
        self.spin_xmax.setRange(-1e6, 1e6); self.spin_xmax.setEnabled(False)
        self.spin_xmax.valueChanged.connect(self._update_plot)
        al.addRow("X Max:", self.spin_xmax)
        
        self.chk_y_auto = QtWidgets.QCheckBox("Auto Y-axis")
        self.chk_y_auto.setChecked(True)
        self.chk_y_auto.stateChanged.connect(self._on_axis_toggle)
        al.addRow(self.chk_y_auto)
        
        self.spin_ymin = QtWidgets.QDoubleSpinBox()
        self.spin_ymin.setRange(-1e6, 1e6); self.spin_ymin.setEnabled(False); self.spin_ymin.setDecimals(3)
        self.spin_ymin.valueChanged.connect(self._update_plot)
        al.addRow("Y Min (Mag):", self.spin_ymin)
        
        self.spin_ymax = QtWidgets.QDoubleSpinBox()
        self.spin_ymax.setRange(-1e6, 1e6); self.spin_ymax.setEnabled(False); self.spin_ymax.setDecimals(3)
        self.spin_ymax.valueChanged.connect(self._update_plot)
        al.addRow("Y Max (Mag):", self.spin_ymax)
        sb_layout.addWidget(g_axis)
        
        # 4. Additional Options (Restored)
        g_opt = QtWidgets.QGroupBox("Additional Options")
        ol = QtWidgets.QVBoxLayout(g_opt)
        self.chk_grid = QtWidgets.QCheckBox("Show Grid")
        self.chk_grid.setChecked(True)
        self.chk_grid.stateChanged.connect(self._update_plot)
        ol.addWidget(self.chk_grid)
        
        self.chk_legend = QtWidgets.QCheckBox("Show Legend")
        self.chk_legend.setChecked(False)
        self.chk_legend.stateChanged.connect(self._update_plot)
        ol.addWidget(self.chk_legend)
        
        self.spin_fw = QtWidgets.QSpinBox()
        self.spin_fw.setRange(5, 30); self.spin_fw.setValue(10)
        self.spin_fw.valueChanged.connect(self._on_figsize_changed)
        ol.addWidget(QtWidgets.QLabel("Figure Width (inches):"))
        ol.addWidget(self.spin_fw)
        
        self.spin_fh = QtWidgets.QSpinBox()
        self.spin_fh.setRange(3, 20); self.spin_fh.setValue(6)
        self.spin_fh.valueChanged.connect(self._on_figsize_changed)
        ol.addWidget(QtWidgets.QLabel("Figure Height (inches):"))
        ol.addWidget(self.spin_fh)
        sb_layout.addWidget(g_opt)
        
        sb_layout.addStretch()
        
        btns = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Save Plot")
        self.btn_save.clicked.connect(self._save_plot)
        btns.addWidget(self.btn_save)
        
        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        btns.addWidget(self.btn_close)
        sb_layout.addLayout(btns)
        
        sidebar_scroll.setWidget(sidebar)
        layout.addWidget(sidebar_scroll)

    def _update_axis_defaults(self):
        x_data = self._get_x_data()[0]
        if len(x_data) > 0:
            xm, xM = x_data.min(), x_data.max()
            xr = xM - xm if xM > xm else 1.0
            self.spin_xmin.setValue(xm - xr*0.05)
            self.spin_xmax.setValue(xM + xr*0.05)
        if len(self.mags) > 0:
            ym, yM = self.mags.min(), self.mags.max()
            yr = yM - ym if yM > ym else 0.1
            self.spin_ymin.setValue(ym - yr*0.1)
            self.spin_ymax.setValue(yM + yr*0.1)

    def _on_xmode_changed(self):
        if self.radio_idx.isChecked(): self.x_mode = "Index"
        elif self.radio_ut.isChecked(): self.x_mode = "UT"
        else: self.x_mode = "Local"
        
        self._update_plot()
        # Reset axes on mode change
        self._update_axis_defaults()

    def _get_x_data(self):
        if self.x_mode == "Index":
            return self.indices, "Frame Index", np.arange(len(self.mags)), None
        
        times = self.ut_times if self.x_mode == "UT" else self.obs_times
        from astropy.time import Time
        import datetime
        
        valid_x = []
        valid_indices = []
        plot_date = None
        
        for i, t_str in enumerate(times):
            if not t_str: continue
            try:
                # astropy.time.Time is robust
                t_obj = Time(str(t_str))
                # Store date from first valid frame for title
                if plot_date is None:
                    plot_date = t_obj.to_datetime().strftime("%Y-%m-%d")
                
                # Matplotlib internal date format
                from matplotlib.dates import date2num
                valid_x.append(date2num(t_obj.to_datetime()))
                valid_indices.append(i)
            except:
                continue
            
        if not valid_x:
            return self.indices, "Index (Time parse failed)", np.arange(len(self.mags)), None
            
        return np.array(valid_x), f"Time [{self.x_mode}]", np.array(valid_indices), plot_date

    def _on_axis_toggle(self):
        self.spin_xmin.setEnabled(not self.chk_x_auto.isChecked())
        self.spin_xmax.setEnabled(not self.chk_x_auto.isChecked())
        self.spin_ymin.setEnabled(not self.chk_y_auto.isChecked())
        self.spin_ymax.setEnabled(not self.chk_y_auto.isChecked())
        self._update_plot()

    def _on_figsize_changed(self):
        self.fig.set_size_inches(self.spin_fw.value(), self.spin_fh.value())
        self.canvas.draw()

    def _update_plot(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('white')
        ax.tick_params(colors='black', labelsize=10)
        for sp in ax.spines.values(): sp.set_color('black')

        x_vals, x_label, subset_idx, plot_date = self._get_x_data()
        y_vals = self.mags[subset_idx]
        e_vals = self.errs[subset_idx]

        ax.errorbar(x_vals, y_vals, yerr=e_vals, fmt='o', 
                   color=self.point_color, ecolor=self.error_color, 
                   capsize=self.spin_cap.value(), markersize=self.spin_psize.value(),
                   label='Data Points')
        
        # Formatting for Time axes
        if self.x_mode != "Index":
            from matplotlib.dates import DateFormatter
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            self.fig.autofmt_xdate(rotation=45)
            if plot_date:
                ax.set_title(f"Light Curve - {plot_date} ({self.x_mode})", color='black')

        if self.chk_spline.isChecked() and len(x_vals) > 3:
            try:
                from scipy.interpolate import UnivariateSpline
                sort_idx = np.argsort(x_vals)
                xs = x_vals[sort_idx]
                ys = y_vals[sort_idx]
                spl = UnivariateSpline(xs, ys, s=self.spin_s.value())
                plot_xs = np.linspace(xs.min(), xs.max(), 500)
                ax.plot(plot_xs, spl(plot_xs), color=self.spline_color, lw=self.spin_slw.value(), label='Spline Fit')
            except: pass
            
        # Limits
        if not self.chk_x_auto.isChecked():
            ax.set_xlim(self.spin_xmin.value(), self.spin_xmax.value())
        if not self.chk_y_auto.isChecked():
            ax.set_ylim(self.spin_ymin.value(), self.spin_ymax.value())
            
        ax.invert_yaxis() 
        
        ax.set_xlabel(x_label, color='black', fontsize=12)
        ax.set_ylabel("Magnitude", color='black', fontsize=12)
        
        if self.chk_grid.isChecked():
            ax.grid(True, ls=':', color='gray', alpha=0.3)
        
        if self.chk_legend.isChecked():
            ax.legend(loc='best', fontsize=9)
            
        self.canvas.draw()

    def _pick_p_color(self):
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.point_color))
        if color.isValid():
            self.point_color = color.name()
            self._update_plot()

    def _pick_e_color(self):
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.error_color))
        if color.isValid():
            self.error_color = color.name()
            self._update_plot()

    def _pick_s_color(self):
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.spline_color))
        if color.isValid():
            self.spline_color = color.name()
            self._update_plot()

    def _save_plot(self):
        title, ok = QtWidgets.QInputDialog.getText(self, "Plot Title", "Enter plot title:", text="Light Curve")
        if not ok: return
        
        dpi, ok = QtWidgets.QInputDialog.getInt(self, "Save DPI", "Enter DPI:", value=300, min=72, max=1200)
        if not ok: return
        
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Plot", "lightcurve.png", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;JPG (*.jpg)")
        if path:
            # Temporarily add title
            ax = self.fig.axes[0]
            old_title = ax.get_title()
            ax.set_title(title, color='black', fontsize=14)
            self.fig.savefig(path, dpi=dpi, bbox_inches='tight')
            ax.set_title(old_title)
            self.canvas.draw()
            QtWidgets.QMessageBox.information(self, "Saved", f"Plot saved to {path}")
