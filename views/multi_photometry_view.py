import os
from PyQt5 import QtWidgets, QtCore, QtGui
from .components.zoom_image_canvas import ZoomImageCanvas
from .components.hist_canvas import MplHistCanvas
from utils.gui_helpers import CollapsibleBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MultiPhotometryView(QtWidgets.QWidget):
    # Signals for Controller
    add_files_clicked = QtCore.pyqtSignal()
    remove_files_clicked = QtCore.pyqtSignal()
    remove_all_clicked = QtCore.pyqtSignal()
    sort_clicked = QtCore.pyqtSignal()
    
    file_selected = QtCore.pyqtSignal(int)
    prev_clicked = QtCore.pyqtSignal()
    next_clicked = QtCore.pyqtSignal()
    bad_frame_toggled = QtCore.pyqtSignal(bool)
    
    canvas_clicked = QtCore.pyqtSignal(float, float)
    view_changed = QtCore.pyqtSignal(object, object)
    
    bulk_photometry_clicked = QtCore.pyqtSignal()
    calibrate_zeropoints_clicked = QtCore.pyqtSignal()
    calc_extinction_clicked = QtCore.pyqtSignal()
    review_table_clicked = QtCore.pyqtSignal()
    show_lightcurve_clicked = QtCore.pyqtSignal()
    save_csv_clicked = QtCore.pyqtSignal()
    save_snr_clicked = QtCore.pyqtSignal()
    save_profile_clicked = QtCore.pyqtSignal()
    display_settings_clicked = QtCore.pyqtSignal()

    param_changed = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        # Using a QSplitter for resizable panels
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.addWidget(self.main_splitter)

        # --- LEFT COLUMN: Files & Params ---
        left_panel = QtWidgets.QFrame()
        left_panel.setMinimumWidth(250)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Wrap left panel contents in a scroll area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(5, 5, 5, 5)
        scroll.setWidget(scroll_content)
        left_layout.addWidget(scroll)
        
        # Target for all left column widgets
        target_layout = scroll_layout

        # Files Group
        grp_files = QtWidgets.QGroupBox("Loaded Frames")
        fl = QtWidgets.QVBoxLayout()
        self.list_files = QtWidgets.QListWidget()
        # Enable Drag and Drop for reordering
        self.list_files.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.list_files.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        fl.addWidget(self.list_files)
        
        btns_f1 = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add")
        self.btn_remove = QtWidgets.QPushButton("Remove")
        self.btn_remove_all = QtWidgets.QPushButton("Clear All")
        btns_f1.addWidget(self.btn_add); btns_f1.addWidget(self.btn_remove); btns_f1.addWidget(self.btn_remove_all)
        fl.addLayout(btns_f1)
        
        btns_f2 = QtWidgets.QHBoxLayout()
        self.btn_sort = QtWidgets.QPushButton("Sort by Date/Time")
        self.btn_sort.setMinimumHeight(28)
        btns_f2.addWidget(self.btn_sort)
        fl.addLayout(btns_f2)
        grp_files.setLayout(fl)
        target_layout.addWidget(grp_files, stretch=1)

        # Params Group
        grp_params = QtWidgets.QGroupBox("Photometry Parameters")
        params_vbox = QtWidgets.QVBoxLayout()
        
        # --- Basic Params ---
        pl_basic = QtWidgets.QFormLayout()
        pl_basic.setContentsMargins(4, 8, 4, 4)
        
        self.spin_stamp = QtWidgets.QSpinBox(); self.spin_stamp.setRange(5, 500); self.spin_stamp.setValue(50)
        self.spin_tracking = QtWidgets.QSpinBox(); self.spin_tracking.setRange(50, 2000); self.spin_tracking.setValue(100)
        self.spin_exptime = QtWidgets.QDoubleSpinBox(); self.spin_exptime.setRange(0, 1e6); self.spin_exptime.setValue(1.0)
        
        pl_basic.addRow("Detection Stamp (px):", self.spin_stamp)
        pl_basic.addRow("Tracking Radius (px):", self.spin_tracking)
        pl_basic.addRow("Exp Time (s):", self.spin_exptime)
        params_vbox.addLayout(pl_basic)

        # --- Advanced Params (Collapsible) ---
        self.adv_box = CollapsibleBox("Advanced Settings")
        pl_adv = QtWidgets.QFormLayout()
        pl_adv.setContentsMargins(4, 4, 4, 4)
        
        self.spin_aperture = QtWidgets.QSpinBox(); self.spin_aperture.setRange(1, 800); self.spin_aperture.setValue(10)
        self.chk_fixed_aperture = QtWidgets.QCheckBox("Fixed aperture for all")
        self.spin_fwhm = QtWidgets.QDoubleSpinBox(); self.spin_fwhm.setRange(0.1, 100); self.spin_fwhm.setValue(12.0)
        self.spin_thresh = QtWidgets.QDoubleSpinBox(); self.spin_thresh.setRange(0.1, 100); self.spin_thresh.setValue(3.0)
        self.spin_inner = QtWidgets.QDoubleSpinBox(); self.spin_inner.setRange(1.1, 10); self.spin_inner.setValue(2.0)
        self.spin_outer = QtWidgets.QDoubleSpinBox(); self.spin_outer.setRange(1.2, 20); self.spin_outer.setValue(3.0)
        self.spin_zp = QtWidgets.QDoubleSpinBox(); self.spin_zp.setRange(-100, 100); self.spin_zp.setValue(0.0)

        pl_adv.addRow("Aperture (px):", self.spin_aperture)
        pl_adv.addRow("", self.chk_fixed_aperture)
        pl_adv.addRow("Fit FWHM (px):", self.spin_fwhm)
        pl_adv.addRow("Threshold (sigma):", self.spin_thresh)
        pl_adv.addRow("Annulus Inner (x r):", self.spin_inner)
        pl_adv.addRow("Annulus Outer (x r):", self.spin_outer)
        pl_adv.addRow("Zeropoint:", self.spin_zp)
        
        self.adv_box.set_content_layout(pl_adv)
        params_vbox.addWidget(self.adv_box)
        
        grp_params.setLayout(params_vbox)
        target_layout.addWidget(grp_params)

        # Actions Group
        grp_actions = QtWidgets.QGroupBox("Actions")
        al = QtWidgets.QVBoxLayout()
        self.btn_bulk = QtWidgets.QPushButton("BULK PHOTOMETRY")
        self.btn_bulk.setStyleSheet("background-color: #0d5e5e; font-weight: bold;")
        self.btn_calibrate_zp = QtWidgets.QPushButton("Calculate Zeropoints")
        self.btn_extinction = QtWidgets.QPushButton("Calc Extinction (k)")
        self.btn_review_table = QtWidgets.QPushButton("Review Results Table")
        self.btn_lightcurve = QtWidgets.QPushButton("Show Light Curve")
        self.btn_save_csv = QtWidgets.QPushButton("Save CSV")
        
        al.addWidget(self.btn_bulk)
        al.addWidget(self.btn_calibrate_zp)
        al.addWidget(self.btn_extinction)
        al.addWidget(self.btn_review_table)
        al.addWidget(self.btn_lightcurve)
        al.addWidget(self.btn_save_csv)
        grp_actions.setLayout(al)
        target_layout.addWidget(grp_actions)
        

        # --- CENTER COLUMN: Image ---
        self.center_panel = QtWidgets.QFrame()
        self.center_layout = QtWidgets.QVBoxLayout(self.center_panel)
        self.center_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_fname = QtWidgets.QLabel("No file loaded")
        self.lbl_fname.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_fname.setStyleSheet("font-weight: bold; font-size: 14px; color: #aaffff;")
        self.center_layout.addWidget(self.lbl_fname)
        
        self.canvas = ZoomImageCanvas(self)
        self.center_layout.addWidget(self.canvas, stretch=1)
        
        nav_layout = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("Previous")
        self.btn_display_settings = QtWidgets.QPushButton("Disp. Settings")
        self.btn_next = QtWidgets.QPushButton("Next")
        self.chk_bad = QtWidgets.QCheckBox("Bad Frame")
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_display_settings)
        nav_layout.addWidget(self.chk_bad)
        nav_layout.addWidget(self.btn_next)
        self.center_layout.addLayout(nav_layout)
        
        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(self.center_panel)

        # --- RIGHT COLUMN: Diagnostics ---
        right_panel = QtWidgets.QFrame()
        right_panel.setFixedWidth(350)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right_layout.addWidget(self.right_splitter)
        
        # Histogram Group (Reorganized)
        self.grp_hist = QtWidgets.QGroupBox("Histogram Controls")
        hist_vbox = QtWidgets.QVBoxLayout()
        
        self.canvas_hist = MplHistCanvas(self, figsize=(4, 2))
        hist_vbox.addWidget(self.canvas_hist)
        
        self.btn_reset_zscale = QtWidgets.QPushButton("Reset (ZScale)")
        self.btn_reset_zscale.setMinimumHeight(30)
        hist_vbox.addWidget(self.btn_reset_zscale)
        
        self.grp_hist.setLayout(hist_vbox)
        self.right_splitter.addWidget(self.grp_hist)
        
        # Radial Profile Group
        profile_widget = QtWidgets.QWidget()
        profile_vbox = QtWidgets.QVBoxLayout(profile_widget)
        profile_vbox.setContentsMargins(0, 0, 0, 0)
        profile_vbox.addWidget(QtWidgets.QLabel("Radial Profile (PSF)"))
        self.fig_profile = Figure(figsize=(4, 2.5), tight_layout=True)
        self.fig_profile.patch.set_facecolor('#162a2a')
        self.canvas_profile = FigureCanvas(self.fig_profile)
        profile_vbox.addWidget(self.canvas_profile)
        self.right_splitter.addWidget(profile_widget)
        
        # SNR Group
        snr_widget = QtWidgets.QWidget()
        snr_vbox = QtWidgets.QVBoxLayout(snr_widget)
        snr_vbox.setContentsMargins(0, 0, 0, 0)
        snr_vbox.addWidget(QtWidgets.QLabel("SNR vs Radius"))
        self.fig_snr = Figure(figsize=(4, 2.5), tight_layout=True)
        self.fig_snr.patch.set_facecolor('#162a2a')
        self.canvas_snr = FigureCanvas(self.fig_snr)
        snr_vbox.addWidget(self.canvas_snr)
        self.right_splitter.addWidget(snr_widget)
        
        self.btn_save_snr = QtWidgets.QPushButton("Save SNR Graph")
        self.btn_save_profile = QtWidgets.QPushButton("Save Radial Profile")
        right_layout.addWidget(self.btn_save_snr)
        right_layout.addWidget(self.btn_save_profile)
        
        # Results Box
        grp_res = QtWidgets.QGroupBox("Current Frame Result")
        rl = QtWidgets.QFormLayout()
        self.lbl_mag = QtWidgets.QLabel("N/A")
        self.lbl_err = QtWidgets.QLabel("N/A")
        self.lbl_flux = QtWidgets.QLabel("N/A")
        self.lbl_snr = QtWidgets.QLabel("N/A")
        rl.addRow("Instr Mag:", self.lbl_mag)
        rl.addRow("Mag Err:", self.lbl_err)
        rl.addRow("Flux:", self.lbl_flux)
        rl.addRow("SNR:", self.lbl_snr)
        
        grp_res.setLayout(rl)
        self.right_splitter.addWidget(grp_res)
        
        right_layout.addStretch()
        self.main_splitter.addWidget(right_panel)
        
        # Initial splitter sizes
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([300, 800, 350])

        # Styling
        self.setStyleSheet("""
            QWidget { background: #162a2a; color: #d9f0ec; font-family: Arial; }
            QGroupBox { border: 1px solid #2e6f6f; margin-top: 10px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }
            QPushButton { background: #0f7a73; border-radius: 4px; padding: 6px; color: white; }
            QPushButton:pressed { background: #0a524d; }
            QListWidget { background: #101616; color: #d9f0ec; border: 1px solid #2e6f6f; }
            QListWidget::item:selected { background: #2e6f6f; color: white; border: 1px solid #0f7a73; }
            QSpinBox, QDoubleSpinBox, QComboBox { background: #0e1a1a; color: #d9f0ec; border: 1px solid #2e6f6f; }
            QLabel { color: #d9f0ec; }
            QSplitter::handle { background: #2e6f6f; }
            QSplitter::handle:horizontal { width: 4px; }
        """)

        # Connect internal signals to public signals
        self.btn_add.clicked.connect(self.add_files_clicked.emit)
        self.btn_remove.clicked.connect(self.remove_files_clicked.emit)
        self.btn_remove_all.clicked.connect(self.remove_all_clicked.emit)
        self.btn_sort.clicked.connect(self.sort_clicked.emit)
        
        self.list_files.currentRowChanged.connect(self.file_selected.emit)
        self.btn_prev.clicked.connect(self.prev_clicked.emit)
        self.btn_next.clicked.connect(self.next_clicked.emit)
        self.chk_bad.toggled.connect(self.bad_frame_toggled.emit)
        
        self.canvas.click_coords.connect(self.canvas_clicked.emit)
        self.canvas.view_changed.connect(self.view_changed.emit)
        self.canvas_hist.range_changed.connect(lambda l, h: self.view_changed.emit(None, None)) # Signal to update clim
        self.btn_reset_zscale.clicked.connect(lambda: self.view_changed.emit("zscale", "zscale"))
        
        self.btn_bulk.clicked.connect(self.bulk_photometry_clicked.emit)
        self.btn_calibrate_zp.clicked.connect(self.calibrate_zeropoints_clicked.emit)
        self.btn_extinction.clicked.connect(self.calc_extinction_clicked.emit)
        self.btn_review_table.clicked.connect(self.review_table_clicked.emit)
        self.btn_lightcurve.clicked.connect(self.show_lightcurve_clicked.emit)
        self.btn_save_csv.clicked.connect(self.save_csv_clicked.emit)
        self.btn_save_snr.clicked.connect(self.save_snr_clicked.emit)
        self.btn_save_profile.clicked.connect(self.save_profile_clicked.emit)
        self.btn_display_settings.clicked.connect(self.display_settings_clicked.emit)

        # Param change connections
        self.spin_aperture.valueChanged.connect(lambda: self.param_changed.emit("aperture"))
        self.chk_fixed_aperture.toggled.connect(lambda: self.param_changed.emit("fixed_aperture"))
        self.spin_stamp.valueChanged.connect(lambda: self.param_changed.emit("stamp_size"))
        self.spin_tracking.valueChanged.connect(lambda: self.param_changed.emit("tracking_radius"))
        self.spin_fwhm.valueChanged.connect(lambda: self.param_changed.emit("fwhm"))
        self.spin_thresh.valueChanged.connect(lambda: self.param_changed.emit("thresh"))
        self.spin_inner.valueChanged.connect(lambda: self.param_changed.emit("inner"))
        self.spin_outer.valueChanged.connect(lambda: self.param_changed.emit("outer"))
        self.spin_zp.valueChanged.connect(lambda: self.param_changed.emit("zeropoint"))
        self.spin_exptime.valueChanged.connect(lambda: self.param_changed.emit("exptime"))

    # --- Update API ---
    def update_file_list(self, files, current_index, bad_indices=None):
        self.list_files.blockSignals(True)
        self.list_files.clear()
        bad_indices = bad_indices or set()
        for i, f in enumerate(files):
            item = QtWidgets.QListWidgetItem(os.path.basename(f))
            item.setData(QtCore.Qt.UserRole, f)
            if i in bad_indices:
                item.setForeground(QtGui.QColor("red"))
            else:
                item.setForeground(QtGui.QColor("#d9f0ec"))
            self.list_files.addItem(item)
        if 0 <= current_index < self.list_files.count():
            self.list_files.setCurrentRow(current_index)
        self.list_files.blockSignals(False)

    def set_results(self, mag, err, flux, snr):
        self.lbl_mag.setText(f"{mag:.4f}" if mag is not None else "N/A")
        self.lbl_err.setText(f"{err:.4f}" if err is not None else "N/A")
        self.lbl_flux.setText(f"{flux:.2f}" if flux is not None else "N/A")
        self.lbl_snr.setText(f"{snr:.2f}" if snr is not None else "N/A")

    def update_params(self, p):
        self.blockSignals(True)
        self.spin_aperture.setValue(int(p.aperture))
        self.chk_fixed_aperture.setChecked(p.fixed_aperture)
        self.spin_stamp.setValue(int(p.stamp_size))
        self.spin_fwhm.setValue(p.fwhm)
        self.spin_thresh.setValue(p.threshold)
        self.spin_inner.setValue(p.inner_coef)
        self.spin_outer.setValue(p.outer_coef)
        self.spin_zp.setValue(p.zeropoint)
        if p.exptime_override is not None:
            self.spin_exptime.setValue(p.exptime_override)
        self.blockSignals(False)

    def set_bad_frame(self, is_bad):
        self.chk_bad.blockSignals(True)
        self.chk_bad.setChecked(is_bad)
        self.chk_bad.blockSignals(False)
        # Visually mark list item? 
        row = self.list_files.currentRow()
        if row >= 0:
            item = self.list_files.item(row)
            if is_bad:
                item.setForeground(QtGui.QColor("red"))
            else:
                item.setForeground(QtGui.QColor("#d9f0ec"))
