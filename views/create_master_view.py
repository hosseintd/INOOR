import os
import sys
from PyQt5 import QtWidgets, QtCore, QtGui

# Adjust path to find root modules
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils.utils_ui import RangeSlider
from .components.image_canvas import MplImageCanvas
from .components.hist_canvas import MplHistCanvas

class CreateMasterView(QtWidgets.QWidget):
    # Signals for Controller
    add_set_clicked = QtCore.pyqtSignal(str, list)
    remove_set_clicked = QtCore.pyqtSignal()
    set_selected = QtCore.pyqtSignal(int)
    
    add_files_clicked = QtCore.pyqtSignal()
    remove_files_clicked = QtCore.pyqtSignal()
    remove_all_files_clicked = QtCore.pyqtSignal()
    file_selected = QtCore.pyqtSignal(int)
    
    param_changed = QtCore.pyqtSignal() 
    create_master_clicked = QtCore.pyqtSignal()
    toggle_bad_frame = QtCore.pyqtSignal(int, bool)
    estimate_sigma_clicked = QtCore.pyqtSignal()
    show_full_header_clicked = QtCore.pyqtSignal()
    
    # Histogram range signals
    hist_update_range_clicked = QtCore.pyqtSignal(float, float)
    hist_reset_clicked = QtCore.pyqtSignal()
    hist_bitdepth_changed = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # --- LEFT PANEL: Frames, Method, Sigma, Gain, Create ---
        left_panel = QtWidgets.QFrame()
        left_panel.setFixedWidth(320)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Frames Box 
        grp_frames = QtWidgets.QGroupBox("Frames")
        fl = QtWidgets.QVBoxLayout()
        self.list_files = QtWidgets.QListWidget()
        self.list_files.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        fl.addWidget(self.list_files, stretch=1)
        
        hbox_file_btns = QtWidgets.QHBoxLayout()
        self.btn_browse = QtWidgets.QPushButton("Load Files")
        self.btn_remove_file = QtWidgets.QPushButton("Remove Selected")
        self.btn_remove_all = QtWidgets.QPushButton("Remove All")
        hbox_file_btns.addWidget(self.btn_browse)
        hbox_file_btns.addWidget(self.btn_remove_file)
        hbox_file_btns.addWidget(self.btn_remove_all)
        fl.addLayout(hbox_file_btns)
        grp_frames.setLayout(fl)
        left_layout.addWidget(grp_frames, stretch=1)

        # Method Group
        grp_method = QtWidgets.QGroupBox("Method")
        ml = QtWidgets.QVBoxLayout()
        self.radio_mean = QtWidgets.QRadioButton("Mean")
        self.radio_median = QtWidgets.QRadioButton("Median")
        self.radio_median.setChecked(True)
        ml.addWidget(self.radio_mean)
        ml.addWidget(self.radio_median)
        grp_method.setLayout(ml)
        left_layout.addWidget(grp_method)

        # Sigma Clipping Group
        grp_sigma = QtWidgets.QGroupBox("Sigma Clip?")
        sl = QtWidgets.QVBoxLayout()
        self.chk_sigma = QtWidgets.QCheckBox("Apply sigma clipping (2D Post-filter)")
        sl.addWidget(self.chk_sigma)
        
        # Auto/Manual Radios
        hl_sig_mode = QtWidgets.QHBoxLayout()
        self.radio_sig_auto = QtWidgets.QRadioButton("Auto")
        self.radio_sig_manual = QtWidgets.QRadioButton("Manual")
        self.radio_sig_auto.setChecked(True)
        hl_sig_mode.addWidget(self.radio_sig_auto)
        hl_sig_mode.addWidget(self.radio_sig_manual)
        sl.addLayout(hl_sig_mode)

        form_sig = QtWidgets.QFormLayout()
        self.spin_sigma_lo = QtWidgets.QDoubleSpinBox(); self.spin_sigma_lo.setRange(0.1, 50.0); self.spin_sigma_lo.setValue(3.0)
        self.spin_sigma_hi = QtWidgets.QDoubleSpinBox(); self.spin_sigma_hi.setRange(0.1, 50.0); self.spin_sigma_hi.setValue(3.0)
        self.spin_sigma_kernel = QtWidgets.QSpinBox(); self.spin_sigma_kernel.setRange(3, 101); self.spin_sigma_kernel.setValue(35); self.spin_sigma_kernel.setSingleStep(2)
        form_sig.addRow("Lower Sigma:", self.spin_sigma_lo)
        form_sig.addRow("Upper Sigma:", self.spin_sigma_hi)
        form_sig.addRow("Kernel Size:", self.spin_sigma_kernel)
        sl.addLayout(form_sig)
        
        self.btn_estimate = QtWidgets.QPushButton("Estimate (from sample)")
        sl.addWidget(self.btn_estimate)
        grp_sigma.setLayout(sl)
        left_layout.addWidget(grp_sigma)

        # Gain Table Group
        grp_gain = QtWidgets.QGroupBox("Gain Table")
        gl = QtWidgets.QVBoxLayout()
        self.chk_gain = QtWidgets.QCheckBox("Create Gain Table?")
        gl.addWidget(self.chk_gain)
        gl.addWidget(QtWidgets.QLabel("This is just for Flat frames."))
        
        hl_deg = QtWidgets.QHBoxLayout()
        hl_deg.addWidget(QtWidgets.QLabel("Degree:"))
        self.spin_degree = QtWidgets.QSpinBox(); self.spin_degree.setRange(1, 6); self.spin_degree.setValue(2)
        hl_deg.addWidget(self.spin_degree)
        gl.addLayout(hl_deg)
        grp_gain.setLayout(gl)
        left_layout.addWidget(grp_gain)

        left_layout.addStretch()
        self.btn_create_master = QtWidgets.QPushButton("Start Creating")
        self.btn_create_master.setMinimumHeight(40)
        self.btn_create_master.setStyleSheet("background: #006d5b; font-weight: bold;")
        left_layout.addWidget(self.btn_create_master)
        
        self.lbl_status = QtWidgets.QLabel("Ready")
        self.lbl_status.setAlignment(QtCore.Qt.AlignCenter)
        left_layout.addWidget(self.lbl_status)
        
        layout.addWidget(left_panel)

        # --- CENTER PANEL: Image Canvas and Basic Nav ---
        center_panel = QtWidgets.QFrame()
        center_layout = QtWidgets.QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas_image = MplImageCanvas(figsize=(6, 6))
        center_layout.addWidget(self.canvas_image, stretch=1)
        
        self.lbl_filename = QtWidgets.QLabel("")
        self.lbl_filename.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_filename.setStyleSheet("font-weight: bold; color: #aaffff;")
        center_layout.addWidget(self.lbl_filename)

        hbox_nav = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("Previous")
        self.chk_mark_bad = QtWidgets.QCheckBox("Bad Frame?")
        self.btn_next = QtWidgets.QPushButton("Next")
        hbox_nav.addWidget(self.btn_prev)
        hbox_nav.addWidget(self.chk_mark_bad)
        hbox_nav.addWidget(self.btn_next)
        center_layout.addLayout(hbox_nav)

        layout.addWidget(center_panel, stretch=1)

        # --- RIGHT PANEL: Header, Stats, Histogram ---
        right_panel = QtWidgets.QFrame()
        right_panel.setFixedWidth(340)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header Information (Moved from Left)
        self.grp_header = QtWidgets.QGroupBox("Header Info (current frame)")
        hl = QtWidgets.QFormLayout()
        self.lbl_head_exptime = QtWidgets.QLabel("N/A")
        self.lbl_head_temp = QtWidgets.QLabel("N/A")
        self.lbl_head_ybin = QtWidgets.QLabel("N/A")
        self.lbl_head_xbin = QtWidgets.QLabel("N/A")
        self.lbl_head_gain = QtWidgets.QLabel("N/A")
        self.lbl_head_date = QtWidgets.QLabel("N/A")
        
        hl.addRow("EXPTIME:", self.lbl_head_exptime)
        hl.addRow("CCD-TEMP:", self.lbl_head_temp)
        hl.addRow("YBINNING:", self.lbl_head_ybin)
        hl.addRow("XBINNING:", self.lbl_head_xbin)
        hl.addRow("GAIN:", self.lbl_head_gain)
        hl.addRow("DATE-OBS:", self.lbl_head_date)
        
        self.btn_full_header = QtWidgets.QPushButton("Show Full Header")
        hl.addRow(self.btn_full_header)
        self.grp_header.setLayout(hl)
        right_layout.addWidget(self.grp_header)

        # Statistics Group
        self.grp_stats = QtWidgets.QGroupBox("Statistics (current frame)")
        sl_form = QtWidgets.QFormLayout()
        self.lbl_stat_mean = QtWidgets.QLabel("N/A")
        self.lbl_stat_median = QtWidgets.QLabel("N/A")
        self.lbl_stat_std = QtWidgets.QLabel("N/A")
        self.lbl_stat_max = QtWidgets.QLabel("N/A")
        self.lbl_stat_min = QtWidgets.QLabel("N/A")
        self.lbl_stat_rms = QtWidgets.QLabel("N/A")
        
        sl_form.addRow("Mean:", self.lbl_stat_mean)
        sl_form.addRow("Median:", self.lbl_stat_median)
        sl_form.addRow("Std:", self.lbl_stat_std)
        sl_form.addRow("Max:", self.lbl_stat_max)
        sl_form.addRow("Min:", self.lbl_stat_min)
        sl_form.addRow("RMS:", self.lbl_stat_rms)
        self.grp_stats.setLayout(sl_form)
        right_layout.addWidget(self.grp_stats)

        # Histogram Group (Reorganized)
        self.grp_hist = QtWidgets.QGroupBox("Histogram Controls")
        hist_vbox = QtWidgets.QVBoxLayout()
        
        # Title/Instructions inside group
        hist_vbox.addWidget(QtWidgets.QLabel("Drag red/green lines to set range"))
        
        self.canvas_hist = MplHistCanvas(figsize=(4, 2))
        hist_vbox.addWidget(self.canvas_hist)
        
        # Controls Form
        form_hist = QtWidgets.QFormLayout()
        self.spin_bitdepth = QtWidgets.QSpinBox(); self.spin_bitdepth.setRange(1, 32); self.spin_bitdepth.setValue(12)
        form_hist.addRow("Bit depth:", self.spin_bitdepth)
        
        hl_manual = QtWidgets.QHBoxLayout()
        self.spin_hist_min = QtWidgets.QDoubleSpinBox(); self.spin_hist_min.setRange(-1e9, 1e9); self.spin_hist_min.setDecimals(4)
        self.spin_hist_max = QtWidgets.QDoubleSpinBox(); self.spin_hist_max.setRange(-1e9, 1e9); self.spin_hist_max.setDecimals(4)
        hl_manual.addWidget(QtWidgets.QLabel("Min:"))
        hl_manual.addWidget(self.spin_hist_min)
        hl_manual.addWidget(QtWidgets.QLabel("Max:"))
        hl_manual.addWidget(self.spin_hist_max)
        form_hist.addRow(hl_manual)
        hist_vbox.addLayout(form_hist)

        # Reset Button (User wants it here)
        self.btn_hist_reset = QtWidgets.QPushButton("Reset (ZScale)")
        self.btn_hist_reset.setMinimumHeight(30)
        hist_vbox.addWidget(self.btn_hist_reset)
        
        self.grp_hist.setLayout(hist_vbox)
        right_layout.addWidget(self.grp_hist)
        
        right_layout.addStretch()

        layout.addWidget(right_panel)

        # Styling
        self.setStyleSheet("""
            QWidget { background: #162a2a; color: #d9f0ec; font-family: Arial; }
            QGroupBox { border: 1px solid #2e6f6f; margin-top: 6px; }
            QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 3px 0 3px; }
            QPushButton { background: #0f7a73; border-radius: 4px; padding: 6px; color: white; }
            QPushButton:pressed { background: #0a524d; }
            QPushButton:disabled { background: #3a3a3a; color: #888; }
            QListWidget { background: #101616; color: #d9f0ec; selection-background-color: #0f7a73; selection-color: white; }
            QListWidget:item:selected { background: #0f7a73; color: white; border: 1px solid #d9f0ec; }
            QLabel { color: #d9f0ec; }
            QSpinBox, QDoubleSpinBox { background: #0e1a1a; color: #d9f0ec; }
            QRadioButton, QCheckBox { color: #d9f0ec; }
        """)

        # Connections
        self.btn_browse.clicked.connect(self.on_browse)
        self.btn_remove_file.clicked.connect(self.remove_files_clicked.emit)
        self.btn_remove_all.clicked.connect(self.remove_all_files_clicked.emit)
        self.list_files.currentRowChanged.connect(self.file_selected.emit)
        
        self.btn_prev.clicked.connect(lambda: self._step_list(-1))
        self.btn_next.clicked.connect(lambda: self._step_list(1))
        self.chk_mark_bad.clicked.connect(self.on_bad_toggled)
        
        self.btn_create_master.clicked.connect(self.create_master_clicked.emit)
        self.btn_estimate.clicked.connect(self.estimate_sigma_clicked.emit)
        self.btn_full_header.clicked.connect(self.show_full_header_clicked.emit)
        
        # Param Changes
        self.radio_mean.toggled.connect(self.param_changed.emit)
        self.radio_median.toggled.connect(self.param_changed.emit)
        self.chk_sigma.stateChanged.connect(self.param_changed.emit)
        self.radio_sig_auto.toggled.connect(self.param_changed.emit)
        self.radio_sig_manual.toggled.connect(self.param_changed.emit)
        
        # UI Behavior: Disable lower sigma in Auto mode
        self.radio_sig_auto.toggled.connect(lambda checked: self.spin_sigma_lo.setEnabled(not checked))
        self.spin_sigma_lo.setEnabled(not self.radio_sig_auto.isChecked())

        self.spin_sigma_lo.valueChanged.connect(self.param_changed.emit)
        self.spin_sigma_hi.valueChanged.connect(self.param_changed.emit)
        self.spin_sigma_kernel.valueChanged.connect(self.param_changed.emit)
        self.chk_gain.stateChanged.connect(self.param_changed.emit)
        self.spin_degree.valueChanged.connect(self.param_changed.emit)
        
        # Histogram Controls
        self.canvas_hist.range_changed.connect(self.on_hist_range_changed)
        self.btn_hist_reset.clicked.connect(self.hist_reset_clicked.emit)
        self.spin_bitdepth.valueChanged.connect(self.on_bitdepth_changed)
        
        # Connect spinboxes to update histogram visual lines too
        self.spin_hist_min.valueChanged.connect(self.on_spin_limits_changed)
        self.spin_hist_max.valueChanged.connect(self.on_spin_limits_changed)

    def _step_list(self, delta):
        row = self.list_files.currentRow()
        count = self.list_files.count()
        if count == 0: return
        new_row = max(0, min(count - 1, row + delta))
        self.list_files.setCurrentRow(new_row)

    def on_browse(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select FITS files", "", "FITS (*.fits *.fit)")
        if paths:
            # For this simple legacy-look view, we default to adding to a "Master" type set
            # The controller will handle the creation of the underlying Model Set
            self.add_set_clicked.emit("Light", paths)

    def on_bad_toggled(self):
        row = self.list_files.currentRow()
        if row >= 0:
            self.toggle_bad_frame.emit(row, self.chk_mark_bad.isChecked())

    def get_selected_file_indices(self):
        return [item.row() for item in self.list_files.selectedIndexes()]

    def on_hist_range_changed(self, mn, mx):
        self.spin_hist_min.blockSignals(True)
        self.spin_hist_max.blockSignals(True)
        self.spin_hist_min.setValue(mn)
        self.spin_hist_max.setValue(mx)
        self.spin_hist_min.blockSignals(False)
        self.spin_hist_max.blockSignals(False)
        self.hist_update_range_clicked.emit(mn, mx)

    def on_spin_limits_changed(self):
        self.canvas_hist.set_limits(self.spin_hist_min.value(), self.spin_hist_max.value())
        self.hist_update_range_clicked.emit(self.spin_hist_min.value(), self.spin_hist_max.value())

    def on_bitdepth_changed(self, val):
        self.hist_bitdepth_changed.emit(val)

    # --- Update API (Called by Controller) ---
    def update_file_list(self, fs):
        self.list_files.blockSignals(True)
        current_row = self.list_files.currentRow()
        self.list_files.clear()
        if fs:
            for i, p in enumerate(fs.files):
                name = os.path.basename(p)
                item = QtWidgets.QListWidgetItem(name)
                if i in fs.bad_indices:
                    item.setForeground(QtGui.QColor('red'))
                    item.setText(name + " [BAD]")
                self.list_files.addItem(item)
        if current_row >= 0 and current_row < self.list_files.count():
            self.list_files.setCurrentRow(current_row)
        self.list_files.blockSignals(False)

    def set_header_info(self, data):
        self.lbl_head_exptime.setText(str(data.get('EXPTIME', 'N/A')))
        self.lbl_head_temp.setText(str(data.get('CCD-TEMP', 'N/A')))
        self.lbl_head_ybin.setText(str(data.get('YBINNING', 'N/A')))
        self.lbl_head_xbin.setText(str(data.get('XBINNING', 'N/A')))
        self.lbl_head_gain.setText(str(data.get('GAIN', 'N/A')))
        self.lbl_head_date.setText(str(data.get('DATE-OBS', 'N/A')))

    def set_stats(self, stats):
        self.lbl_stat_mean.setText(f"{stats['mean']:.4g}")
        self.lbl_stat_median.setText(f"{stats['median']:.4g}")
        self.lbl_stat_std.setText(f"{stats['std']:.4g}")
        self.lbl_stat_max.setText(f"{stats['max']:.4g}")
        self.lbl_stat_min.setText(f"{stats['min']:.4g}")
        self.lbl_stat_rms.setText(f"{stats['rms']:.4g}")

    def clear_stats(self):
        for lbl in [self.lbl_stat_mean, self.lbl_stat_median, self.lbl_stat_std, self.lbl_stat_max, self.lbl_stat_min, self.lbl_stat_rms]:
            lbl.setText("N/A")

    def update_params(self, fs):
        if not fs: return
        self.radio_mean.blockSignals(True)
        self.radio_median.blockSignals(True)
        self.radio_mean.setChecked(fs.method == 'mean')
        self.radio_median.setChecked(fs.method == 'median')
        self.radio_mean.blockSignals(False)
        self.radio_median.blockSignals(False)
        
        self.chk_sigma.setChecked(fs.do_sigma_clip)
        self.spin_sigma_lo.setValue(fs.sigma_lower)
        self.spin_sigma_hi.setValue(fs.sigma_upper)
        self.chk_gain.setChecked(fs.create_gain_table)
        self.spin_degree.setValue(fs.gain_poly_degree)

    def update_hist_spinboxes(self, mn, mx):
        self.spin_hist_min.blockSignals(True)
        self.spin_hist_max.blockSignals(True)
        self.spin_hist_min.setValue(mn)
        self.spin_hist_max.setValue(mx)
        self.spin_hist_min.blockSignals(False)
        self.spin_hist_max.blockSignals(False)
        # Also update visual lines in canvas
        self.canvas_hist.set_limits(mn, mx)

    def set_bad_checkbox(self, is_bad):
        self.chk_mark_bad.blockSignals(True)
        self.chk_mark_bad.setChecked(is_bad)
        self.chk_mark_bad.blockSignals(False)
