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
from .theme import mpl_colors, set_variant, theme_manager

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
        set_variant(self.btn_remove_file, "danger")
        set_variant(self.btn_remove_all, "danger")
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

        self.chk_scale_levels = QtWidgets.QCheckBox("Scale frames to a common level")
        self.chk_scale_levels.setToolTip(
            "Divide out each frame's own level before combining.\n"
            "Sky flats fade during the sequence, so without this the median\n"
            "just picks the middle frame and you keep one frame's noise.\n"
            "Leave OFF for bias and dark, whose absolute level is the signal.")
        ml.addWidget(self.chk_scale_levels)
        self.lbl_scale_hint = QtWidgets.QLabel("For flats. Leave off for bias/dark.")
        set_variant(self.lbl_scale_hint, "muted")
        ml.addWidget(self.lbl_scale_hint)
        grp_method.setLayout(ml)
        left_layout.addWidget(grp_method)

        # Sigma Clipping Group
        grp_sigma = QtWidgets.QGroupBox("Sigma Clip?")
        sl = QtWidgets.QVBoxLayout()
        self.chk_sigma = QtWidgets.QCheckBox("Apply sigma clipping (2D Post-filter)")
        self.chk_sigma.setToolTip(
            "Replaces spatial outliers in the FINISHED master with the median\n"
            "of their 35x35 block.\n\n"
            "On a master bias or dark this erases stable hot pixels, which is\n"
            "exactly what the master exists to subtract: on a real 6-frame\n"
            "bias it flattened three 1500 ADU hot pixels to 98 ADU, leaving a\n"
            "+1486 ADU spike in every calibrated frame. It also rewrites about\n"
            "as much plain noise as real defects.\n\n"
            "Leave off unless you specifically want this.")
        sl.addWidget(self.chk_sigma)
        self.lbl_sigma_hint = QtWidgets.QLabel(
            "Replaces outliers with the block median.\nErases hot pixels from bias/dark masters.")
        self.lbl_sigma_hint.setWordWrap(True)
        set_variant(self.lbl_sigma_hint, "warning")
        sl.addWidget(self.lbl_sigma_hint)
        
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

        left_layout.addStretch()
        self.btn_create_master = QtWidgets.QPushButton("Start Creating")
        self.btn_create_master.setMinimumHeight(42)
        set_variant(self.btn_create_master, "primary")
        left_layout.addWidget(self.btn_create_master)

        self.lbl_status = QtWidgets.QLabel("Ready")
        self.lbl_status.setAlignment(QtCore.Qt.AlignCenter)
        set_variant(self.lbl_status, "muted")
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
        set_variant(self.lbl_filename, "filename")
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

        # Widget colours come from the application stylesheet (views/theme.py).
        theme_manager.changed.connect(self._apply_theme)

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
        self.chk_scale_levels.stateChanged.connect(self.param_changed.emit)
        self.chk_sigma.stateChanged.connect(self.param_changed.emit)
        self.radio_sig_auto.toggled.connect(self.param_changed.emit)
        self.radio_sig_manual.toggled.connect(self.param_changed.emit)
        
        # UI Behavior: Disable lower sigma in Auto mode
        self.radio_sig_auto.toggled.connect(lambda checked: self.spin_sigma_lo.setEnabled(not checked))
        self.spin_sigma_lo.setEnabled(not self.radio_sig_auto.isChecked())

        self.spin_sigma_lo.valueChanged.connect(self.param_changed.emit)
        self.spin_sigma_hi.valueChanged.connect(self.param_changed.emit)
        self.spin_sigma_kernel.valueChanged.connect(self.param_changed.emit)
        
        # Histogram Controls
        self.canvas_hist.range_changed.connect(self.on_hist_range_changed)
        self.btn_hist_reset.clicked.connect(self.hist_reset_clicked.emit)
        
        # Connect spinboxes to update histogram visual lines too
        self.spin_hist_min.valueChanged.connect(self.on_spin_limits_changed)
        self.spin_hist_max.valueChanged.connect(self.on_spin_limits_changed)

    def _apply_theme(self, *_):
        """Recolour list rows, which carry an explicit per-item foreground."""
        normal = QtGui.QColor(theme_manager.color("text"))
        bad = QtGui.QColor(mpl_colors()["danger"])
        for i in range(self.list_files.count()):
            item = self.list_files.item(i)
            item.setForeground(bad if item.text().endswith(" [BAD]") else normal)

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
                    item.setForeground(QtGui.QColor(mpl_colors()["danger"]))
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

    def clear_header_info(self):
        for lbl in [self.lbl_head_exptime, self.lbl_head_temp, self.lbl_head_ybin,
                    self.lbl_head_xbin, self.lbl_head_gain, self.lbl_head_date]:
            lbl.setText("N/A")

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
        
        self.chk_scale_levels.blockSignals(True)
        self.chk_scale_levels.setChecked(fs.scale_to_common_level)
        self.chk_scale_levels.blockSignals(False)

        self.chk_sigma.setChecked(fs.do_sigma_clip)
        self.spin_sigma_lo.setValue(fs.sigma_lower)
        self.spin_sigma_hi.setValue(fs.sigma_upper)

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
