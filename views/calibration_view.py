import os
import sys
from PyQt5 import QtWidgets, QtCore, QtGui

# Adjust path to find root modules
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from .components.image_canvas import MplImageCanvas
from .components.hist_canvas import MplHistCanvas
from utils.exptime_utils import format_exptime_from_raw

class CalibrationView(QtWidgets.QWidget):
    # Signals for Controller
    browse_files_clicked = QtCore.pyqtSignal(str) # set_type
    remove_item_clicked = QtCore.pyqtSignal()
    
    file_selection_changed = QtCore.pyqtSignal(str, int) # set_type, file_index
    bad_frame_toggled = QtCore.pyqtSignal(str, int, bool) # set_type, index, is_bad
    
    set_selection_changed = QtCore.pyqtSignal(str)  # set_type
    
    review_set_clicked = QtCore.pyqtSignal()
    start_calibration_clicked = QtCore.pyqtSignal()
    
    param_changed = QtCore.pyqtSignal()
    hist_update_range_clicked = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.current_selection = None  # Track current selection to restore after tree update

    def _setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # --- LEFT PANEL: Controls ---
        left_panel = QtWidgets.QFrame()
        left_panel.setFixedWidth(320)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        grp_add = QtWidgets.QGroupBox("Load Frames")
        al = QtWidgets.QVBoxLayout()
        
        fl_type = QtWidgets.QFormLayout()
        self.combo_type = QtWidgets.QComboBox()
        self.combo_type.addItems(["Flat", "Dark", "Bias", "Light", "GainTable", "HotPixelMap"])
        fl_type.addRow("Type to Load:", self.combo_type)
        al.addLayout(fl_type)
        
        self.btn_browse = QtWidgets.QPushButton("Browse and Load Files")
        self.btn_browse.setMinimumHeight(30)
        al.addWidget(self.btn_browse)
        grp_add.setLayout(al)
        left_layout.addWidget(grp_add)

        # Master Options
        self.grp_options = QtWidgets.QGroupBox("Master Options (Flat/Dark/Bias)")
        ol = QtWidgets.QFormLayout()
        ol.setContentsMargins(8, 18, 8, 8)
        self.combo_method = QtWidgets.QComboBox()
        self.combo_method.addItems(["median", "mean"])
        self.chk_sigma = QtWidgets.QCheckBox("Apply sigma clipping")
        self.spin_sigma_lo = QtWidgets.QDoubleSpinBox(); self.spin_sigma_lo.setRange(0.1, 50.0); self.spin_sigma_lo.setValue(3.0)
        self.spin_sigma_hi = QtWidgets.QDoubleSpinBox(); self.spin_sigma_hi.setRange(0.1, 50.0); self.spin_sigma_hi.setValue(3.0)
        
        self.chk_scale_dark = QtWidgets.QCheckBox("Scale Dark by Exposure Time")
        self.chk_scale_dark.setChecked(True)
        
        ol.addRow("Method:", self.combo_method)
        ol.addRow(self.chk_sigma)
        ol.addRow("Lower sigma:", self.spin_sigma_lo)
        ol.addRow("Upper sigma:", self.spin_sigma_hi)
        ol.addRow(self.chk_scale_dark)
        self.grp_options.setLayout(ol)
        left_layout.addWidget(self.grp_options)

        # Post-Processing Options
        self.grp_post = QtWidgets.QGroupBox("Post-Processing (Output)")
        pl = QtWidgets.QFormLayout()
        pl.setContentsMargins(8, 18, 8, 8)
        
        self.chk_cosmic = QtWidgets.QCheckBox("Remove Cosmic Rays / Hot Pixels")
        self.chk_cosmic.setChecked(False)
        
        self.spin_outlier_sigma = QtWidgets.QDoubleSpinBox(); self.spin_outlier_sigma.setRange(0.1, 100.0); self.spin_outlier_sigma.setValue(5.0)
        self.spin_outlier_radius = QtWidgets.QSpinBox(); self.spin_outlier_radius.setRange(1, 10); self.spin_outlier_radius.setValue(1)
        
        self.combo_binning = QtWidgets.QComboBox()
        self.combo_binning.addItems(["1x1 (None)", "2x2", "4x4", "8x8"])
        
        pl.addRow(self.chk_cosmic)
        pl.addRow("Outlier Threshold (σ):", self.spin_outlier_sigma)
        pl.addRow("Outlier Radius (px):", self.spin_outlier_radius)
        pl.addRow("Output Binning:", self.combo_binning)
        self.grp_post.setLayout(pl)
        left_layout.addWidget(self.grp_post)

        left_layout.addStretch()
        self.btn_review_set = QtWidgets.QPushButton("Review Selected (Large)")
        self.btn_remove_item = QtWidgets.QPushButton("Remove Selected Files/Set")
        left_layout.addWidget(self.btn_review_set)
        left_layout.addWidget(self.btn_remove_item)
        
        left_layout.addStretch()
        self.btn_start = QtWidgets.QPushButton("START CALIBRATION")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setStyleSheet("background-color: #0d5e5e; font-weight: bold; border-radius: 6px;")
        left_layout.addWidget(self.btn_start)
        
        self.lbl_status = QtWidgets.QLabel("Ready")
        self.lbl_status.setAlignment(QtCore.Qt.AlignCenter)
        left_layout.addWidget(self.lbl_status)
        layout.addWidget(left_panel)

        # --- CENTER PANEL: Tree View ---
        center_panel = QtWidgets.QFrame()
        center_layout = QtWidgets.QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)

        center_layout.addWidget(QtWidgets.QLabel("Calibration Data Tree"))
        self.tree_data = QtWidgets.QTreeWidget()
        self.tree_data.setHeaderLabels(["Item", "XBIN", "YBIN", "Size", "Temp (C)", "ExpTime", "Sigma Clip"])
        self.tree_data.setColumnCount(7)
        self.tree_data.setColumnWidth(0, 220)
        self.tree_data.setColumnWidth(1, 40)
        self.tree_data.setColumnWidth(2, 40)
        self.tree_data.setColumnWidth(3, 60)
        self.tree_data.setColumnWidth(4, 60)
        self.tree_data.setColumnWidth(5, 60)
        self.tree_data.setColumnWidth(6, 80)
        center_layout.addWidget(self.tree_data, stretch=1)
        
        layout.addWidget(center_panel, stretch=1)

        # --- RIGHT PANEL: Preview ---
        right_panel = QtWidgets.QFrame()
        right_panel.setFixedWidth(360)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        right_layout.addWidget(QtWidgets.QLabel("Preview (Click file to see)"))
        self.canvas_image = MplImageCanvas(figsize=(4, 4))
        right_layout.addWidget(self.canvas_image, stretch=1)
        
        self.lbl_fname = QtWidgets.QLabel("")
        self.lbl_fname.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_fname.setStyleSheet("font-weight: bold; color: #aaffff;")
        right_layout.addWidget(self.lbl_fname)
        
        # Histogram Group (Reorganized)
        self.grp_hist = QtWidgets.QGroupBox("Histogram Controls")
        hist_vbox = QtWidgets.QVBoxLayout()
        hist_vbox.addWidget(QtWidgets.QLabel("Drag red/green lines to set range"))
        
        self.canvas_hist = MplHistCanvas(figsize=(4, 2))
        hist_vbox.addWidget(self.canvas_hist)
        
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
            QTreeWidget { background: #101616; color: #d9f0ec; border: 1px solid #2e6f6f; }
            QHeaderView::section { background-color: #0e1a1a; color: #d9f0ec; }
            QLabel { color: #d9f0ec; }
            QSpinBox, QDoubleSpinBox, QComboBox { background: #0e1a1a; color: #d9f0ec; border: 1px solid #2e6f6f; }
            QCheckBox, QRadioButton { color: #d9f0ec; }
        """)

        # Connections
        self.btn_browse.clicked.connect(lambda: self.browse_files_clicked.emit(self.combo_type.currentText()))
        self.btn_remove_item.clicked.connect(self.remove_item_clicked.emit)
        self.btn_review_set.clicked.connect(self.review_set_clicked.emit)
        self.btn_start.clicked.connect(self.start_calibration_clicked.emit)
        
        self.tree_data.itemSelectionChanged.connect(self._on_tree_selection_changed)
        self.tree_data.itemChanged.connect(self._on_tree_item_changed)
        
        self.combo_method.currentTextChanged.connect(self.param_changed.emit)
        self.chk_sigma.stateChanged.connect(self.param_changed.emit)
        self.spin_sigma_lo.valueChanged.connect(self.param_changed.emit)
        self.spin_sigma_hi.valueChanged.connect(self.param_changed.emit)

        self.canvas_hist.range_changed.connect(self.hist_update_range_clicked.emit)
        self.btn_hist_reset.clicked.connect(lambda: self.hist_update_range_clicked.emit(-1, -1))

    def _on_tree_selection_changed(self):
        item = self.tree_data.currentItem()
        
        # Track current selection to restore after tree updates
        if item:
            if not item.parent():
                # Parent item (set)
                self.current_selection = (item.text(0), None)  # (set_type, None)
            else:
                # Child item (file)
                parent = item.parent()
                set_type = parent.text(0)
                file_index = parent.indexOfChild(item)
                self.current_selection = (set_type, file_index)
        else:
            self.current_selection = None
        
        # Just emit signals - let Qt handle the native selection highlight
        # Don't use custom background colors as they interfere with Qt's selection mechanism
        
        if item:
            if not item.parent():
                # Parent item (set name like "Flat", "Dark", "Bias")
                set_type = item.text(0)
                self.set_selection_changed.emit(set_type)
            else:
                # Child item (file)
                parent = item.parent()
                set_type = parent.text(0)
                file_index = parent.indexOfChild(item)
                self.file_selection_changed.emit(set_type, file_index)

    def _on_tree_item_changed(self, item, column):
        if column != 0 or not item.parent(): return
        
        # Checkbox toggle
        parent = item.parent()
        set_type = parent.text(0)
        index = parent.indexOfChild(item)
        is_bad = item.checkState(0) == QtCore.Qt.Checked
        self.bad_frame_toggled.emit(set_type, index, is_bad)

    # --- Update API ---
    def update_tree(self, sets_data, metadata_cache=None):
        """
        Expects sets_data as a list of FileSet objects.
        metadata_cache: dict mapping file path to metadata dict or None
        """
        self.tree_data.blockSignals(True)
        
        # Save current selection BEFORE clearing the tree
        saved_set_type = None
        saved_file_index = None
        if self.current_selection:
            saved_set_type, saved_file_index = self.current_selection
        
        self.tree_data.clear()
        
        metadata_cache = metadata_cache or {}
        
        for fs in sets_data:
            parent = QtWidgets.QTreeWidgetItem(self.tree_data)
            parent.setText(0, fs.set_type)
            parent.setText(1, "") # XBIN
            parent.setText(2, "") # YBIN
            parent.setText(3, f"{len(fs.files)}")
            # Show sigma clipping status at set level
            sigma_status = "Yes" if fs.do_sigma_clip else "No"
            parent.setText(6, sigma_status)
            parent.setExpanded(True)
            
            for i, fpath in enumerate(fs.files):
                child = QtWidgets.QTreeWidgetItem(parent)
                child.setText(0, os.path.basename(fpath))
                
                meta = metadata_cache.get(fpath)
                if meta:
                    child.setText(1, str(meta.get('XBINNING', '-')))
                    child.setText(2, str(meta.get('YBINNING', '-')))
                    child.setText(3, str(meta.get('NAXIS1', '-')))
                    child.setText(4, f"{meta.get('CCD-TEMP', '-'):.1f}" if isinstance(meta.get('CCD-TEMP'), (int, float)) else str(meta.get('CCD-TEMP', '-')))
                    
                    # Format exposure time with appropriate units using utility
                    et = meta.get('EXPTIME')
                    if isinstance(et, (int, float)):
                        child.setText(5, format_exptime_from_raw(et))
                    else:
                        child.setText(5, str(et))

                child.setCheckState(0, QtCore.Qt.Checked if i in fs.bad_indices else QtCore.Qt.Unchecked)
                if i in fs.bad_indices:
                    child.setForeground(0, QtGui.QColor("red"))
                else:
                    child.setForeground(0, QtGui.QColor("#d9f0ec"))
        
        # Restore selection AFTER tree is rebuilt
        if saved_set_type:
            for i in range(self.tree_data.topLevelItemCount()):
                parent_item = self.tree_data.topLevelItem(i)
                if parent_item.text(0) == saved_set_type:
                    if saved_file_index is None:
                        # Selection was on the set itself
                        self.tree_data.setCurrentItem(parent_item)
                    elif saved_file_index < parent_item.childCount():
                        # Selection was on a file
                        child_item = parent_item.child(saved_file_index)
                        self.tree_data.setCurrentItem(child_item)
                    break
        
        self.tree_data.blockSignals(False)

    def update_params(self, s):
        if not s:
            self.grp_options.setEnabled(False)
            return
        self.grp_options.setEnabled(True)
        self.combo_method.blockSignals(True)
        idx = self.combo_method.findText(s.method)
        if idx >= 0: self.combo_method.setCurrentIndex(idx)
        self.combo_method.blockSignals(False)
        
        # Block signals to prevent triggering param_changed while updating
        self.chk_sigma.blockSignals(True)
        self.spin_sigma_lo.blockSignals(True)
        self.spin_sigma_hi.blockSignals(True)
        
        # Light frames are allowed to use sigma clipping, but default to disabled
        # User can manually enable it if desired
        self.chk_sigma.setChecked(s.do_sigma_clip)
        self.spin_sigma_lo.setValue(s.sigma_lower)
        self.spin_sigma_hi.setValue(s.sigma_upper)
        
        # Enable/disable sigma spin boxes based on checkbox state
        self.spin_sigma_lo.setEnabled(self.chk_sigma.isChecked())
        self.spin_sigma_hi.setEnabled(self.chk_sigma.isChecked())
        
        # Re-enable signals
        self.chk_sigma.blockSignals(False)
        self.spin_sigma_lo.blockSignals(False)
        self.spin_sigma_hi.blockSignals(False)

    def select_tree_item(self, set_type, index):
        self.tree_data.blockSignals(True)
        # Find parent
        for i in range(self.tree_data.topLevelItemCount()):
            p = self.tree_data.topLevelItem(i)
            if p.text(0) == set_type:
                if 0 <= index < p.childCount():
                    self.tree_data.setCurrentItem(p.child(index))
                break
        self.tree_data.blockSignals(False)
