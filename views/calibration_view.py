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
from .theme import mpl_colors, set_variant, theme_manager
from utils.exptime_utils import format_exptime_from_raw

class CalibrationView(QtWidgets.QWidget):
    # Signals for Controller
    browse_files_clicked = QtCore.pyqtSignal(str) # set_type
    remove_item_clicked = QtCore.pyqtSignal()
    
    file_selection_changed = QtCore.pyqtSignal(str, int) # set_type, file_index
    bad_frame_toggled = QtCore.pyqtSignal(str, int, bool) # set_type, index, is_bad
    
    set_selection_changed = QtCore.pyqtSignal(str)  # set_type
    
    review_set_clicked = QtCore.pyqtSignal()
    screen_flats_clicked = QtCore.pyqtSignal()
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
        self.combo_type.addItems(["Flat", "Dark", "Bias", "Light", "HotPixelMap"])
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
        # Off by default. A master dark is pedestal + dark current, and only
        # the current scales with exposure. On the KL4040 at -10 C the median
        # pixel accrues under 1 ADU of dark current in 30 s, so there is almost
        # nothing for the factor to scale -- it mostly multiplies the dark's
        # own read noise (3.7 -> 9.9 ADU at x3.33) and, with no bias frame to
        # hold the pedestal out, the quadrant seam along with it (measured
        # -0.11% with a bias, -1.14% without). Turn it on deliberately when the
        # dark current really is significant.
        self.chk_scale_dark.setChecked(False)
        self.chk_scale_dark.setToolTip(
            "Off by default.\n\n"
            "A master dark is pedestal + dark current, and only the dark "
            "current scales with exposure time. At -10 C this sensor's median "
            "pixel gains under 1 ADU in 30 s, so scaling buys almost nothing "
            "and mainly amplifies the dark's read noise.\n\n"
            "Without a Bias set it also multiplies the pedestal, which puts the "
            "quadrant seam back across the middle of every frame.\n\n"
            "Turn it on when the darks really do differ in exposure AND the "
            "dark current matters -- and load bias frames when you do.")

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
        # The filter compares each pixel to its local median against a GLOBAL
        # sigma, so a real stellar core is flagged just like a cosmic ray and
        # gets flattened. Measured loss with the defaults: ~0.06 mag at a
        # 200 ADU peak rising to ~0.21 mag at 3000 ADU -- brightness dependent,
        # so differential photometry does not cancel it.
        self.chk_cosmic.setToolTip(
            "WARNING: this filter also clips real stellar cores.\n"
            "Measured flux loss with sigma=5, radius=1: ~0.06 mag at a 200 ADU\n"
            "peak, ~0.18 mag at 1000 ADU, ~0.21 mag at 3000 ADU. The error grows\n"
            "with brightness, so it does NOT cancel in differential photometry.\n"
            "Leave this off for photometric data.")
        set_variant(self.chk_cosmic, "warning")


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

        # These act on the tree selection, so they belong with the panel above
        # them. Previously a stretch sat here as well as below, which split the
        # spare height in two and left a large empty gap mid-column.
        grp_sel = QtWidgets.QGroupBox("Selected Item")
        selL = QtWidgets.QVBoxLayout(grp_sel)
        self.btn_review_set = QtWidgets.QPushButton("Review Selected (Large)")
        self.btn_screen_flats = QtWidgets.QPushButton("Check Flat Frames")
        self.btn_screen_flats.setToolTip(
            "Measure every loaded flat and flag the ones that should not go "
            "into the master: outside the linear range, saturated, full of "
            "stars, or not matching the illumination pattern the other flats "
            "agree on. Nothing changes until you confirm.")
        self.btn_remove_item = QtWidgets.QPushButton("Remove Selected Files/Set")
        set_variant(self.btn_remove_item, "danger")
        selL.addWidget(self.btn_review_set)
        selL.addWidget(self.btn_screen_flats)
        selL.addWidget(self.btn_remove_item)
        left_layout.addWidget(grp_sel)

        left_layout.addStretch()
        self.btn_start = QtWidgets.QPushButton("START CALIBRATION")
        self.btn_start.setMinimumHeight(42)
        set_variant(self.btn_start, "primary")
        left_layout.addWidget(self.btn_start)

        self.lbl_status = QtWidgets.QLabel("Ready")
        self.lbl_status.setAlignment(QtCore.Qt.AlignCenter)
        set_variant(self.lbl_status, "muted")
        left_layout.addWidget(self.lbl_status)
        layout.addWidget(left_panel)

        # --- CENTER PANEL: Tree View ---
        center_panel = QtWidgets.QFrame()
        center_layout = QtWidgets.QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)

        center_layout.addWidget(QtWidgets.QLabel("Calibration Data Tree"))
        self.tree_data = QtWidgets.QTreeWidget()
        self.tree_data.setHeaderLabels(["Use / Item", "XBIN", "YBIN", "Size", "Temp (C)", "ExpTime", "Sigma Clip"])
        self.tree_data.headerItem().setToolTip(
            0, "Ticked = the frame takes part in calibration (the default).\n"
               "Untick a frame to exclude it; excluded rows turn red.")
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
        set_variant(self.lbl_fname, "filename")
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

        # Widget colours come from the application stylesheet (views/theme.py).
        theme_manager.changed.connect(self._apply_theme)

        # Connections
        self.btn_browse.clicked.connect(lambda: self.browse_files_clicked.emit(self.combo_type.currentText()))
        self.btn_remove_item.clicked.connect(self.remove_item_clicked.emit)
        self.btn_review_set.clicked.connect(self.review_set_clicked.emit)
        self.btn_screen_flats.clicked.connect(self.screen_flats_clicked.emit)
        self.btn_start.clicked.connect(self.start_calibration_clicked.emit)
        
        self.tree_data.itemSelectionChanged.connect(self._on_tree_selection_changed)
        self.tree_data.itemChanged.connect(self._on_tree_item_changed)
        
        self.combo_method.currentTextChanged.connect(self.param_changed.emit)
        self.chk_sigma.stateChanged.connect(self.param_changed.emit)
        self.spin_sigma_lo.valueChanged.connect(self.param_changed.emit)
        self.spin_sigma_hi.valueChanged.connect(self.param_changed.emit)

        self.canvas_hist.range_changed.connect(self.hist_update_range_clicked.emit)
        self.btn_hist_reset.clicked.connect(lambda: self.hist_update_range_clicked.emit(-1, -1))

    def _apply_theme(self, *_):
        """Recolour the tree rows, which carry explicit per-item foregrounds."""
        normal = QtGui.QColor(theme_manager.color("text"))
        bad = QtGui.QColor(mpl_colors()["danger"])
        for i in range(self.tree_data.topLevelItemCount()):
            parent = self.tree_data.topLevelItem(i)
            parent.setForeground(0, normal)
            for j in range(parent.childCount()):
                child = parent.child(j)
                excluded = child.checkState(0) == QtCore.Qt.Unchecked
                child.setForeground(0, bad if excluded else normal)

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
        # A ticked box means "use this frame" -- the common convention. The
        # tree used to mean the opposite: you ticked a frame to CONDEMN it.
        excluded = item.checkState(0) == QtCore.Qt.Unchecked
        self.bad_frame_toggled.emit(set_type, index, excluded)

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

                # Ticked = included. Frames arrive ticked; unticking excludes.
                child.setCheckState(0, QtCore.Qt.Unchecked if i in fs.bad_indices
                                    else QtCore.Qt.Checked)
                child.setForeground(0, QtGui.QColor(
                    mpl_colors()["danger"] if i in fs.bad_indices
                    else theme_manager.color("text")))
        
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
