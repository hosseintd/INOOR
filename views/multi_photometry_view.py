import os
from PyQt5 import QtWidgets, QtCore, QtGui
from .components.zoom_image_canvas import ZoomImageCanvas
from .components.hist_canvas import MplHistCanvas
from .components.elided_label import ElidedLabel
from .theme import mpl_colors, set_variant, style_axes, style_figure, theme_manager
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
    plot_mode_changed = QtCore.pyqtSignal()
    all_plots_clicked = QtCore.pyqtSignal()
    measure_fwhm_clicked = QtCore.pyqtSignal()
    # The radial-profile and SNR figures are drawn by the controller from model
    # data, so the view can repaint their backgrounds but not their contents.
    # It says the theme moved and lets the controller redraw.
    theme_changed = QtCore.pyqtSignal()

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
        # Enough for the widest row (a label plus its spin box) without the
        # scroll area falling back to a horizontal scrollbar.
        left_panel.setMinimumWidth(310)
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
        # Discarding a loaded frame list should not look like adding to it.
        set_variant(self.btn_remove, "danger")
        set_variant(self.btn_remove_all, "danger")
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
        self.spin_exptime = QtWidgets.QDoubleSpinBox(); self.spin_exptime.setRange(0, 1e6); self.spin_exptime.setValue(1.0)
        
        pl_basic.addRow("Detection Stamp (px):", self.spin_stamp)
        pl_basic.addRow("Exp Time (s):", self.spin_exptime)
        params_vbox.addLayout(pl_basic)

        # --- Advanced Params (Collapsible) ---
        self.adv_box = CollapsibleBox("Advanced Settings")
        pl_adv = QtWidgets.QFormLayout()
        pl_adv.setContentsMargins(4, 4, 4, 4)
        
        self.spin_aperture = QtWidgets.QDoubleSpinBox(); self.spin_aperture.setDecimals(2)
        self.spin_aperture.setRange(1, 800); self.spin_aperture.setValue(10)
        # Commit the radius on Enter, not per keystroke. Each change re-measures
        # the star and update_params() then rewrites this box, so typing "7.25"
        # stopped at "7": the box became "7.00" before the "." arrived.
        self.spin_aperture.setKeyboardTracking(False)
        self.chk_fixed_aperture = QtWidgets.QCheckBox("Fixed aperture for all")
        self.cmb_ap_policy = QtWidgets.QComboBox()
        self.cmb_ap_policy.addItem("Max SNR (per star)", "snr")
        self.cmb_ap_policy.addItem("Curve of growth (per star)", "growth")
        self.cmb_ap_policy.addItem("Frame FWHM x factor", "frame_fwhm")
        self.cmb_ap_policy.setToolTip(
            "How the aperture radius is chosen.\n\n"
            "Max SNR (per star): the radius giving the highest signal-to-noise\n"
            "  for this star in this frame. Use it to measure one star in one\n"
            "  frame as precisely as possible. It gives each star a different\n"
            "  radius, so magnitudes measured this way are not directly\n"
            "  comparable between stars.\n\n"
            "Curve of growth (per star): the radius containing a chosen fraction\n"
            "  of that star's light. Almost independent of brightness, but read\n"
            "  off each star's own profile, so it still differs slightly from\n"
            "  star to star.\n\n"
            "Frame FWHM x factor: the radius comes from the PSF width of the\n"
            "  whole FRAME, so every star in a frame gets exactly the same\n"
            "  radius while the radius still follows the seeing from frame to\n"
            "  frame. Use this for light curves, and whenever a zeropoint\n"
            "  measured on one star will be applied to another.")
        self.spin_ap_param = QtWidgets.QDoubleSpinBox()
        self.spin_ap_param.setRange(0.05, 20.0); self.spin_ap_param.setDecimals(2)
        self.spin_ap_param.setSingleStep(0.05); self.spin_ap_param.setValue(0.90)
        self.lbl_ap_param = QtWidgets.QLabel("Enclosed fraction:")
        self.chk_ap_corr = QtWidgets.QCheckBox("Aperture correction (frame)")
        self.chk_ap_corr.setToolTip(
            "Add the frame's aperture correction so the reported magnitude is a\n"
            "TOTAL magnitude instead of just the light inside the aperture.\n\n"
            "The correction is measured from the curve of growth of bright,\n"
            "isolated stars in the frame. Turn it ON when comparing against a\n"
            "catalogue; it is unnecessary for differential light curves, where\n"
            "the aperture loss already cancels as long as the radius is shared.")
        self.spin_fwhm = QtWidgets.QDoubleSpinBox(); self.spin_fwhm.setRange(2.0, 100); self.spin_fwhm.setValue(12.0)
        self.spin_thresh = QtWidgets.QDoubleSpinBox(); self.spin_thresh.setRange(0.1, 100); self.spin_thresh.setValue(3.0)
        self.spin_inner = QtWidgets.QDoubleSpinBox(); self.spin_inner.setRange(1.1, 10); self.spin_inner.setValue(2.0)
        self.spin_outer = QtWidgets.QDoubleSpinBox(); self.spin_outer.setRange(1.2, 20); self.spin_outer.setValue(3.0)
        # Measured on PSR J2215 frame High_1 (27 stars): a 2r-3r annulus at
        # r = 1.05 FWHM holds ~3% of the star's own light.
        annulus_tip = (
            "Sky annulus, in units of the aperture radius r.\n"
            "With r near 1 FWHM a 2r-3r annulus still holds ~3% of the star's own\n"
            "light, so every flux reads ~8 mmag faint. It is the same for all stars\n"
            "of a frame and cancels in differential photometry and same-frame\n"
            "zeropoints, but not in absolute photometry.")
        self.spin_inner.setToolTip(annulus_tip)
        self.spin_outer.setToolTip(annulus_tip)
        self.spin_zp = QtWidgets.QDoubleSpinBox(); self.spin_zp.setRange(-100, 100); self.spin_zp.setValue(0.0)
        # Same reason as the Aperture box: any change ends in update_params(),
        # which rewrites every box here, so "1.07", "12.19" or "2.5" stopped at
        # the first digit. Values now apply on Enter or when the box loses focus.
        for box in (self.spin_exptime, self.spin_ap_param, self.spin_fwhm,
                    self.spin_thresh, self.spin_inner, self.spin_outer, self.spin_zp):
            box.setKeyboardTracking(False)

        # "Fit FWHM" is only a starting guess: it sets the star-detection kernel
        # and the radius grid that is searched. The radius actually used by the
        # Frame FWHM rule comes from measuring the frame, which is what the
        # button does -- on demand, because measuring it costs a second or two
        # and must not run again on every keystroke in this panel.
        self.btn_measure_fwhm = QtWidgets.QPushButton("Measure")
        self.btn_measure_fwhm.setToolTip(
            "Measure the PSF width of the CURRENT frame from its stars and put\n"
            "the result in this box. The Frame FWHM rule measures this per frame\n"
            "by itself during a bulk run; this button is for checking the value\n"
            "and for setting a sensible starting guess.")
        fwhm_row = QtWidgets.QHBoxLayout()
        fwhm_row.setContentsMargins(0, 0, 0, 0)
        fwhm_row.addWidget(self.spin_fwhm, 1)
        fwhm_row.addWidget(self.btn_measure_fwhm)
        fwhm_holder = QtWidgets.QWidget()
        fwhm_holder.setLayout(fwhm_row)
        self.lbl_fwhm_measured = QtWidgets.QLabel("")
        set_variant(self.lbl_fwhm_measured, "muted")

        pl_adv.addRow("Aperture (px):", self.spin_aperture)
        pl_adv.addRow("", self.chk_fixed_aperture)
        pl_adv.addRow("Auto radius rule:", self.cmb_ap_policy)
        pl_adv.addRow(self.lbl_ap_param, self.spin_ap_param)
        pl_adv.addRow("", self.chk_ap_corr)
        pl_adv.addRow("Fit FWHM (px):", fwhm_holder)
        pl_adv.addRow("", self.lbl_fwhm_measured)
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
        set_variant(self.btn_bulk, "primary")
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
        
        self.lbl_fname = ElidedLabel("No file loaded")
        self.lbl_fname.setAlignment(QtCore.Qt.AlignCenter)
        set_variant(self.lbl_fname, "filename")
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
        self.canvas_profile = FigureCanvas(self.fig_profile)
        profile_vbox.addWidget(self.canvas_profile)
        self.right_splitter.addWidget(profile_widget)
        
        # SNR Group
        snr_widget = QtWidgets.QWidget()
        snr_vbox = QtWidgets.QVBoxLayout(snr_widget)
        snr_vbox.setContentsMargins(0, 0, 0, 0)
        # The lower-right plot shows whichever curve the active aperture rule
        # actually reads its radius from; it follows the rule automatically but
        # stays switchable, because comparing the two is often the point.
        snr_hdr = QtWidgets.QHBoxLayout()
        snr_hdr.setContentsMargins(0, 0, 0, 0)
        self.cmb_plot_mode = QtWidgets.QComboBox()
        self.cmb_plot_mode.addItem("SNR vs Radius", "snr")
        self.cmb_plot_mode.addItem("Curve of Growth", "growth")
        self.cmb_plot_mode.setToolTip(
            "Follows the aperture rule by default. SNR vs radius is what the\n"
            "Max-SNR rule reads; the curve of growth is what the Curve-of-growth\n"
            "rule reads.")
        self.btn_all_plots = QtWidgets.QToolButton()
        self.btn_all_plots.setText("All plots")
        self.btn_all_plots.setToolTip("Show radial profile, SNR curve and curve of "
                                      "growth together, and save them.")
        snr_hdr.addWidget(self.cmb_plot_mode, 1)
        snr_hdr.addWidget(self.btn_all_plots, 0)
        snr_vbox.addLayout(snr_hdr)
        self.fig_snr = Figure(figsize=(4, 2.5), tight_layout=True)
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
        self.lbl_mag_title = QtWidgets.QLabel("Instr Mag:")
        rl.addRow(self.lbl_mag_title, self.lbl_mag)
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
        self.main_splitter.setSizes([340, 760, 350])

        # Widget colours come from the application stylesheet (views/theme.py).
        # Only the matplotlib figures need telling, since they are not Qt-styled.
        self._apply_theme()
        theme_manager.changed.connect(self._apply_theme)

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
        self.cmb_ap_policy.currentIndexChanged.connect(self._on_ap_policy_changed)
        self.cmb_plot_mode.currentIndexChanged.connect(lambda: self.plot_mode_changed.emit())
        self.btn_all_plots.clicked.connect(self.all_plots_clicked.emit)
        self.spin_ap_param.valueChanged.connect(self._on_ap_param_changed)
        self.chk_ap_corr.toggled.connect(lambda: self.param_changed.emit("aperture_correct"))
        self.btn_measure_fwhm.clicked.connect(self.measure_fwhm_clicked.emit)
        self.spin_stamp.valueChanged.connect(lambda: self.param_changed.emit("stamp_size"))
        self.spin_fwhm.valueChanged.connect(lambda: self.param_changed.emit("fwhm"))
        self.spin_thresh.valueChanged.connect(lambda: self.param_changed.emit("thresh"))
        self.spin_inner.valueChanged.connect(lambda: self.param_changed.emit("inner"))
        self.spin_outer.valueChanged.connect(lambda: self.param_changed.emit("outer"))
        self.spin_zp.valueChanged.connect(lambda: self.param_changed.emit("zeropoint"))
        self.spin_exptime.valueChanged.connect(lambda: self.param_changed.emit("exptime"))

    def _apply_theme(self, *_):
        """Repaint the two matplotlib panels for the current theme."""
        for fig, canvas in ((self.fig_profile, self.canvas_profile),
                            (self.fig_snr, self.canvas_snr)):
            style_figure(fig)
            # style_figure only touches the figure patch. The axes inside were
            # drawn by the controller with the previous theme's colours, so
            # restyle them too -- otherwise the panel keeps a dark plotting area
            # and unreadable pale tick labels on a light figure, until some
            # unrelated action (switching SNR/curve-of-growth) happens to redraw.
            for ax in fig.get_axes():
                style_axes(ax)
            canvas.draw_idle()
        # Rows carry an explicit foreground colour (bad frames are red), so the
        # good ones have to be recoloured by hand when the palette changes.
        self._repaint_rows()
        # Axis furniture is now right, but the plotted lines still carry the old
        # theme's colours; only the controller can redraw those from the data.
        self.theme_changed.emit()

    def _repaint_rows(self):
        normal = QtGui.QColor(theme_manager.color("text"))
        bad = QtGui.QColor(mpl_colors()["danger"])
        for i in range(self.list_files.count()):
            item = self.list_files.item(i)
            is_bad = bool(item.data(QtCore.Qt.UserRole + 1))
            item.setForeground(bad if is_bad else normal)

    # --- Update API ---
    def update_file_list(self, files, current_index, bad_indices=None):
        self.list_files.blockSignals(True)
        self.list_files.clear()
        bad_indices = bad_indices or set()
        normal = QtGui.QColor(theme_manager.color("text"))
        bad_color = QtGui.QColor(mpl_colors()["danger"])
        for i, f in enumerate(files):
            item = QtWidgets.QListWidgetItem(os.path.basename(f))
            item.setData(QtCore.Qt.UserRole, f)
            is_bad = i in bad_indices
            # Remembered so a theme switch can recolour the row later.
            item.setData(QtCore.Qt.UserRole + 1, is_bad)
            item.setForeground(bad_color if is_bad else normal)
            self.list_files.addItem(item)
        if 0 <= current_index < self.list_files.count():
            self.list_files.setCurrentRow(current_index)
        self.list_files.blockSignals(False)

    def set_results(self, mag, err, flux, snr, mag_note=None, zeropoint=0.0):
        # mag is instrumental + zeropoint, so with a zeropoint it is no longer
        # an instrumental magnitude.
        self.lbl_mag_title.setText("Mag (ZP):" if zeropoint else "Instr Mag:")
        text = f"{mag:.4f}" if mag is not None else "N/A"
        if mag is not None and mag_note:
            text += f"  ({mag_note})"
        self.lbl_mag.setText(text)
        self.lbl_err.setText(f"{err:.4f}" if err is not None else "N/A")
        self.lbl_flux.setText(f"{flux:.2f}" if flux is not None else "N/A")
        self.lbl_snr.setText(f"{snr:.2f}" if snr is not None else "N/A")

    def plot_mode(self):
        """Which curve the lower-right panel is showing ("snr" or "growth")."""
        return self.cmb_plot_mode.currentData() or "snr"

    def set_plot_mode(self, mode):
        i = self.cmb_plot_mode.findData(mode)
        if i >= 0 and i != self.cmb_plot_mode.currentIndex():
            self.cmb_plot_mode.blockSignals(True)
            self.cmb_plot_mode.setCurrentIndex(i)
            self.cmb_plot_mode.blockSignals(False)

    def aperture_policy(self):
        """Currently selected auto-radius rule ("snr" / "growth" / "frame_fwhm")."""
        return self.cmb_ap_policy.currentData() or "snr"

    # One spin box serves two different rules, so its value has to be remembered
    # per rule. Resetting it to the rule's default on every switch meant that
    # leaving "Frame FWHM x 1.2", looking at another rule and coming back gave
    # x 1.0 -- silently changing the reported magnitude.
    _AP_PARAM_DEFAULTS = {"frame_fwhm": 1.0, "growth": 0.90}

    def _remembered_ap_param(self, pol):
        if not hasattr(self, "_ap_param_memory"):
            self._ap_param_memory = dict(self._AP_PARAM_DEFAULTS)
        return self._ap_param_memory.get(pol, self._AP_PARAM_DEFAULTS.get(pol, 0.90))

    def _remember_ap_param(self, pol, value):
        if pol in self._AP_PARAM_DEFAULTS:
            if not hasattr(self, "_ap_param_memory"):
                self._ap_param_memory = dict(self._AP_PARAM_DEFAULTS)
            self._ap_param_memory[pol] = float(value)

    def _on_ap_param_changed(self, *_):
        self._remember_ap_param(self.aperture_policy(), self.spin_ap_param.value())
        self.param_changed.emit("aperture_policy")

    def _sync_ap_param_widget(self):
        """The one parameter box means different things per rule, so relabel it."""
        pol = self.aperture_policy()
        if pol == "frame_fwhm":
            self.lbl_ap_param.setText("Radius = FWHM x:")
            self.spin_ap_param.setRange(0.3, 6.0); self.spin_ap_param.setSingleStep(0.1)
            self.spin_ap_param.setToolTip(
                "Aperture radius as a multiple of the frame's PSF width.\n"
                "Around 1.0 gives the least scatter in a differential light\n"
                "curve; 1.5-2.0 keeps more of the star's light, which matters\n"
                "when the magnitude is compared against a catalogue.")
        else:
            self.lbl_ap_param.setText("Enclosed fraction:")
            self.spin_ap_param.setRange(0.50, 0.995); self.spin_ap_param.setSingleStep(0.01)
            self.spin_ap_param.setToolTip(
                "Fraction of the star's light the aperture should contain.\n"
                "Around 0.85-0.90 is stable; higher targets sit on the noisy\n"
                "outer part of the growth curve, where the radius is poorly\n"
                "determined.")
        both = pol in ("growth", "frame_fwhm")
        self.lbl_ap_param.setVisible(both)
        self.spin_ap_param.setVisible(both)

    def _on_ap_policy_changed(self, *_):
        pol = self.aperture_policy()
        self.set_plot_mode("growth" if pol == "growth" else "snr")
        self.spin_ap_param.blockSignals(True)
        self._sync_ap_param_widget()
        if pol in self._AP_PARAM_DEFAULTS:
            self.spin_ap_param.setValue(self._remembered_ap_param(pol))
        self.spin_ap_param.blockSignals(False)
        self.param_changed.emit("aperture_policy")

    def set_measured_fwhm(self, value, note=None):
        """Show the FWHM actually measured from the current frame."""
        if value is None:
            self.lbl_fwhm_measured.setText(note or "")
        else:
            self.lbl_fwhm_measured.setText(
                f"measured: {float(value):.2f} px" + (f"  ({note})" if note else ""))

    def update_params(self, p):
        self.blockSignals(True)
        self.spin_aperture.setValue(float(p.aperture))
        self.chk_fixed_aperture.setChecked(p.fixed_aperture)
        pol = getattr(p, "aperture_policy", "snr")
        i = self.cmb_ap_policy.findData(pol)
        if i >= 0:
            self.cmb_ap_policy.setCurrentIndex(i)
        # Keep the per-rule memory in step with the model, so a switch away and
        # back restores what the model holds rather than the built-in default.
        self._remember_ap_param("frame_fwhm", float(p.fwhm_factor))
        self._remember_ap_param("growth", float(p.growth_fraction))
        self.spin_ap_param.setValue(
            float(p.fwhm_factor) if pol == "frame_fwhm" else float(p.growth_fraction))
        self._sync_ap_param_widget()
        self.chk_ap_corr.setChecked(bool(getattr(p, "aperture_correct", False)))
        self.spin_stamp.setValue(int(p.stamp_size))
        # Tracking radius was missing here, so the model -> view direction of the
        # sync silently dropped it and the two could disagree.
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
            item.setData(QtCore.Qt.UserRole + 1, bool(is_bad))
            item.setForeground(QtGui.QColor(
                mpl_colors()["danger"] if is_bad else theme_manager.color("text")))
