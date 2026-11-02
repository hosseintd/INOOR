from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import os
from matplotlib.patches import Circle, Rectangle
from views.components.zoom_image_canvas import ZoomImageCanvas
from core import photometry_core as pc
from core.camera_config import detect_mode_or_none
from views.theme import mpl_colors, set_variant, theme_manager

# Fallbacks used only when the caller does not hand us the app's settings.
# The controller always does (see _review_params), so these matter for tests
# and for any older call site.
DEFAULT_PARAMS = {
    'stamp_size': 50,
    'fwhm': 12.0,
    'threshold': 3.0,
    'inner_coef': 2.0,
    'outer_coef': 3.0,
    'aperture': 10,
    'fixed_aperture': False,
    'aperture_policy': pc.POLICY_SNR,
    'growth_fraction': 0.90,
    'fwhm_factor': 1.0,
    'aperture_correct': False,
    'gain': None,
    'read_noise': None,
    'dark_noise': None,
    'exptime_override': None,
}


class CalibrationReviewDialog(QtWidgets.QDialog):
    """
    Review, and where necessary correct by hand, the reference star found in
    each frame of a zeropoint calibration run.

    Two views of the same frame are shown side by side: the whole field, and a
    detection stamp at full resolution around the current position. The stamp
    exists because the field view is decimated to ~1024 px for speed, so on a
    4096x4096 frame one screen pixel covers four detector pixels -- enough to
    hide a close neighbour entirely and far too coarse to click a blended star
    accurately. Clicking either view re-runs detection and photometry at that
    spot; the stamp is what makes that practical in a crowded field.

    A manual re-measure replaces the frame's entry in the calibration history
    in place, so it feeds straight back into that frame's zeropoint. It
    therefore runs through the same pc.measure_star with the same settings the
    bulk run used -- a magnitude measured through a different radius rule or a
    different sky annulus is not comparable with the rest of the series.
    """

    def __init__(self, parent, file_list, calibration_results_list, active_col_idx=0,
                 params=None):
        super().__init__(parent)
        self.setWindowTitle("Review Calibration Stars")
        self.resize(1240, 820)

        self.files = file_list
        self.history_item = calibration_results_list[active_col_idx]
        self.results_map = self.history_item.get('results', {})
        self.col_name = self.history_item.get('name', f"Ref Star {active_col_idx+1}")

        # None is a meaningful value only for the camera/exposure entries
        # (it means "fall back"); for the rest it means "not supplied".
        self.p = dict(DEFAULT_PARAMS)
        _nullable = ('gain', 'read_noise', 'dark_noise', 'exptime_override')
        for k, v in (params or {}).items():
            if v is not None or k in _nullable:
                self.p[k] = v

        self._current_idx = 0
        self.current_data = None
        self._stamp_center = None      # where the stamp view is centred (full-frame px)
        self._notfound_text = None     # artist, so it can be removed after a fix
        self._originals = {}           # idx -> result before the first manual edit
        self._frame_fwhm_cache = {}    # idx -> FWHM of that frame (frame_fwhm policy)
        self._frame_cog_cache = {}     # idx -> (radii, E(r)) for the aperture correction
        self._exptime_cache = {}

        self._setup_ui()
        self._apply_params()
        self.load_frame(0)

    # ------------------------------------------------------------------ UI

    def _setup_ui(self):
        self.layout = QtWidgets.QVBoxLayout(self)

        # Header Info
        top_layout = QtWidgets.QHBoxLayout()
        info = QtWidgets.QLabel(f"Reviewing: {self.col_name}")
        set_variant(info, "heading")
        top_layout.addWidget(info)
        top_layout.addStretch()
        self.lbl_policy = QtWidgets.QLabel("")
        set_variant(self.lbl_policy, "muted")
        top_layout.addWidget(self.lbl_policy)
        self.layout.addLayout(top_layout)

        # Main Content: full field | detection stamp + controls
        content_layout = QtWidgets.QHBoxLayout()

        self.canvas = ZoomImageCanvas(self)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.canvas.click_coords.connect(self.on_canvas_click)
        content_layout.addWidget(self.canvas, stretch=3)

        right = QtWidgets.QVBoxLayout()
        right.setSpacing(6)

        # --- Detection stamp -------------------------------------------
        grp_stamp = QtWidgets.QGroupBox("Detection Stamp")
        gs = QtWidgets.QVBoxLayout(grp_stamp)
        gs.setContentsMargins(4, 4, 4, 4)
        self.stamp = ZoomImageCanvas(self, figsize=(3.2, 3.2))
        self.stamp.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.stamp.setFixedHeight(290)
        self.stamp.click_coords.connect(self.on_stamp_click)
        self.stamp.setToolTip(
            "Full-resolution view around the current position.\n"
            "Click the star you want measured. The dashed box is the region\n"
            "the detector actually searches (Detection stamp, below).")
        gs.addWidget(self.stamp)
        right.addWidget(grp_stamp)

        # --- Controls ---------------------------------------------------
        controls = QtWidgets.QWidget()
        sb_layout = QtWidgets.QVBoxLayout(controls)
        sb_layout.setContentsMargins(0, 0, 0, 0)

        grp_det = QtWidgets.QGroupBox("Detection")
        fd = QtWidgets.QFormLayout(grp_det)
        self.spin_det_stamp = QtWidgets.QSpinBox()
        self.spin_det_stamp.setRange(5, 500)
        self.spin_det_stamp.setToolTip(
            "Half-size of the box searched around your click.\n"
            "Make it small in a crowded field so the detector cannot lock\n"
            "onto a brighter neighbour instead of your star.")
        self.spin_fwhm = QtWidgets.QDoubleSpinBox(); self.spin_fwhm.setRange(0.1, 100)
        self.spin_thresh = QtWidgets.QDoubleSpinBox(); self.spin_thresh.setRange(0.1, 100)
        self.chk_exact = QtWidgets.QCheckBox("Measure exactly where I click")
        self.chk_exact.setToolTip(
            "Skip centroiding entirely. Last resort for a star so blended that\n"
            "the fit always walks off onto its neighbour.")
        fd.addRow("Detection stamp (px):", self.spin_det_stamp)
        fd.addRow("FWHM (px):", self.spin_fwhm)
        fd.addRow("Threshold (sigma):", self.spin_thresh)
        fd.addRow("", self.chk_exact)
        sb_layout.addWidget(grp_det)

        grp_ap = QtWidgets.QGroupBox("Aperture")
        fa = QtWidgets.QFormLayout(grp_ap)
        self.cmb_rule = QtWidgets.QComboBox()
        self.cmb_rule.addItem("Max SNR (per star)", pc.POLICY_SNR)
        self.cmb_rule.addItem("Curve of growth (per star)", pc.POLICY_GROWTH)
        self.cmb_rule.addItem("Frame FWHM x factor", pc.POLICY_FRAME_FWHM)
        self.cmb_rule.setToolTip(
            "Must match the rule the rest of this calibration run used.\n"
            "Two stars measured through different radii enclose different\n"
            "fractions of their light, and that difference goes straight into\n"
            "the zeropoint as a systematic offset.")
        self.spin_rule_param = QtWidgets.QDoubleSpinBox()
        self.spin_rule_param.setDecimals(2)
        self.lbl_rule_param = QtWidgets.QLabel("Enclosed fraction:")
        self.chk_fixed_ap = QtWidgets.QCheckBox("Fixed:")
        self.spin_aperture = QtWidgets.QDoubleSpinBox(); self.spin_aperture.setDecimals(2)
        self.spin_aperture.setRange(1, 800)
        self.spin_aperture.setSuffix(" px")
        self.spin_inner = QtWidgets.QDoubleSpinBox(); self.spin_inner.setRange(1.1, 10)
        self.spin_outer = QtWidgets.QDoubleSpinBox(); self.spin_outer.setRange(1.2, 20)
        fixed_row = QtWidgets.QHBoxLayout()
        fixed_row.setContentsMargins(0, 0, 0, 0)
        fixed_row.addWidget(self.chk_fixed_ap)
        fixed_row.addWidget(self.spin_aperture, 1)
        ann_row = QtWidgets.QHBoxLayout()
        ann_row.setContentsMargins(0, 0, 0, 0)
        ann_row.addWidget(self.spin_inner, 1)
        ann_row.addWidget(self.spin_outer, 1)
        fa.addRow("Auto radius rule:", self.cmb_rule)
        fa.addRow(self.lbl_rule_param, self.spin_rule_param)
        fa.addRow("Override radius:", fixed_row)
        fa.addRow("Annulus in / out:", ann_row)
        sb_layout.addWidget(grp_ap)

        self.lbl_status = QtWidgets.QLabel("Click the star to re-measure it.")
        self.lbl_status.setWordWrap(True)
        set_variant(self.lbl_status, "muted")
        _italic = self.lbl_status.font()
        _italic.setItalic(True)
        self.lbl_status.setFont(_italic)
        sb_layout.addWidget(self.lbl_status)

        self.btn_revert = QtWidgets.QPushButton("Undo edit on this frame")
        self.btn_revert.setEnabled(False)
        sb_layout.addWidget(self.btn_revert)
        sb_layout.addStretch()

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(controls)
        right.addWidget(scroll, stretch=1)

        holder = QtWidgets.QFrame()
        holder.setFixedWidth(370)
        _t = theme_manager.tokens()
        holder.setStyleSheet(
            f"background: {_t['bg_surface']}; border-left: 1px solid {_t['border_muted']};")
        holder.setLayout(right)
        content_layout.addWidget(holder)

        self.layout.addLayout(content_layout)

        # Navigation Bar
        nav = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("<< Previous")
        self.lbl_frame = QtWidgets.QLabel("Frame 1/N")
        self.lbl_frame.setAlignment(QtCore.Qt.AlignCenter)
        set_variant(self.lbl_frame, "heading")
        self.btn_next = QtWidgets.QPushButton("Next >>")

        nav.addWidget(self.btn_prev)
        nav.addWidget(self.lbl_frame)
        nav.addWidget(self.btn_next)
        self.layout.addLayout(nav)

        # Footer Actions
        footer = QtWidgets.QHBoxLayout()
        self.btn_outliers = QtWidgets.QPushButton("Identify Outliers")
        self.btn_next_bad = QtWidgets.QPushButton("Next Not-Found")
        self.btn_close = QtWidgets.QPushButton("Done / Close")
        footer.addWidget(self.btn_outliers)
        footer.addWidget(self.btn_next_bad)
        footer.addStretch()
        footer.addWidget(self.btn_close)
        self.layout.addLayout(footer)

        # Connections
        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_next.clicked.connect(self.next_frame)
        self.btn_outliers.clicked.connect(self.identify_outliers)
        self.btn_next_bad.clicked.connect(self.goto_next_missing)
        self.btn_revert.clicked.connect(self.revert_frame)
        self.btn_close.clicked.connect(self.accept)
        self.cmb_rule.currentIndexChanged.connect(self._sync_rule_param)
        self.spin_det_stamp.valueChanged.connect(self._refresh_stamp)
        self.chk_fixed_ap.toggled.connect(self._sync_enabled)
        self.spin_aperture.valueChanged.connect(
            lambda: self.lbl_policy.setText(f"Settings: {self._policy_label()}"))
        self.spin_rule_param.valueChanged.connect(
            lambda: self.lbl_policy.setText(f"Settings: {self._policy_label()}"))

        # Styling comes from the application stylesheet (views/theme.py).

    def _apply_params(self):
        """Seed every control from the app's current settings."""
        p = self.p
        self.spin_det_stamp.setValue(int(p['stamp_size']))
        self.spin_fwhm.setValue(float(p['fwhm']))
        self.spin_thresh.setValue(float(p['threshold']))
        self.spin_inner.setValue(float(p['inner_coef']))
        self.spin_outer.setValue(float(p['outer_coef']))
        self.spin_aperture.setValue(float(p['aperture']))
        self.chk_fixed_ap.setChecked(bool(p['fixed_aperture']))
        i = self.cmb_rule.findData(p['aperture_policy'])
        if i >= 0:
            self.cmb_rule.setCurrentIndex(i)
        self._sync_rule_param()
        self.lbl_policy.setText(f"Settings: {self._policy_label()}")

    def _sync_rule_param(self):
        """The parameter spinbox means a different quantity per rule, so its
        range, step and label are rebuilt when the rule changes."""
        pol = self.cmb_rule.currentData() or pc.POLICY_SNR
        if pol == pc.POLICY_FRAME_FWHM:
            self.spin_rule_param.setRange(0.3, 6.0)
            self.spin_rule_param.setSingleStep(0.1)
            self.spin_rule_param.setPrefix("x ")
            self.lbl_rule_param.setText("FWHM factor:")
            self.spin_rule_param.setValue(float(self.p['fwhm_factor']))
        else:
            self.spin_rule_param.setRange(0.50, 0.995)
            self.spin_rule_param.setSingleStep(0.01)
            self.spin_rule_param.setPrefix("E ")
            self.lbl_rule_param.setText("Enclosed fraction:")
            self.spin_rule_param.setValue(float(self.p['growth_fraction']))
        self._sync_enabled()

    def _sync_enabled(self):
        """Which aperture controls are live. Kept apart from _sync_rule_param
        so toggling the fixed-aperture box does not throw away a factor the
        user has just typed."""
        pol = self.cmb_rule.currentData() or pc.POLICY_SNR
        fixed = self.chk_fixed_ap.isChecked()
        show = (pol != pc.POLICY_SNR) and not fixed
        self.spin_rule_param.setVisible(show)
        self.lbl_rule_param.setVisible(show)
        self.cmb_rule.setEnabled(not fixed)
        self.spin_aperture.setEnabled(fixed)
        self.lbl_policy.setText(f"Settings: {self._policy_label()}")

    def _policy_label(self):
        if self.chk_fixed_ap.isChecked():
            return f"fixed {self.spin_aperture.value():g} px"
        pol = self.cmb_rule.currentData() or pc.POLICY_SNR
        if pol == pc.POLICY_FRAME_FWHM:
            return f"frame FWHM x {self.spin_rule_param.value():g}"
        if pol == pc.POLICY_GROWTH:
            return f"growth E={self.spin_rule_param.value():g}"
        return "max SNR"

    # ------------------------------------------------- per-frame metadata

    def _header(self, idx):
        from astropy.io import fits
        try:
            return fits.getheader(self.files[idx])
        except Exception:
            return None

    def _camera_params(self, idx):
        """
        (gain, read_noise, dark_noise, mode) for this frame, resolved the way
        BulkPhotometryWorker does it -- from the frame's own header/filename,
        falling back to the batch values when the mode cannot be identified.
        """
        try:
            mode = detect_mode_or_none(self._header(idx), self.files[idx])
        except Exception:
            mode = None
        if mode is not None:
            # Dark charge = dark current x this frame's exposure, as in the
            # bulk worker that produced the rest of the run.
            return (mode.gain, mode.read_noise,
                    mode.dark_charge(self._exptime(idx)), mode.name)
        d = pc.DEFAULT_MODE
        return (self.p['gain'] if self.p['gain'] is not None else d.gain,
                self.p['read_noise'] if self.p['read_noise'] is not None else d.read_noise,
                self.p['dark_noise'] if self.p['dark_noise'] is not None else d.dark_noise,
                None)

    def _exptime(self, idx):
        if idx in self._exptime_cache:
            return self._exptime_cache[idx]
        et = None
        old = self.results_map.get(idx) or {}
        if old.get('exptime'):
            et = float(old['exptime'])
        if et is None:
            hdr = self._header(idx)
            if hdr is not None:
                for key in ('EXPTIME', 'EXPOSURE'):
                    try:
                        if key in hdr and float(hdr[key]) > 0:
                            et = float(hdr[key]); break
                    except Exception:
                        pass
        if et is None and (self.p['exptime_override'] or 0) > 0:
            et = float(self.p['exptime_override'])
        self._exptime_cache[idx] = et
        return et

    def _frame_fwhm(self, idx):
        """PSF width of this frame; measured once, then cached."""
        if idx not in self._frame_fwhm_cache:
            try:
                self._frame_fwhm_cache[idx] = pc.estimate_frame_fwhm(
                    self.current_data, fwhm_guess=self.spin_fwhm.value())
            except Exception:
                self._frame_fwhm_cache[idx] = None
        return self._frame_fwhm_cache[idx]

    def _frame_cog(self, idx, fwhm):
        """Curve of growth of this frame, for the aperture correction; cached."""
        if idx not in self._frame_cog_cache:
            try:
                self._frame_cog_cache[idx] = pc.frame_growth_curve(
                    self.current_data, fwhm,
                    inner_coef=float(self.spin_inner.value()),
                    outer_coef=float(self.spin_outer.value()))
            except Exception:
                self._frame_cog_cache[idx] = (None, None)
        return self._frame_cog_cache[idx]

    # -------------------------------------------------------- navigation

    def identify_outliers(self):
        # Scan results for potential outliers (e.g. mag deviates > 2 sigma from median)
        mags = []
        indices = []
        for idx in range(len(self.files)):
            res = self.results_map.get(idx, {})
            if res and res.get('success'):
                m = res.get('mag')
                if m is not None:
                    mags.append(m)
                    indices.append(idx)

        if not mags:
            QtWidgets.QMessageBox.information(self, "No Data", "No valid magnitudes to check.")
            return

        med = np.median(mags)
        std = np.std(mags)

        # Simple Z-score check
        outliers = []
        for i, m, idx in zip(range(len(mags)), mags, indices):
            if abs(m - med) > 2.5 * std:  # >2.5 sigma
                outliers.append(idx)

        if not outliers:
            QtWidgets.QMessageBox.information(self, "Clean", "No obvious statistical outliers found.")
        else:
            # Ask user to review them
            res = QtWidgets.QMessageBox.question(self, "Outliers Found",
                                                 f"Found {len(outliers)} frames with potential outlier magnitudes.\nReview them now?",
                                                 QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if res == QtWidgets.QMessageBox.Yes:
                # Navigate to first outlier
                self.load_frame(outliers[0])
                self.lbl_status.setText(f"Outlier Review: Frame {outliers[0]+1} (Mag deviates significantly)")

    def goto_next_missing(self):
        """Jump to the next frame where the star was not found -- those are the
        frames that actually need a hand."""
        n = len(self.files)
        for step in range(1, n + 1):
            idx = (self._current_idx + step) % n
            res = self.results_map.get(idx)
            if not (res and res.get('success')):
                self.load_frame(idx)
                self.lbl_status.setText("Star not found here. Click it in the stamp.")
                return
        QtWidgets.QMessageBox.information(self, "All Found",
                                          "Every frame has a detection for this reference star.")

    def prev_frame(self):
        if self._current_idx > 0:
            self._current_idx -= 1
            self.load_frame(self._current_idx)

    def next_frame(self):
        if self._current_idx < len(self.files) - 1:
            self._current_idx += 1
            self.load_frame(self._current_idx)

    def _nearest_known_xy(self, idx):
        """Where to point the stamp when this frame has no detection: the
        position from the closest frame that does have one."""
        res = self.results_map.get(idx)
        if res and res.get('success') and res.get('picked_full'):
            return tuple(res['picked_full'])
        n = len(self.files)
        for step in range(1, n):
            for j in (idx - step, idx + step):
                if 0 <= j < n:
                    r = self.results_map.get(j)
                    if r and r.get('success') and r.get('picked_full'):
                        return tuple(r['picked_full'])
        return None

    def load_frame(self, idx):
        self._current_idx = idx
        tag = " [MANUAL]" if idx in self._originals else ""
        self.lbl_frame.setText(
            f"Frame {idx+1} / {len(self.files)} : {os.path.basename(self.files[idx])}{tag}")
        self.btn_revert.setEnabled(idx in self._originals)
        self._notfound_text = None

        # Load FITS
        from astropy.io import fits
        try:
            with fits.open(self.files[idx]) as hdul:
                # Assume image data is in first HDU with data
                data = None
                for hdu in hdul:
                    if hdu.data is not None:
                        data = hdu.data
                        break
                if data is not None:
                    self.current_data = data.astype('float32')
                else:
                    self.current_data = None

            if self.current_data is not None:
                # Optimization for display
                ny, nx = self.current_data.shape
                max_dim = 1024
                if max(ny, nx) > max_dim:
                    scale = int(np.ceil(max(ny, nx) / max_dim))
                    display_data = self.current_data[::scale, ::scale].copy()
                else:
                    display_data = self.current_data

                # Show on canvas with extent to keep coordinates correct
                self.canvas.show_image(display_data, extent=[0, nx, 0, ny])
                self._stamp_center = self._nearest_known_xy(idx)
                self._visualize_result()
            else:
                self.canvas.show_image(None)
                self.stamp.show_image(None)

        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            self.current_data = None
            self.canvas.show_image(None)
            self.stamp.show_image(None)

    # ----------------------------------------------------- visualisation

    def _draw_markers(self, canvas, xy, r, inner_r, outer_r):
        # Overlay colours follow the theme so the night mode stays free of the
        # green and blue that would cost the observer their dark adaptation.
        c = mpl_colors()
        canvas.add_patch(Circle(xy, r, color=c["accent"], fill=False, lw=1.5))
        canvas.add_patch(Circle(xy, inner_r, color=c["warning"], fill=False, lw=1, ls='--'))
        canvas.add_patch(Circle(xy, outer_r, color=c["warning"], fill=False, lw=1, ls='--'))
        canvas.add_patch(Circle(xy, 3, color=c["danger"], fill=True))

    def _refresh_stamp(self):
        """Redraw the zoomed cutout around the current position."""
        if self.current_data is None:
            self.stamp.show_image(None)
            return
        center = self._stamp_center
        if center is None:
            self.stamp.show_image(None)
            return

        cx, cy = float(center[0]), float(center[1])
        search = int(self.spin_det_stamp.value())
        res = self.results_map.get(self._current_idx)

        # Wide enough to show the search box AND the whole sky annulus. The
        # annulus matters: a neighbour that lands inside it biases the
        # background and therefore the magnitude, and that is invisible unless
        # the outer ring is on screen. A tight detection stamp (used to stop
        # the detector hopping onto that neighbour) must not shrink the view
        # below the annulus.
        outer = 0.0
        if res and res.get('success'):
            r = float(res.get('aperture_used') or 0.0)
            ap = res.get('aperture_result', {}) or {}
            outer = r * float(ap.get('outer_coef', self.spin_outer.value()))
        half = int(round(max(30.0, search * 1.5, outer * 1.15)))
        ny, nx = self.current_data.shape
        x0 = int(max(0, np.floor(cx - half))); x1 = int(min(nx, np.ceil(cx + half)))
        y0 = int(max(0, np.floor(cy - half))); y1 = int(min(ny, np.ceil(cy + half)))
        if x1 - x0 < 3 or y1 - y0 < 3:
            self.stamp.show_image(None)
            return

        self.stamp.show_image(self.current_data[y0:y1, x0:x1], extent=[x0, x1, y0, y1])

        # The region detection will actually consider.
        self.stamp.add_patch(Rectangle((cx - search, cy - search), 2 * search, 2 * search,
                                       edgecolor=mpl_colors()["warning"],
                                       facecolor='none', lw=1.0, ls=':'))

        if res and res.get('success') and res.get('picked_full'):
            xy = tuple(res['picked_full'])
            r = float(res.get('aperture_used') or 10.0)
            ap = res.get('aperture_result', {}) or {}
            self._draw_markers(self.stamp, xy, r,
                               r * float(ap.get('inner_coef', self.spin_inner.value())),
                               r * float(ap.get('outer_coef', self.spin_outer.value())))
        self.stamp.draw_idle()

    def _visualize_result(self):
        self.canvas.clear_patches()
        if self._notfound_text is not None:
            try:
                self._notfound_text.remove()
            except Exception:
                pass
            self._notfound_text = None

        res = self.results_map.get(self._current_idx)

        if res and res.get('success'):
            xy = res.get('picked_full')
            r = float(res.get('aperture_used') or 10.0)
            if xy:
                ap = res.get('aperture_result', {}) or {}
                inner_r = r * float(ap.get('inner_coef', self.spin_inner.value()))
                outer_r = r * float(ap.get('outer_coef', self.spin_outer.value()))
                self._draw_markers(self.canvas, xy, r, inner_r, outer_r)

                mag_val = res.get('mag')
                mag_str = f"{mag_val:.3f}" if mag_val is not None else "N/A"
                how = " (manual)" if res.get('manual') else ""
                self.lbl_status.setText(
                    f"Found at ({xy[0]:.1f}, {xy[1]:.1f}){how}\n"
                    f"Instr mag: {mag_str}   r = {r:.2f} px")
        else:
            self._notfound_text = self.canvas.figure.text(
                0.5, 0.5, "Star Not Found", color=mpl_colors()["danger"],
                ha='center', va='center', fontsize=20,
                bbox=dict(facecolor=theme_manager.color("bg_canvas"), alpha=0.6))
            self.lbl_status.setText(
                "Not found in this frame. Click the star in the stamp on the right.")

        self.canvas.draw()
        self._refresh_stamp()

    # -------------------------------------------------------- re-measure

    def on_canvas_click(self, x, y):
        # Move the stamp AND measure. The field view is decimated to ~1024 px,
        # so a click on it is only accurate to a few detector pixels -- but the
        # detector re-centres inside the detection stamp anyway, so for an
        # isolated star that is already enough. Requiring a second click in the
        # stamp for every frame made reviewing a long run needlessly slow; the
        # stamp is still there, and still the right tool for a blended star.
        self._stamp_center = (float(x), float(y))
        self._refresh_stamp()
        if self.chk_exact.isChecked():
            # "Exactly where I click" plus a decimated view is a bad pairing:
            # honour the request for precision by asking for the precise click.
            self.lbl_status.setText(
                f"Stamp moved to ({x:.0f}, {y:.0f}).\n"
                "'Measure exactly where I click' is on -- click in the stamp so "
                "the position is exact.")
            return
        self._remeasure_at(float(x), float(y))

    def on_stamp_click(self, x, y):
        self._remeasure_at(float(x), float(y))

    def _remeasure_at(self, x, y):
        if self.current_data is None:
            return

        idx = self._current_idx
        stamp_px = int(self.spin_det_stamp.value())
        fwhm = float(self.spin_fwhm.value())

        if self.chk_exact.isChecked():
            found_xy, method = (x, y), 'manual-exact'
        else:
            # Same strict localisation the main window's click path uses: the
            # search box is the detection stamp, no expansion, so the detector
            # cannot wander onto a brighter neighbour.
            found_xy, _params, _stamp, method = pc.detect_then_refine(
                self.current_data, (x, y),
                crop_half_size=stamp_px, stamp_radius=stamp_px,
                fwhm=fwhm, threshold_sigma=float(self.spin_thresh.value()),
                expand_steps=(0,))
            if not found_xy:
                self.lbl_status.setText(
                    f"No star found within {stamp_px} px of ({x:.1f}, {y:.1f}).\n"
                    "Raise the detection stamp, lower the threshold, or tick\n"
                    "'Measure exactly where I click'.")
                return
            method = 'manual'

        override = float(self.spin_aperture.value()) if self.chk_fixed_ap.isChecked() else None
        policy = self.cmb_rule.currentData() or pc.POLICY_SNR
        param = float(self.spin_rule_param.value())
        # The aperture correction is not offered as a control here: a manually
        # re-measured frame must land on the same magnitude scale as the rest of
        # the run, so it simply follows whatever the run itself used.
        ap_correct = bool(self.p.get('aperture_correct', False))
        frame_fwhm = None
        if (override is None and policy == pc.POLICY_FRAME_FWHM) or ap_correct:
            frame_fwhm = self._frame_fwhm(idx)
        frame_cog = self._frame_cog(idx, frame_fwhm or fwhm) if ap_correct else None

        gain, read_noise, dark_noise, mode = self._camera_params(idx)
        exptime = self._exptime(idx)

        # Zeropoint 0: a calibration run measures instrumental magnitudes, and
        # the zeropoint is formed from them afterwards.
        m = pc.measure_star(
            self.current_data, found_xy,
            fwhm=fwhm,
            aperture_override=override,
            aperture_policy=policy,
            growth_fraction=(param if policy == pc.POLICY_GROWTH else self.p['growth_fraction']),
            fwhm_factor=(param if policy == pc.POLICY_FRAME_FWHM else self.p['fwhm_factor']),
            frame_fwhm=frame_fwhm,
            aperture_correct=ap_correct,
            frame_cog=frame_cog,
            inner_coef=float(self.spin_inner.value()),
            outer_coef=float(self.spin_outer.value()),
            gain=gain, read_noise=read_noise, dark_noise=dark_noise,
            zeropoint=0.0, exptime=exptime)

        ap_res = m['phot']
        ap_res['inner_coef'] = float(self.spin_inner.value())
        ap_res['outer_coef'] = float(self.spin_outer.value())

        # Keep the automatic result so a mis-click can be undone.
        if idx not in self._originals:
            self._originals[idx] = self.results_map.get(idx)

        # Same keys BulkPhotometryWorker emits, so the zeropoint table and the
        # provenance recorded on Apply read this entry the same as any other.
        self.results_map[idx] = {
            'index': idx,
            'success': True,
            'msg': f'Manual re-measure ({self._policy_label()})',
            'method': method,
            'manual': True,
            'file': os.path.basename(self.files[idx]),
            'picked_full': found_xy,
            'radii': np.asarray(m['radii']).tolist() if m['radii'] is not None else None,
            'snrs': np.asarray(m['snrs']).tolist() if m['snrs'] is not None else None,
            'rp_radii': np.asarray(m['rp_radii']).tolist() if m['rp_radii'] is not None else None,
            'rp_profile': np.asarray(m['rp_profile']).tolist() if m['rp_profile'] is not None else None,
            'r_best': float(m['r_best']),
            'aperture_used': float(m['r_used']),
            'mag': ap_res['mag'],
            'mag_err': ap_res['mag_err'],
            'flux': ap_res['flux'],
            'snr': ap_res['snr'],
            'aperture_result': ap_res,
            'camera_mode': mode,
            'gain_used': float(gain),
            'read_noise_used': float(read_noise),
            'dark_noise_used': float(dark_noise),
            'exptime': (float(exptime) if exptime is not None else None),
        }

        self._stamp_center = tuple(found_xy)
        self.btn_revert.setEnabled(True)
        self.lbl_frame.setText(
            f"Frame {idx+1} / {len(self.files)} : "
            f"{os.path.basename(self.files[idx])} [MANUAL]")
        self._visualize_result()

        mag_str = f"{ap_res['instr_mag']:.3f}" if ap_res.get('instr_mag') is not None else "N/A"
        self.lbl_status.setText(
            f"Updated: ({found_xy[0]:.1f}, {found_xy[1]:.1f})\n"
            f"Instr mag: {mag_str}   r = {m['r_used']:.2f} px  [{self._policy_label()}]")

    def revert_frame(self):
        """Put back the automatic measurement for this frame."""
        idx = self._current_idx
        if idx not in self._originals:
            return
        original = self._originals.pop(idx)
        if original is None:
            self.results_map.pop(idx, None)
        else:
            self.results_map[idx] = original
        self.btn_revert.setEnabled(False)
        self.lbl_frame.setText(
            f"Frame {idx+1} / {len(self.files)} : {os.path.basename(self.files[idx])}")
        self._stamp_center = self._nearest_known_xy(idx)
        self._visualize_result()
        self.lbl_status.setText("Reverted to the automatic measurement.")
