import os
import sys
import json
import numpy as np
from datetime import datetime, timezone
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QProgressDialog

from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
import astropy.units as u
import matplotlib.pyplot as plt

# Local imports
from models.multi_photometry_model import MultiPhotometryModel, PhotometryRow
from views.multi_photometry_view import MultiPhotometryView
from views.components.photometry_table_dialog import PhotometryTableDialog
from views.components.zeropoint_calibration_dialog import ZeropointCalibrationDialog
from views.components.advanced_lightcurve_dialog import AdvancedLightCurveDialog
from views.components.extinction_plot_dialog import ExtinctionPlotDialog

from core import photometry_core as pc
from core.multi_photometry_worker import BulkPhotometryWorker, LAST_POSITION
from utils.gui_helpers import (plot_profile_and_snr, plot_growth_curve,
                               ExtinctionDialog, get_output_dir)
from views.theme import mpl_colors, save_publication_figure, set_variant
from utils.image_cache import ImageDisplayCache
from utils.exptime_utils import get_exptime_seconds
from core.camera_config import detect_mode_or_none
from utils.logging_config import get_logger
from utils.thread_guard import is_worker_busy

log = get_logger(__name__)
from core import masterFrame_creator as mfc
from views.components.display_settings_dialog import DisplaySettingsDialog

# --- Worker Threads ---

class DetectionWorker(QtCore.QThread):
    # Sends: found_xy, params, stamp, method, result_dict
    finished = QtCore.pyqtSignal(object, object, object, str, dict)
    
    def __init__(self, img, x, y, model_params):
        super().__init__()
        self.img = img
        self.x = x
        self.y = y
        self.p = model_params # dict containing necessary params
        
    def run(self):
        try:
            # 1. Detect & Refine
            # Use strict localization: crop_half_size and stamp_radius both use detection_stamp
            found_xy, params, stamp, method = pc.detect_then_refine(
                self.img, (self.x, self.y),
                crop_half_size=self.p['stamp_size'],
                stamp_radius=self.p['stamp_size'],
                fwhm=self.p['fwhm'],
                threshold_sigma=self.p['threshold'],
                expand_steps=(0,) # No expansion for strict click detection
            )
            
            phot_res = {}
            if found_xy:
                # 2. Measurement: one shared routine (pc.measure_star) also used
                # by _measure_frame and BulkPhotometryWorker, so a click and a
                # bulk run on the same star produce the same curves and the same
                # aperture. This path used to inline its own variant that fed a
                # placeholder radius of 10.0 to the radial profile, giving a
                # visibly different plot from bulk.
                m = pc.measure_star(
                    self.img, found_xy,
                    fwhm=self.p['fwhm'],
                    aperture_override=(self.p['aperture'] if self.p['fixed_ap'] else None),
                    aperture_policy=self.p.get('ap_policy', pc.POLICY_SNR),
                    growth_fraction=self.p.get('growth_fraction', 0.90),
                    fwhm_factor=self.p.get('fwhm_factor', 1.0),
                    frame_fwhm=self.p.get('frame_fwhm'),
                    aperture_correct=self.p.get('aperture_correct', False),
                    frame_cog=self.p.get('frame_cog'),
                    inner_coef=self.p['inner'], outer_coef=self.p['outer'],
                    gain=self.p['gain'], read_noise=self.p['read_noise'],
                    dark_noise=self.p['dark_noise'],
                    zeropoint=self.p['zp'], exptime=self.p['exptime'],
                    linear_limit=self.p.get('linear_limit'),
                    frame_fwhm_spread=self.p.get('frame_fwhm_spread'))

                phot_res = {
                    'r_used': m['r_used'],
                    'radii': m['radii'], 'snrs': m['snrs'], 'r_best': m['r_best'],
                    'rp_rad': m['rp_radii'], 'rp_prof': m['rp_profile'],
                    'phot': m['phot']
                }

            self.finished.emit(found_xy, params, stamp, method, phot_res)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(None, None, None, 'error', {})

from controllers.workers import ImagePreloadWorker


class InitialLoadWorker(ImagePreloadWorker):
    """Backwards-compatible alias; the implementation now lives in
    controllers.workers so the calibration tab can share it."""

    def __init__(self, files, cache, max_dim=None):
        super().__init__(files, cache)

class MultiPhotometryController(QtCore.QObject):
    def __init__(self, model: MultiPhotometryModel, view: MultiPhotometryView):
        super().__init__()
        self.model = model
        self.view = view
        
        self.current_img = None
        self.image_cache = ImageDisplayCache(max_display_dim=1024) # Speed up 4k display
        self.zeropoint_calibration_mode = False
        self.calibration_ref_xy = None
        self.confirm_cancel_buttons = None
        self.calibration_history = [] # List of runs for the current calibration session
        # Per-frame PSF measurements, keyed by frame index. Measuring the frame
        # FWHM costs a second or two, so the interactive paths must never redo
        # it on every parameter tweak; a bulk run measures it inside the worker.
        self._frame_fwhm_cache = {}
        self._frame_fwhm_spread_cache = {}   # star-to-star FWHM scatter (PSF check)
        self._frame_cog_cache = {}

        # Connect View signals
        self.view.add_files_clicked.connect(self.add_files)
        self.view.remove_files_clicked.connect(self.remove_files)
        self.view.remove_all_clicked.connect(self.remove_all)
        self.view.file_selected.connect(self.select_file)
        self.view.prev_clicked.connect(self.go_prev)
        self.view.next_clicked.connect(self.go_next)
        self.view.bad_frame_toggled.connect(self.toggle_bad)
        self.view.canvas_clicked.connect(self.on_canvas_click)
        self.view.param_changed.connect(self.update_params_from_view)
        self.view.plot_mode_changed.connect(self.on_plot_mode_changed)
        # Redraw the radial-profile / SNR curves in the new theme's colours.
        # Reuses the same refresh path as a plot-mode switch, which is why
        # flipping SNR <-> curve-of-growth used to be the only way to fix them.
        self.view.theme_changed.connect(self.show_current_data)
        self.view.all_plots_clicked.connect(self.on_all_plots_clicked)
        self.view.measure_fwhm_clicked.connect(self.measure_frame_fwhm)
        self.view.view_changed.connect(self.on_view_changed)
        
        self.view.bulk_photometry_clicked.connect(self.start_bulk)
        self.view.calibrate_zeropoints_clicked.connect(self.calibrate_zeropoints)
        self.view.calc_extinction_clicked.connect(self.calc_extinction)
        self.view.review_table_clicked.connect(self.review_table)
        self.view.show_lightcurve_clicked.connect(self.show_lightcurve)
        self.view.save_csv_clicked.connect(self.save_csv)
        self.view.save_snr_clicked.connect(self.save_snr)
        self.view.save_profile_clicked.connect(self.save_profile)
        self.view.display_settings_clicked.connect(self.on_display_settings_clicked)
        self.view.sort_clicked.connect(self.sort_by_time)
        
        # Connect reordering signal
        self.view.list_files.model().rowsMoved.connect(self.on_rows_moved)
        
        # Connect Model signals
        self.model.files_changed.connect(self.on_files_changed)
        self.model.selection_changed.connect(self.on_selection_changed)
        self.model.data_changed.connect(self.show_current_data)

        # Show the model's settings before any frame is loaded; otherwise the
        # view's own initial values (e.g. the first radius rule in the list)
        # are what the first parameter edit copies into the model.
        self.view.update_params(self.model)
        self.refresh_all()

    def refresh_all(self):
        self.on_files_changed()
        self.on_selection_changed(self.model._current_index)

    @QtCore.pyqtSlot()
    def on_files_changed(self):
        # The PSF caches are keyed by frame index, so adding, removing or
        # reordering files invalidates all of them.
        self._invalidate_frame_psf()
        bad_indices = {i for i, d in self.model._frame_data.items() if d.is_bad}
        self.view.update_file_list(self.model.files, self.model._current_index, bad_indices=bad_indices)
        # If we just loaded files for the first time, or after a clear, trigger load
        if len(self.model.files) > 0:
            if self.current_img is None:
                self.model.set_current_index(0)
            else:
                # Force reload of current image/data if index shifted due to sort
                self.on_selection_changed(self.model._current_index)
        elif len(self.model.files) == 0:
            self.on_selection_changed(-1)

    @QtCore.pyqtSlot()
    def on_display_settings_clicked(self):
        dlg = DisplaySettingsDialog(self.view, self.model.display_max_dim)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_dim = dlg.get_value()
            if new_dim != self.model.display_max_dim:
                self.model.display_max_dim = new_dim
                self.image_cache.max_display_dim = new_dim
                self.image_cache.clear() # Clear so they reload at new resolution
                self.on_selection_changed(self.model._current_index)

    def on_selection_changed(self, index):
        if index < 0 or index >= len(self.model.files):
            self.current_img = None
            self.view.canvas.show_image(None)
            self.view.lbl_fname.setText("No file loaded")
            self.view.canvas_hist.show_hist(None, None)
            self.view.canvas_profile.figure.clear()
            self.view.canvas_snr.figure.clear()
            self.view.canvas_profile.draw()
            self.view.canvas_snr.draw()
            # No frame, no result: after Clear All the panel and the ZP box
            # kept showing the last frame's numbers.
            self.view.set_results(None, None, None, None)
            self.view.spin_zp.blockSignals(True)
            self.view.spin_zp.setValue(self.model.zeropoint)
            self.view.spin_zp.blockSignals(False)
            # Ensure view list has no selection or reflects -1
            self.view.list_files.blockSignals(True)
            self.view.list_files.clearSelection()
            self.view.list_files.blockSignals(False)
            return
            
        path = self.model.files[index]
        self.view.lbl_fname.setText(os.path.basename(path))
        
        # Sync the list widget selection (block signals to prevent recursion)
        self.view.list_files.blockSignals(True)
        self.view.list_files.setCurrentRow(index)
        self.view.list_files.blockSignals(False)
        
        if self.image_cache.max_display_dim != self.model.display_max_dim:
            self.image_cache.max_display_dim = self.model.display_max_dim
            
        try:
            # Use display cache
            full_data = mfc.load_fits(path)
            display_entry = self.image_cache.get_display_data(path, full_data)
            self.current_img = full_data
            
            # Show on canvas using extent to preserve coordinates
            # Extent: [x0, x1, y0, y1] original
            nx, ny = display_entry['original_shape'][1], display_entry['original_shape'][0]
            
            data = self.model.get_current_data()
            if not data:
                # Initialize per-frame data if missing
                self.model.update_frame_data(index,
                    fname=os.path.basename(path),
                    full_path=path,
                    exptime=None,
                    date_ut=None,
                    date_obs=None,
                    selected_source=None,
                    aperture_radius=self.model.aperture_radius,
                    rp_radii=None, rp_profile=None,
                    radii=None, snrs=None, r_best=None,
                    mag=None, mag_err=None, flux=None, snr=None,
                    is_bad=False,
                    view_xlim=None, view_ylim=None,
                    vmin=None, vmax=None
                )
                data = self.model.get_current_data()
            
            # Prefer stored limits, otherwise use cache-calculated ZScale
            use_vmin = data.vmin if (data and data.vmin is not None) else display_entry['vmin']
            use_vmax = data.vmax if (data and data.vmax is not None) else display_entry['vmax']

            self.view.canvas.show_image(display_entry['display_data'], 
                                       vmin=use_vmin, vmax=use_vmax,
                                       extent=[0, nx, 0, ny],
                                       xlim=data.view_xlim if data else None,
                                       ylim=data.view_ylim if data else None)
            
            # Update Hist using downsampled data for speed
            arr = display_entry['display_data'].ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                # Use a deterministic subset for histogram stability
                sample_limit = 100000
                if arr.size > sample_limit:
                    step = arr.size // sample_limit
                    sample = arr[::step]
                else:
                    sample = arr
                
                # Update markers to match currently used limits
                self.view.canvas_hist.set_limits(use_vmin, use_vmax)
                
                counts, bins = np.histogram(sample, bins=1000)
                self.view.canvas_hist.show_hist(counts, bins)
            
            if data and data.selected_source and (data.rp_profile is None or data.snrs is None):
                xy = data.selected_source
                r = data.aperture_radius
                rp_rad, rp_prof = pc.compute_radial_profile(self.current_img, xy, max_radius=max(30, r*3))
                # Same inputs measure_star uses: this frame's own camera mode,
                # the FWHM box (it sets the radius grid) and the user's sky
                # annulus. This path used High noise, fwhm 12 and 2r-3r, so after
                # a Fit FWHM change a Low frame showed a curve and an r_best that
                # did not belong to the measurement beside it.
                f_gain, f_rn, f_dn, _ = self._frame_camera_params(index)
                radii, snrs, r_best, _ = pc.compute_snr_vs_radius(
                    self.current_img, xy, fwhm=self.model.fwhm,
                    gain=f_gain, read_noise=f_rn, dark_noise=f_dn,
                    inner_coef=self.model.inner_coef, outer_coef=self.model.outer_coef)
                self.model.update_frame_data(index, 
                    rp_radii=rp_rad, rp_profile=rp_prof,
                    radii=radii, snrs=snrs, r_best=r_best
                )

            self.show_current_data()
        except Exception as e:
            print(f"Load error: {e}")

    def show_current_data(self):
        data = self.model.get_current_data()
        if not data: return
        
        ap = data.aperture_result or {}
        notes = []
        if ap.get('aperture_correction_status') == 'unavailable':
            notes.append("no aperture correction: too few clean stars")
        if ap.get('saturation') in ('nonlinear', 'saturated'):
            notes.append(self._saturation_text(ap, short=True))   # the panel is narrow
        if ap.get('psf_check') == 'broad':
            notes.append(f"PSF BROAD, {ap['star_fwhm']:.1f} px")
        zp_val = self.model.zeropoints.get(data.fname, self.model.zeropoint)
        self.view.set_results(data.mag, data.mag_err, data.flux, data.snr,
                              mag_note=("; ".join(notes) or None), zeropoint=zp_val)
        self.view.set_bad_frame(data.is_bad)
        self.view.update_params(self.model)
        
        # Explicitly update Exptime spinbox with current frame's exptime if available
        # Priority: 1) Current frame's per-frame exptime, 2) Global override, 3) Do nothing
        self.view.spin_exptime.blockSignals(True)
        if data.exptime is not None and data.exptime > 0:
            # Current frame has its own exposure time from header - use it
            self.view.spin_exptime.setValue(data.exptime)
        elif self.model.exptime_override is not None and self.model.exptime_override > 0:
            # No per-frame exptime, but global override exists - use it
            self.view.spin_exptime.setValue(self.model.exptime_override)
        self.view.spin_exptime.blockSignals(False)

        # Explicitly update ZP spinbox from per-file dict if available
        # This overrides generic model param update for this specific field
        self.view.spin_zp.blockSignals(True)
        self.view.spin_zp.setValue(zp_val)
        self.view.spin_zp.blockSignals(False)
        
        # Update Plots. The lower-right panel shows whichever curve the active
        # aperture rule actually reads its radius from, so what the user sees is
        # the thing the number came from.
        plot_profile_and_snr(self.view.fig_profile, self.view.fig_snr,
                             data.rp_radii, data.rp_profile,
                             data.radii, data.snrs, data.r_best,
                             r_used=data.aperture_radius,
                             frame_fwhm=self._frame_fwhm_cache.get(self.model._current_index))
        if self.view.plot_mode() == "growth":
            cog_r, cog_f = self._growth_curve_for(data)
            plot_growth_curve(self.view.fig_snr, cog_r, cog_f,
                              r_used=data.aperture_radius,
                              target=self.model.growth_fraction)
        self.view.canvas_profile.draw_idle()
        self.view.canvas_snr.draw_idle()
        
        # Update Canvas patches
        self.view.canvas.clear_patches()
        if data.selected_source:
            from matplotlib.patches import Circle
            x, y = data.selected_source
            r = data.aperture_radius
            c = mpl_colors()
            self.view.canvas.add_patch(Circle((x,y), r, color=c["aperture"], fill=False, lw=1.5))
            self.view.canvas.add_patch(Circle((x,y), r*self.model.inner_coef, color=c["annulus"], fill=False, lw=1, ls='--'))
            self.view.canvas.add_patch(Circle((x,y), r*self.model.outer_coef, color=c["annulus"], fill=False, lw=1, ls='--'))

    @QtCore.pyqtSlot()
    def add_files(self):
        if is_worker_busy(self, 'load_worker', view=self.view,
                          busy_msg="Previous image pre-loading is still in progress."):
            return

        paths, _ = QFileDialog.getOpenFileNames(self.view, "Add FITS files", "", "FITS (*.fits *.fit *.fit.gz)")
        if not paths: return

        added = self.model.add_files(paths)
        if not added:
            QMessageBox.information(self.view, "Already Loaded",
                                    "All selected files are already in the list.")
            return

        # 1. Metadata extraction (Fast)
        from astropy.io import fits
        exptimes = []
        for i in added:
            p = self.model.files[i]
            try:
                with fits.open(p) as hdul:
                    hdr = hdul[0].header

                    # Use unified exposure time detection; default=None so a
                    # genuinely missing EXPTIME is detectable below.
                    et = get_exptime_seconds(hdr, default=None)

                    date_ut = hdr.get('DATE')
                    date_obs = f"{hdr.get('DATE-OBS', '')}".replace('NOGPS', '').strip()
                    if not date_obs: date_obs = None

                    self.model.update_frame_data(i, exptime=et, date_ut=date_ut, date_obs=date_obs)
                    if et is not None: exptimes.append(et)
            except Exception as e:
                log.warning("Metadata extraction failed for %s: %s", p, e)

        # Update override if we found something in headers
        if exptimes and self.model.exptime_override is None:
            med_et = float(np.median(exptimes))
            self.model.exptime_override = med_et
            self.view.spin_exptime.blockSignals(True)
            self.view.spin_exptime.setValue(med_et)
            self.view.spin_exptime.blockSignals(False)

        # Frames in THIS batch that still lack an exposure time
        missing = [i for i in added
                   if (fd := self.model._frame_data.get(i)) is not None and fd.exptime is None]

        if missing and self.model.exptime_override is None:
            # No override available at all -> ask the user once.
            val, ok = QtWidgets.QInputDialog.getDouble(self.view, "Exposure Time",
                                                      "Exposure time missing from headers.\nPlease enter exposure time (s):",
                                                      1.0, 0, 1e6, 2)
            if ok:
                self.model.exptime_override = val
                self.view.spin_exptime.blockSignals(True)
                self.view.spin_exptime.setValue(val)
                self.view.spin_exptime.blockSignals(False)

        # Fill any still-missing per-frame exptimes from the override. This
        # also covers LATER batches: previously a second batch without header
        # EXPTIME silently kept exptime=None (no prompt fired because an
        # override from batch 1 already existed), so those frames never got
        # a magnitude.
        if self.model.exptime_override is not None:
            for i in missing:
                fd = self.model._frame_data.get(i)
                if fd and fd.exptime is None:
                    self.model.update_frame_data(i, exptime=self.model.exptime_override)

        # 2. Pre-loading images for fast toggling (Slow, needs Progress)
        self.progress_dlg = QProgressDialog("Pre-loading images for fast display...", "Cancel", 0, len(paths), self.view)
        self.progress_dlg.setWindowModality(QtCore.Qt.WindowModal)
        self.progress_dlg.show()
        
        # Find which indices correlate to the newly added paths
        # Actually easier to just reload all active files in cache if not present
        self.load_worker = InitialLoadWorker(self.model.files, self.image_cache, self.model.display_max_dim)
        self.load_worker.progress.connect(self.progress_dlg.setValue)
        
        def on_load_done():
            self.progress_dlg.close()
            # Ensure current frame is visible
            if self.model._current_index == -1 and len(self.model.files) > 0:
                self.model.set_current_index(0)
            else:
                self.on_selection_changed(self.model._current_index)
                
        self.load_worker.done.connect(on_load_done)
        self.load_worker.start()

    @QtCore.pyqtSlot()
    def remove_files(self):
        rows = [i.row() for i in self.view.list_files.selectedIndexes()]
        if rows:
            self.model.remove_files(rows)

    @QtCore.pyqtSlot(QtCore.QModelIndex, int, int, QtCore.QModelIndex, int)
    def on_rows_moved(self, parent, start, end, destination, row):
        """Sync model when files are reordered in the list view."""
        count = self.view.list_files.count()
        new_indices = []
        old_files = list(self.model.files)
        
        for i in range(count):
            full_path = self.view.list_files.item(i).data(QtCore.Qt.UserRole)
            try:
                old_idx = old_files.index(full_path)
                new_indices.append(old_idx)
            except ValueError:
                pass
        
        if len(new_indices) == len(self.model.files):
            self.model.reorder_files(new_indices)

    @QtCore.pyqtSlot()
    def remove_all(self):
        if QMessageBox.question(self.view, "Clear All", "Are you sure you want to clear all loaded frames and data?", 
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.model.remove_files(list(range(len(self.model.files))))

    @QtCore.pyqtSlot(int)
    def select_file(self, index):
        self.model.set_current_index(index)

    @QtCore.pyqtSlot()
    def go_prev(self):
        if self.model._current_index > 0:
            self.model.set_current_index(self.model._current_index - 1)

    @QtCore.pyqtSlot()
    def go_next(self):
        if self.model._current_index < len(self.model.files) - 1:
            self.model.set_current_index(self.model._current_index + 1)

    @QtCore.pyqtSlot(bool)
    def toggle_bad(self, is_bad):
        self.model.update_frame_data(self.model._current_index, is_bad=is_bad)

    @QtCore.pyqtSlot(float, float)
    def on_canvas_click(self, x, y):
        if self.current_img is None: return

        # Ignore clicks while a previous detection is still running
        # (rebinding det_worker would GC a live QThread -> crash)
        if is_worker_busy(self, 'det_worker'):
            return

        if self.zeropoint_calibration_mode:
            self._handle_calibration_click(x, y)
            return

        # Show worker for detection
        self.progress_dlg = QProgressDialog("Detecting & Measuring Source...", "Cancel", 0, 0, self.view)
        self.progress_dlg.setWindowModality(QtCore.Qt.WindowModal)
        self.progress_dlg.show()
        
        # ALWAYS use the CURRENT spinbox value for photometry
        # This ensures we use whatever exposure time the user is currently viewing/editing
        exptime_to_use = self.view.spin_exptime.value()
        
        # Use THIS frame's detected camera mode, like the bulk worker does.
        idx = self.model._current_index
        f_gain, f_rn, f_dn, f_mode = self._frame_camera_params(idx)

        # This frame's own calibrated zeropoint, exactly as _measure_frame and
        # the bulk worker resolve it. Reading the global model.zeropoint here
        # meant that after Apply Zeropoints, clicking a new star reported a raw
        # instrumental magnitude until some unrelated parameter was nudged.
        zp_to_use = self.model.zeropoint
        if 0 <= idx < len(self.model.files):
            zp_to_use = self.model.zeropoints.get(
                os.path.basename(self.model.files[idx]), self.model.zeropoint)

        # Frame-level PSF quantities come from the cache, so a click costs one
        # measurement per frame rather than one per click.
        need_cog = bool(self.model.aperture_correct)
        f_fwhm, f_cog = self._frame_psf(
            idx, want_cog=need_cog) if (
                need_cog or self.model.aperture_policy == pc.POLICY_FRAME_FWHM
            ) else (None, None)

        # Prepare params for worker
        params = {
            'stamp_size': self.model.stamp_size,
            'fwhm': self.model.fwhm,
            'threshold': self.model.threshold,
            'fixed_ap': self.model.fixed_aperture,
            'aperture': self.model.aperture,
            'ap_policy': self.model.aperture_policy,
            'growth_fraction': self.model.growth_fraction,
            'fwhm_factor': self.model.fwhm_factor,
            'frame_fwhm': f_fwhm,
            'aperture_correct': self.model.aperture_correct,
            'frame_cog': f_cog,
            'inner': self.model.inner_coef,
            'outer': self.model.outer_coef,
            'zp': zp_to_use,
            'exptime': exptime_to_use,
            'gain': f_gain,
            'read_noise': f_rn,
            'dark_noise': f_dn,
            'linear_limit': self._frame_linear_limit(idx),
            'frame_fwhm_spread': self._frame_fwhm_spread_cache.get(idx)
        }

        self.det_worker = DetectionWorker(self.current_img, x, y, params)

        def on_done(found_xy, fit_params, stamp, method, phot_res):
            self.progress_dlg.close()
            if found_xy:
                # Update model with pre-calculated results
                r_used = phot_res['r_used']
                res = phot_res['phot']
                self.model.update_frame_data(self.model._current_index, 
                    selected_source=found_xy,
                    aperture_radius=r_used,
                    radii=phot_res['radii'], snrs=phot_res['snrs'], r_best=phot_res['r_best'],
                    rp_radii=phot_res['rp_rad'], rp_profile=phot_res['rp_prof'],
                    mag=res['mag'], mag_err=res['mag_err'], flux=res['flux'], snr=res['snr'],
                    aperture_result=res,
                    # Record which camera mode this measurement actually used, so
                    # exports carry the same provenance a bulk run would give.
                    camera_mode=f_mode,
                    gain_used=f_gain, read_noise_used=f_rn, dark_noise_used=f_dn
                )
            else:
                msg = "No star detected in target area."
                if method == 'not-found-in-crop':
                    msg = f"No any source detected in localized stamp ({self.model.stamp_size} px)."
                QMessageBox.warning(self.view, "Not Found", msg)
        
        self.det_worker.finished.connect(on_done)
        self.det_worker.start()

    def _frame_psf(self, index, want_cog=False):
        """
        (frame_fwhm, frame_cog) for one frame, measured once and cached.

        Both are properties of the frame, not of the star, so re-deriving them
        every time a spin box moves was pure waste: estimate_frame_fwhm runs a
        star detection plus ~25 radial profiles and takes a second or two, and
        it used to run on EVERY live parameter update.
        """
        img = self.current_img
        if img is None:
            return None, None

        fwhm = self._frame_fwhm_cache.get(index)
        if fwhm is None:
            spread = None
            try:
                fwhm, spread = pc.estimate_frame_fwhm(img, fwhm_guess=self.model.fwhm,
                                                      return_spread=True)
            except Exception as e:
                log.warning("Frame FWHM measurement failed for frame %s: %s", index, e)
                fwhm = None
            self._frame_fwhm_cache[index] = fwhm
            self._frame_fwhm_spread_cache[index] = spread

        cog = None
        if want_cog:
            cog = self._frame_cog_cache.get(index)
            if cog is None:
                try:
                    cog = pc.frame_growth_curve(
                        img, fwhm if fwhm else self.model.fwhm,
                        inner_coef=self.model.inner_coef,
                        outer_coef=self.model.outer_coef)
                except Exception as e:
                    log.warning("Frame growth curve failed for frame %s: %s", index, e)
                    cog = (None, None)
                self._frame_cog_cache[index] = cog
        return fwhm, cog

    def _invalidate_frame_psf(self, index=None):
        """Drop cached PSF measurements (all frames, or just one)."""
        if index is None:
            self._frame_fwhm_cache.clear()
            self._frame_fwhm_spread_cache.clear()
            self._frame_cog_cache.clear()
        else:
            self._frame_fwhm_cache.pop(index, None)
            self._frame_fwhm_spread_cache.pop(index, None)
            self._frame_cog_cache.pop(index, None)

    @QtCore.pyqtSlot()
    def measure_frame_fwhm(self):
        """Measure the current frame's PSF width on demand and show it."""
        if self.current_img is None:
            QMessageBox.information(self.view, "No Frame",
                                    "Load and select a frame first.")
            return
        idx = self.model._current_index
        self._invalidate_frame_psf(idx)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            fwhm, _ = self._frame_psf(idx)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if fwhm is None:
            self.view.set_measured_fwhm(None, "could not measure (too few stars)")
            return
        # The box holds the starting guess the detector uses; seeding it with the
        # measured value makes every later detection start from the right scale.
        self.view.spin_fwhm.blockSignals(True)
        self.view.spin_fwhm.setValue(float(fwhm))
        self.view.spin_fwhm.blockSignals(False)
        self.model.fwhm = float(fwhm)
        self.view.set_measured_fwhm(fwhm, "this frame")

    def _frame_camera_params(self, index):
        """
        Camera noise parameters for one frame: (gain, read_noise, dark_noise, mode_name).

        Resolved exactly the way BulkPhotometryWorker does it -- from the
        frame's own header/filename via detect_mode_or_none, falling back to
        the model's configured values when the mode cannot be identified.
        The interactive paths previously always used the model globals (High),
        so clicking a star on a Low/Merge frame produced a different SNR curve
        and a different SNR-optimised aperture than a bulk run on that same
        frame.
        """
        if 0 <= index < len(self.model.files):
            path = self.model.files[index]
            try:
                mode = detect_mode_or_none(fits.getheader(path), path)
            except Exception as e:
                log.warning("Camera-mode detection failed for %s: %s", path, e)
                mode = None
            if mode is not None:
                # Dark charge = dark current x THIS frame's exposure, as in
                # the bulk worker.
                fd = self.model._frame_data.get(index)
                et = (fd.exptime if fd is not None and fd.exptime
                      else self.model.exptime_override)
                return mode.gain, mode.read_noise, mode.dark_charge(et), mode.name
        return self.model.gain, self.model.read_noise, self.model.dark_noise, None

    def _frame_linear_limit(self, index):
        """The linear limit (calibrated ADU) of this frame's camera mode, or
        None when the mode is unknown or its limit was never measured."""
        if 0 <= index < len(self.model.files):
            path = self.model.files[index]
            try:
                mode = detect_mode_or_none(fits.getheader(path), path)
            except Exception as e:
                log.warning("Camera-mode detection failed for %s: %s", path, e)
                return None
            return mode.linear_limit_adu if mode is not None else None
        return None

    def _measure_frame(self, index, xy, fast_mode=False):
        img = self.current_img
        if img is None: return
        
        data = self.model._frame_data.get(index)

        # Reuse the cached SNR curve when the caller only tweaked a parameter
        # that cannot change it (fast_mode). measure_star recomputes otherwise.
        reuse = None
        if data is not None and data.snrs is not None:
            if fast_mode or self.model.fixed_aperture:
                reuse = (data.radii, data.snrs, data.r_best)

        # Prefer the per-file zeropoint if one was calibrated for this frame
        fname = self.model._frame_data[index].fname
        zp = self.model.zeropoints.get(fname, self.model.zeropoint)

        # Use this frame's own exposure time; fall back to the global override.
        et = data.exptime if (data is not None and data.exptime and data.exptime > 0) \
            else self.model.exptime_override

        f_gain, f_rn, f_dn, f_mode = self._frame_camera_params(index)

        need_cog = bool(self.model.aperture_correct)
        if need_cog or self.model.aperture_policy == pc.POLICY_FRAME_FWHM:
            f_fwhm, f_cog = self._frame_psf(index, want_cog=need_cog)
        else:
            f_fwhm, f_cog = None, None

        # Same shared routine as the click path and the bulk worker.
        m = pc.measure_star(
            img, xy,
            fwhm=self.model.fwhm,
            aperture_override=(self.model.aperture if self.model.fixed_aperture else None),
            aperture_policy=self.model.aperture_policy,
            growth_fraction=self.model.growth_fraction,
            fwhm_factor=self.model.fwhm_factor,
            frame_fwhm=f_fwhm,
            aperture_correct=self.model.aperture_correct,
            frame_cog=f_cog,
            inner_coef=self.model.inner_coef, outer_coef=self.model.outer_coef,
            gain=f_gain, read_noise=f_rn, dark_noise=f_dn,
            zeropoint=zp, exptime=et,
            snr_curve=reuse,
            linear_limit=self._frame_linear_limit(index),
            frame_fwhm_spread=self._frame_fwhm_spread_cache.get(index))

        res = m['phot']
        self.model.update_frame_data(index,
            selected_source=xy,
            aperture_radius=m['r_used'],
            radii=m['radii'], snrs=m['snrs'], r_best=m['r_best'],
            rp_radii=m['rp_radii'], rp_profile=m['rp_profile'],
            mag=res['mag'], mag_err=res['mag_err'], flux=res['flux'], snr=res['snr'],
            aperture_result=res,
            camera_mode=f_mode,
            gain_used=f_gain, read_noise_used=f_rn, dark_noise_used=f_dn
        )

    @QtCore.pyqtSlot(object, object)
    def on_view_changed(self, xlim, ylim):
        """Handle zoom change or contrast change (from histogram)."""
        if self.current_img is None: return
        
        data = self.model.get_current_data()
        if data:
            if xlim == "zscale" and ylim == "zscale":
                # Reset to cached ZScale values for performance
                path = self.model.files[self.model._current_index]
                display_entry = self.image_cache.get_display_data(path, self.current_img)
                vmin, vmax = display_entry['vmin'], display_entry['vmax']
                
                self.view.canvas_hist.set_limits(vmin, vmax)
                self.view.canvas.set_clim(vmin, vmax)
                
                if data:
                    data.vmin, data.vmax = vmin, vmax
                    self.model.trigger_update()
                return

            if xlim is not None and ylim is not None:
                # Zoom change
                data.view_xlim = xlim
                data.view_ylim = ylim
            else:
                # Contrast change from interactive histogram
                vmin, vmax = self.view.canvas_hist.low_val, self.view.canvas_hist.high_val
                self.view.canvas.set_clim(vmin, vmax)
                data.vmin, data.vmax = vmin, vmax
                self.model.trigger_update()

    @QtCore.pyqtSlot(str)
    def update_params_from_view(self, param_name=None):
        # Block signals to prevent recursion
        self.view.blockSignals(True)
        
        # Helper: auto-fix Inner/Outer relationship logic can settle before we read final values
        # (Though with QDoubleSpinBox, we usually get valid values. We can keep the check.)
        inner = self.view.spin_inner.value()
        outer = self.view.spin_outer.value()
        if inner >= outer:
            outer = inner + 0.1
            self.view.spin_outer.setValue(outer)

        # Logic for specific parameter changes
        if param_name == "aperture":
            # If user manually changed aperture box, assume they want it fixed
            self.model.fixed_aperture = True
            self.view.chk_fixed_aperture.setChecked(True)
        elif param_name == "fixed_aperture":
            pass # Checkbox toggled, just update model below
            
        # Update model parameters
        elif param_name == "exptime":
            # If user manually changes exposure time spinbox, save it ONLY to current frame's per-frame data
            # Do NOT update global exptime_override (that would affect all frames)
            # Each frame maintains its own exposure value independently
            exptime_val = self.view.spin_exptime.value()
            current_data = self.model.get_current_data()
            if current_data:
                self.model.update_frame_data(self.model._current_index, exptime=exptime_val)

        # Update model parameters
        self.model.aperture = float(self.view.spin_aperture.value())
        self.model.fixed_aperture = self.view.chk_fixed_aperture.isChecked()
        self.model.aperture_policy = self.view.aperture_policy()
        if self.model.aperture_policy == "frame_fwhm":
            self.model.fwhm_factor = float(self.view.spin_ap_param.value())
        elif self.model.aperture_policy == "growth":
            self.model.growth_fraction = float(self.view.spin_ap_param.value())
        self.model.aperture_correct = self.view.chk_ap_corr.isChecked()
        self.model.stamp_size = int(self.view.spin_stamp.value())
        prev_fwhm = self.model.fwhm
        self.model.fwhm = self.view.spin_fwhm.value()
        self.model.threshold = self.view.spin_thresh.value()
        self.model.inner_coef = self.view.spin_inner.value()
        self.model.outer_coef = self.view.spin_outer.value()
        # model.zeropoint is the DEFAULT for frames with no per-file value, so it
        # is only written when the user actually edits the zeropoint box. It used
        # to be assigned here on every parameter change, and show_current_data
        # puts the viewed frame's own zeropoint in that box -- so changing the
        # aperture while looking at a calibrated frame promoted that frame's
        # zeropoint to the default for all the uncalibrated ones.
        # Note: We do NOT blindly read back exptime_override from view here every time, 
        # unless it was the specific parameter changed, because we want to allow per-frame viewing.
        if param_name == "exptime":
            self.model.exptime_override = self.view.spin_exptime.value()
        elif param_name == "zeropoint":
             # If user manually changed ZP, update it for THIS file (and globally? or just file?)
             # User expectation: "each frame should have its own filled zeropoint"
             val = self.view.spin_zp.value()
             # Update global default too, just in case
             self.model.zeropoint = val
             # Update specific file overrides
             if self.model._current_index >= 0 and self.model._current_index < len(self.model.files):
                  # BASENAME: this used to store the full path while every
                  # reader looked the value up by basename, so a zeropoint
                  # typed here was written to a key nothing could ever find.
                  fname = os.path.basename(self.model.files[self.model._current_index])
                  self.model.set_zeropoint_for_file(fname, val)
        
        self.view.blockSignals(False)

        # The Fit FWHM box is the detector's starting guess AND it sets the
        # radius grid the SNR curve is sampled on, so a cached curve measured at
        # the old value is stale: r_best would keep reporting the old optimum.
        # The frame's measured PSF is seeded from the same box, so that goes too.
        if param_name == "fwhm" and self.view.spin_fwhm.value() != prev_fwhm:
            self._invalidate_frame_psf()
            for d in self.model._frame_data.values():
                d.radii = d.snrs = d.r_best = None
            self.view.set_measured_fwhm(None, "")

        # If parameters affecting photometry result changed (ZP or Exptime), update ALL frames with existing results
        if param_name in ["zeropoint", "exptime"]:
            self._update_all_frame_results()

        # Live re-measure, but ONLY for parameters that can change the result.
        # Detection Stamp is used when a star is picked, not when it is measured,
        # so re-measuring on every step of that spin box bought nothing and made
        # it feel like it was lagging -- each step was quietly running a full
        # measurement (and, under the Frame FWHM rule, a whole-frame PSF
        # measurement) behind the UI.
        MEASUREMENT_PARAMS = {
            "aperture", "fixed_aperture", "aperture_policy", "aperture_correct",
            "fwhm", "inner", "outer", "zeropoint", "exptime",
        }
        data = self.model.get_current_data()
        if (param_name in MEASUREMENT_PARAMS
                and data and data.selected_source):
            # A stale SNR curve must not be reused after the grid changed.
            self._measure_frame(self.model._current_index, data.selected_source,
                                fast_mode=(param_name != "fwhm"))
            self.show_current_data() # Update UI
        else:
            # If no current source, still trigger update so table refreshes if we updated all frames
            self.model.trigger_update()

    def _update_all_frame_results(self):
        """Recompute every measured frame's magnitude after a ZP/exptime change.

        Both halves of this used to be wrong. It applied the GLOBAL zeropoint to
        every frame, never consulting model.zeropoints, so it wiped per-file
        calibration -- and it fired after every bulk run, because on_bulk_done
        set the exposure box without blocking signals. It also rebuilt the
        instrumental magnitude from the raw aperture flux, which does not carry
        the frame's aperture correction, so that quietly vanished too.

        Both now come from MultiPhotometryModel.magnitude_for, the one place
        the formula lives.
        """
        for idx, data in self.model._frame_data.items():
            if data.aperture_result:
                self.model.update_frame_data(
                    idx, mag=self.model.magnitude_for(data))

    def _resolve_exptimes(self, prompt=True):
        """
        Per-frame exposure times for a worker run: (exptime_map, ok).

        Each frame still gets a FRESH header read inside the worker, so frames
        with differing exposures keep their own value. What this adds is a
        per-frame fallback map for frames whose header has no usable EXPTIME:
        those values were already resolved when the files were added (from the
        header, or from the user if they were asked then).

        Both the bulk run and the zeropoint calibration run go through here.
        They must, because a zeropoint is ZP = m_cat - m_instr and m_instr
        carries -2.5*log10(exptime): calibration used to force ONE global
        exposure time onto every frame while the bulk run read each header, so
        frames whose exposure differed from the median came out offset by
        2.5*log10(et_bulk/et_calib) -- 34 mmag for 116.3 s against 120 s. The
        reference star then failed to reproduce its own catalogue magnitude.
        """
        exptime_map = {}
        for i, path in enumerate(self.model.files):
            fd = self.model._frame_data.get(i)
            if fd is not None and fd.exptime is not None and fd.exptime > 0:
                exptime_map[os.path.basename(path)] = float(fd.exptime)

        # A frame only needs prompting if the header has no usable EXPTIME AND
        # we have no previously-resolved value for it.
        missing = []
        for i, path in enumerate(self.model.files):
            if os.path.basename(path) in exptime_map:
                continue
            try:
                if get_exptime_seconds(fits.getheader(path), default=None) is None:
                    missing.append(path)
            except Exception as e:
                log.warning("EXPTIME pre-check failed for %s: %s", path, e)
                missing.append(path)

        if missing and prompt:
            # Fall back to whatever the user already set globally, if anything.
            default_val = (self.model.exptime_override
                           if (self.model.exptime_override or 0) > 0 else 1.0)
            val, ok = QtWidgets.QInputDialog.getDouble(
                self.view, "Exposure Time",
                f"{len(missing)} frame(s) have no EXPTIME in their header.\n"
                "Enter exposure time (s) to use for them:",
                float(default_val), 0.001, 1e5, 3)
            if not ok:
                return exptime_map, False
            # Record it so this is not asked again for the same frames.
            self.model.exptime_override = val
            for i, path in enumerate(self.model.files):
                if path in missing:
                    self.model.update_frame_data(i, exptime=val)
                    exptime_map[os.path.basename(path)] = float(val)
        return exptime_map, True

    @QtCore.pyqtSlot()
    def start_bulk(self):
        if is_worker_busy(self, 'worker', view=self.view,
                          busy_msg="A bulk photometry or calibration run is already in progress."):
            return
        data = self.model.get_current_data()
        if not data or not data.selected_source:
            QMessageBox.warning(self.view, "No Source", "Select a star in the current frame first.")
            return

        exptime_final = None
        exptime_map, ok = self._resolve_exptimes()
        if not ok:
            return

        # Start worker with fresh header reading enabled (exptime_override=None)
        self.worker = BulkPhotometryWorker(
            self.model.files, self.model._current_index, data.selected_source,
            fwhm=self.model.fwhm, threshold_sigma=self.model.threshold,
            detection_stamp_size=self.model.stamp_size, # Pass the precision stamp
            aperture_override=self.model.aperture if self.model.fixed_aperture else None,
            aperture_policy=self.model.aperture_policy,
            growth_fraction=self.model.growth_fraction,
            fwhm_factor=self.model.fwhm_factor,
            aperture_correct=self.model.aperture_correct,
            inner_coef=self.model.inner_coef,
            outer_coef=self.model.outer_coef,
            gain=self.model.gain,
            read_noise=self.model.read_noise,
            dark_noise=self.model.dark_noise,
            zeropoint=self.model.zeropoint,
            zeropoint_map=self.model.zeropoints, # Sync per-frame ZPs
            exptime_override=exptime_final,  # Only set if user manually provided a value
            exptime_map=exptime_map  # Per-frame fallback when a header lacks EXPTIME
        )
        self.progress_dlg = QProgressDialog("Bulk Photometry...", "Cancel", 0, 100, self.view)
        self.progress_dlg.show() # Show immediately
        self.worker.progress.connect(self.on_bulk_progress)
        self.worker.done.connect(self.on_bulk_done)
        self.progress_dlg.canceled.connect(self.worker.requestAbort)
        
        self.worker.start()

    def on_bulk_progress(self, msg, frac):
        self.progress_dlg.setLabelText(msg)
        self.progress_dlg.setValue(int(frac*100))

    @staticmethod
    def _apcorr_unavailable(results):
        """Frames where the aperture correction was asked for but the frame's
        curve of growth could not be measured, so the magnitude is not total."""
        return [r['file'] for r in results if r.get('success')
                and (r.get('aperture_result') or {}).get(
                    'aperture_correction_status') == 'unavailable']

    @staticmethod
    def _apcorr_unavailable_text(files):
        listed = "\n".join(f"  • {f}" for f in files[:8])
        more = f"\n  ... and {len(files) - 8} more" if len(files) > 8 else ""
        return (f"Aperture correction could not be measured on {len(files)} frame(s) "
                "(fewer than 4 bright, isolated stars with a clean curve of growth). "
                "Their magnitudes are aperture magnitudes, NOT total magnitudes:\n"
                + listed + more)

    @staticmethod
    def _saturation_text(ap, short=False):
        peak = ap.get('peak_adu')
        peak_s = f"peak {peak:.0f} ADU" if peak is not None else "peak"
        if ap.get('saturation') == 'saturated':
            return f"SATURATED, {peak_s}" if short else f"SATURATED: {peak_s} at the 12-bit ceiling"
        return f"NON-LINEAR, {peak_s}" if short else f"NON-LINEAR: {peak_s} above the linear limit"

    def on_bulk_done(self, results):
        self.progress_dlg.close()
        # Results is list of dicts
        exptimes = []
        successful = 0
        failed = 0
        failed_frames = []
        
        for r in results:
            idx = r['index']
            if r['success']:
                successful += 1
                # The worker measured this frame's PSF; keep it so revisiting the
                # frame neither re-measures it nor loses the SNR plot's k label.
                if r.get('frame_fwhm'):
                    self._frame_fwhm_cache[idx] = r['frame_fwhm']
                    self._frame_fwhm_spread_cache[idx] = r.get('frame_fwhm_spread')
                self.model.update_frame_data(idx,
                    selected_source=r['picked_full'],
                    aperture_radius=r['aperture_used'],
                    radii=np.array(r['radii']), snrs=np.array(r['snrs']), r_best=r['r_best'],
                    # Store the worker-computed radial profile too. Without it
                    # rp_profile stays None for every frame and on_selection_changed
                    # recomputes profile+SNR on the GUI thread the first time you
                    # visit each frame -- the post-bulk switching lag.
                    rp_radii=(np.array(r['rp_radii']) if r.get('rp_radii') is not None else None),
                    rp_profile=(np.array(r['rp_profile']) if r.get('rp_profile') is not None else None),
                    mag=r['mag'], mag_err=r['mag_err'], flux=r['flux'], snr=r['snr'],
                    aperture_result=r['aperture_result'],
                    exptime=r['exptime'],
                    # camera parameters this frame's noise model actually used
                    camera_mode=r.get('camera_mode'),
                    gain_used=r.get('gain_used'),
                    read_noise_used=r.get('read_noise_used'),
                    dark_noise_used=r.get('dark_noise_used')
                )
                if r['exptime']: exptimes.append(r['exptime'])
            else:
                failed += 1
                failed_frames.append((r['file'], r['msg']))
        
        if exptimes:
            med_et = float(np.median(exptimes))
            # Blocked: this emits param_changed("exptime") otherwise, which ran
            # _update_all_frame_results at the end of EVERY bulk run.
            self.view.spin_exptime.blockSignals(True)
            self.view.spin_exptime.setValue(med_et)
            self.view.spin_exptime.blockSignals(False)
            self.model.exptime_override = med_et
        
        # Build user-friendly summary message
        summary = f"Bulk Photometry Complete.\n\nSuccessful: {successful}\nFailed: {failed}"

        # Only 1x1 and 2x2 have measured gain/read noise. Frames binned any
        # other way were measured with the configured values, which are the
        # unbinned High ones unless changed -- say so rather than let error
        # bars that may be off by a large factor pass unremarked.
        odd_bins = sorted({tuple(r['binning']) for r in results
                           if r.get('success') and r.get('binning')
                           and tuple(r['binning']) not in ((1, 1), (2, 2))})
        if odd_bins:
            summary += (
                "\n\nNoise model not calibrated for "
                + ", ".join(f"{x}x{y}" for x, y in odd_bins)
                + f" binning: those frames used gain {self.model.gain:g} e-/ADU and "
                f"read noise {self.model.read_noise:g} e-. Their magnitudes are fine; "
                "their error bars and SNR may be wrong.")

        no_apcorr = self._apcorr_unavailable(results)
        if no_apcorr:
            summary += "\n\n" + self._apcorr_unavailable_text(no_apcorr)

        # Frames whose field could not be matched were re-centred where the
        # star was in the previous frame: right for a small drift, but not
        # checked against the field, so they are named.
        unmatched = [r['file'] for r in results if r.get('success')
                     and str(r.get('method', '')).startswith(LAST_POSITION)]
        if unmatched:
            # no-break space: a long name must not wrap away from its bullet
            listed = "\n".join(f"  • {f}" for f in unmatched[:8])
            more = f"\n  ... and {len(unmatched) - 8} more" if len(unmatched) > 8 else ""
            summary += (f"\n\nThe field could not be matched on {len(unmatched)} frame(s); "
                        "the star was re-centred at its position in the previous frame. "
                        "Check these frames:\n" + listed + more)

        # Stars past the linear range read too faint, with an error bar that
        # still looks excellent (a clipped PSR star: SNR 1507, 0.7 mmag).
        flagged = [(r['file'], r['aperture_result']) for r in results if r.get('success')
                   and (r.get('aperture_result') or {}).get('saturation') in ('nonlinear', 'saturated')]
        if flagged:
            listed = "\n".join(f"  •\u00a0{f}: {self._saturation_text(ap)}" for f, ap in flagged[:8])
            more = f"\n  ... and {len(flagged) - 8} more" if len(flagged) > 8 else ""
            summary += (f"\n\nThe star is beyond the linear range on {len(flagged)} frame(s); "
                        "its magnitude there is too faint and its error bar too small:\n"
                        + listed + more)

        # Under the frame-FWHM rule the radius fits the frame's PSF; a star
        # wider than that (blend, galaxy, flattened core, broader PSF there)
        # loses more light outside it and reads too faint.
        broad = [(r['file'], r['aperture_result'].get('star_fwhm'), r.get('frame_fwhm'))
                 for r in results if r.get('success')
                 and (r.get('aperture_result') or {}).get('psf_check') == 'broad']
        if broad:
            listed = "\n".join(f"  • {f}: star {s:.1f} px, frame {fw:.1f} px"
                               for f, s, fw in broad[:8])
            more = f"\n  ... and {len(broad) - 8} more" if len(broad) > 8 else ""
            summary += (f"\n\nThe star is wider than the frame's PSF on {len(broad)} frame(s) "
                        "(a blend, a galaxy, a flattened core or a broader PSF there); the "
                        "frame-FWHM radius loses more of its light, so it reads too faint:\n"
                        + listed + more)

        if failed_frames or odd_bins or no_apcorr or unmatched or flagged or broad:
            # Show detailed error information
            detail_msg = ""
            if failed_frames:
                detail_msg = "\n\nFailed Frames:\n" + "\n".join([f"  • {fname}: {msg}" for fname, msg in failed_frames[:10]])
            if len(failed_frames) > 10:
                detail_msg += f"\n  ... and {len(failed_frames) - 10} more"

            QMessageBox.warning(self.view, "Bulk Photometry Results", f"{summary}{detail_msg}")
        else:
            QMessageBox.information(self.view, "Bulk Photometry Results", summary)
        
        self.model.trigger_update()

    @QtCore.pyqtSlot()
    def calc_extinction(self):
        """Extinction coefficient fit. All science lives in core.ext_coeff
        (time parsing, airmass, weighted fit); this method only orchestrates
        UI in/out."""
        from core import ext_coeff

        rows = self.model.get_photometry_table()
        if len(rows) < 2:
            QMessageBox.warning(self.view, "Insufficient Data", "Need at least 2 photometry points in the table.")
            return

        dlg = ExtinctionDialog(self.view)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            vals = dlg.values()
            ra_txt = vals['ra']; dec_txt = vals['dec']; filter_name = vals['flt'] or "unknown"
            lat = vals['lat']; lon = vals['lon']

            # Rows with a valid magnitude
            valid_rows = [r for r in rows if r.mag is not None and np.isfinite(r.mag)]
            if len(valid_rows) < 2:
                QMessageBox.warning(self.view, "Insufficient Data", "Fewer than 2 frames have a valid magnitude.")
                return

            per_frame, errors = ext_coeff.compute_airmass_for_files(
                self.model.files, [r.index for r in valid_rows],
                ra_txt, dec_txt, lat, lon, elevation_m=vals['elev'])

            if not per_frame:
                QMessageBox.critical(self.view, "Airmass Error",
                                     "Could not compute airmass for any frame:\n" + "\n".join(errors[:8]))
                return

            # Join airmass results with magnitudes by frame index
            row_by_idx = {r.index: r for r in valid_rows}
            mags, mag_errs, airmass, fnames = [], [], [], []
            # A frame without a usable error bar is left out and counted. It
            # used to be given a made-up 0.1 mag, which then set its weight in
            # the fit. 9148adb meant to drop these, but only its success message
            # changed, so every fit that succeeded died there on an undefined
            # n_no_err and the user never saw k or the plot.
            n_no_err = 0
            for info in per_frame:
                r = row_by_idx.get(info['index'])
                if r is None or not np.isfinite(info['airmass']):
                    continue
                info['mag'] = float(r.mag)
                if r.mag_err is None or not np.isfinite(r.mag_err) or r.mag_err <= 0:
                    n_no_err += 1
                    info['excluded'] = 'no error bar'
                    continue
                info['mag_err'] = float(r.mag_err)
                mags.append(info['mag']); mag_errs.append(info['mag_err'])
                airmass.append(info['airmass']); fnames.append(r.filename)

            if len(mags) < 2:
                QMessageBox.warning(self.view, "No Time Info",
                                    "Missing/unparsable DATE-OBS in headers for selected frames.\n"
                                    + (f"{n_no_err} frame(s) have no error bar.\n" if n_no_err else "")
                                    + "\n".join(errors[:8]))
                return

            try:
                fit = ext_coeff.fit_mag_vs_airmass(mags, mag_errs, airmass)
            except ValueError as e:
                QMessageBox.critical(self.view, "Fit Error", str(e))
                return

            k = fit['slope']; intercept = fit['intercept']
            k_err = fit['slope_err']

            # Save JSON via the core module
            outdir = os.path.join(get_output_dir(), "k_results")
            os.makedirs(outdir, exist_ok=True)
            # datetime.utcnow() is deprecated in Python 3.12; same instant, UT.
            _now_ut = datetime.now(timezone.utc)
            outpath = os.path.join(outdir, f"extinction_k_{_now_ut.strftime('%Y%m%d_%H%M%S')}.json")
            res = ext_coeff.build_results_json(ra_txt, dec_txt, filter_name, lat, lon, per_frame, fit)
            res['observatory_elevation_m'] = float(vals['elev'])
            res['created_at'] = _now_ut.isoformat()
            ext_coeff.save_json(outpath, res)

            if errors:
                QMessageBox.warning(self.view, "Some Frames Skipped",
                                    f"{len(errors)} frame(s) were skipped:\n" + "\n".join(errors[:8]))

            if not (np.isfinite(k) and np.isfinite(k_err)):
                QMessageBox.critical(
                    self.view, "Extinction Fit Failed",
                    "The fit did not converge (k or its uncertainty is not a "
                    f"number).\n\n{len(mags)} frames over airmass "
                    f"{min(airmass):.3f} to {max(airmass):.3f}.\n"
                    "A range this narrow, or too few frames, cannot constrain "
                    "a slope.\n\nResults were still written to:\n" + outpath)
                return

            # Scatter beyond the error bars (chi2_nu > 1) means the plain error
            # trusts the error bars too much; show the scaled one beside it.
            chi2_nu = fit.get('chi2_nu')
            fit_line = ""
            if chi2_nu is not None:
                fit_line = f"chi2_nu = {chi2_nu:.2f}"
                if chi2_nu > 1:
                    fit_line += (f": points scatter more than their error bars; "
                                 f"scaled k error \u00b1 {fit['slope_err_scaled']:.5f}")
                fit_line += "\n"
            QMessageBox.information(
                self.view, "Success",
                f"k = {k:.5f} \u00b1 {k_err:.5f}\n"
                + fit_line +
                f"{len(mags)} frames, airmass {min(airmass):.3f}-{max(airmass):.3f}\n"
                "Times: UT, mid-exposure\n"
                + (f"{n_no_err} frame(s) excluded for having no error bar\n" if n_no_err else "")
                + f"Saved to {outpath}")

            # Advanced Plot
            p_dlg = ExtinctionPlotDialog(self.view, np.array(airmass), np.array(mags),
                                         np.array(mag_errs), k, intercept, k_err,
                                         chi2_nu=chi2_nu, k_err_scaled=fit.get('slope_err_scaled'))
            p_dlg.exec_()

    @QtCore.pyqtSlot()
    def _aperture_policy_label(self):
        """Short human-readable name of the aperture rule currently in force."""
        if self.model.fixed_aperture:
            label = f"fixed:{self.model.aperture:g}px"
        else:
            p = self.model.aperture_policy
            if p == "frame_fwhm":
                label = f"frame_fwhm:x{self.model.fwhm_factor:g}"
            elif p == "growth":
                label = f"growth:E{self.model.growth_fraction:g}"
            else:
                label = "snr"
        # A zeropoint built from total magnitudes and one built from
        # inside-the-aperture magnitudes differ by the aperture correction, so
        # the two are not interchangeable and the label has to say which it is.
        if self.model.aperture_correct:
            label += "+apcorr"
        return label

    def manage_zeropoints(self):
        from views.components.zeropoint_manager_dialog import ZeropointManagerDialog
        dlg = ZeropointManagerDialog(self.view, self.model.files,
                                     self.model.zeropoints, self.model.zeropoint_meta)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        zps, meta, cleared_all = dlg.results()
        self.model.clear_zeropoints()
        for fname, zp in zps.items():
            self.model.set_zeropoint_for_file(fname, zp, meta=meta.get(fname))
        # Magnitudes already computed used the old zeropoints, so re-derive them.
        self._update_all_frame_results()
        self.show_current_data()
        if cleared_all and not zps:
            QMessageBox.information(self.view, "Zeropoints cleared",
                                    "All zeropoints removed; magnitudes are instrumental again.")
        else:
            QMessageBox.information(self.view, "Zeropoints updated",
                                    f"{len(zps)} frames now carry a zeropoint.")

    @QtCore.pyqtSlot()
    def review_table(self):
        rows = self.model.get_photometry_table()
        # Mix in the phot_table from model?
        dlg = PhotometryTableDialog(rows, self.view)
        dlg.manage_zeropoints_clicked.connect(self.manage_zeropoints)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            # Sync back removals
            # We compare the original lists or simplest check ids
            current_indices = {r.index for r in dlg.updated_rows}
            original_indices = {r.index for r in rows}
            removed = original_indices - current_indices
            if removed:
                self.model.remove_results(list(removed))

    @QtCore.pyqtSlot()
    def save_profile(self):
        path, _ = QFileDialog.getSaveFileName(self.view, "Save Radial Profile", "", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            try:
                # Black on white regardless of the screen theme, so an export
                # made during a night session is still usable in the paper.
                save_publication_figure(self.view.fig_profile, path, dpi=300)
            except PermissionError:
                QMessageBox.warning(self.view, "Cannot save graph",
                    f"Permission denied writing:\n{path}\n\n"
                    "The file is probably open in another program. "
                    "Close it and try again, or choose a different name.")
                return
            except OSError as e:
                QMessageBox.warning(self.view, "Cannot save graph",
                    f"Could not save the graph:\n{path}\n\n{e}")
                return
            QMessageBox.information(self.view, "Saved", f"Profile graph saved to {path}")

    @QtCore.pyqtSlot()
    @QtCore.pyqtSlot()
    def show_lightcurve(self):
        rows = self.model.get_photometry_table()
        if not rows:
            QMessageBox.information(self.view, "No Data", "Add some photometry results first.")
            return
        dlg = AdvancedLightCurveDialog(self.view, rows)
        dlg.exec_()

    # --- Zeropoint Calibration Logic ---
    @QtCore.pyqtSlot()
    def _growth_curve_for(self, data):
        """
        Curve of growth for the star currently selected in `data`.

        Measured on demand and cached on the frame's data, because it is only
        needed when the growth curve is actually being displayed and it costs a
        second pass over the star's neighbourhood.
        """
        if getattr(data, "_cog", None) is not None:
            return data._cog
        if not data.selected_source or self.current_img is None:
            return None, None
        try:
            r, f = pc.curve_of_growth(self.current_img, data.selected_source,
                                      fwhm=self.model.fwhm,
                                      inner_coef=self.model.inner_coef,
                                      outer_coef=self.model.outer_coef)
        except Exception:
            return None, None
        try:
            data._cog = (r, f)
        except Exception:
            pass
        return r, f

    @QtCore.pyqtSlot()
    def on_plot_mode_changed(self):
        self.show_current_data()

    @QtCore.pyqtSlot()
    def on_all_plots_clicked(self):
        data = self.model.get_current_data()
        if not data or not data.selected_source:
            QMessageBox.information(self.view, "No star selected",
                                    "Click a star first, then open the radial diagnostics.")
            return
        from views.components.radial_plots_dialog import RadialPlotsDialog
        cog_r, cog_f = self._growth_curve_for(data)
        labels = {"snr": "Max SNR (per star)",
                  "growth": f"Curve of growth (E={self.model.growth_fraction:g})",
                  "frame_fwhm": f"Frame FWHM x {self.model.fwhm_factor:g}"}
        policy_label = ("Fixed aperture" if self.model.fixed_aperture
                        else labels.get(self.model.aperture_policy, self.model.aperture_policy))
        dlg = RadialPlotsDialog(
            self.view,
            dict(rp_radii=data.rp_radii, rp_profile=data.rp_profile,
                 radii=data.radii, snrs=data.snrs, r_best=data.r_best,
                 r_used=data.aperture_radius, cog_radii=cog_r, cog_frac=cog_f,
                 growth_target=self.model.growth_fraction,
                 frame_fwhm=self._frame_fwhm_cache.get(self.model._current_index),
                 frame=os.path.basename(data.fname), star_xy=data.selected_source),
            title_suffix=f"- {os.path.basename(data.fname)}",
            policy_label=policy_label)
        dlg.exec_()

    @QtCore.pyqtSlot()
    def calibrate_zeropoints(self):
        if not self.model.files:
            QMessageBox.warning(self.view, "No Files", "Load files first.")
            return

        # A zeropoint moves to another star only if both are measured through
        # the same radius; a per-star rule gives each star its own, and the
        # aperture loss differs between them (r = 20 vs 27 px was 62 mmag).
        if (not self.model.fixed_aperture
                and self.model.aperture_policy in (pc.POLICY_SNR, pc.POLICY_GROWTH)):
            reply = QMessageBox.warning(
                self.view, "Per-star radius rule",
                f"The radius rule is \"{self.view.cmb_ap_policy.currentText()}\": every "
                "star gets its own radius, so a zeropoint measured on the reference "
                "star does not carry over to a star measured through a different "
                "radius.\n\nUse \"Frame FWHM x factor\" or a fixed aperture for "
                "zeropoints. Continue anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        # Clear previous session history
        self.calibration_history = []
        
        reply = QMessageBox.information(self.view, "Calibration Mode", 
                                       "Please click on a reference star in the current frame, then click Confirm.\n"
                                       "This star will be used to calculate zeropoints for all frames.",
                                       QMessageBox.Ok | QMessageBox.Cancel)
        if reply == QMessageBox.Cancel: return
        
        self.zeropoint_calibration_mode = True
        self._setup_calibration_ui()

    def _setup_calibration_ui(self):
        if self.confirm_cancel_buttons is None:
            self.confirm_cancel_buttons = QtWidgets.QWidget()
            l = QtWidgets.QHBoxLayout(self.confirm_cancel_buttons)
            self.btn_confirm_ref = QtWidgets.QPushButton("Confirm Reference Star")
            self.btn_cancel_ref = QtWidgets.QPushButton("Cancel Calibration")
            set_variant(self.btn_confirm_ref, "success")
            set_variant(self.btn_cancel_ref, "danger")
            l.addWidget(self.btn_confirm_ref)
            l.addWidget(self.btn_cancel_ref)
            
            # Insert calibration bar into center layout
            if hasattr(self.view, 'center_layout'):
                self.view.center_layout.insertWidget(0, self.confirm_cancel_buttons)
            
            self.btn_confirm_ref.clicked.connect(self._on_confirm_ref)
            self.btn_cancel_ref.clicked.connect(self._on_cancel_calib)
        
        self.confirm_cancel_buttons.show()

    def _handle_calibration_click(self, x, y):
        if is_worker_busy(self, 'det_worker'):
            return
        self.progress_dlg = QProgressDialog("Detecting Calibration Star...", "Cancel", 0, 0, self.view)
        self.progress_dlg.setWindowModality(QtCore.Qt.WindowModal)
        self.progress_dlg.show()
        
        cal_gain, cal_rn, cal_dn, _ = self._frame_camera_params(self.model._current_index)
        params = {
            'stamp_size': self.model.stamp_size,
            'fwhm': self.model.fwhm,
            'threshold': self.model.threshold,
            'fixed_ap': self.model.fixed_aperture,
            'aperture': self.model.aperture,
            'ap_policy': self.model.aperture_policy,
            'growth_fraction': self.model.growth_fraction,
            'fwhm_factor': self.model.fwhm_factor,
            'inner': self.model.inner_coef,
            'outer': self.model.outer_coef,
            'zp': self.model.zeropoint,
            'exptime': self.model.exptime_override,
            'gain': cal_gain,
            'read_noise': cal_rn,
            'dark_noise': cal_dn
        }
        self.det_worker = DetectionWorker(self.current_img, x, y, params)
        
        def on_done(found_xy, fit_params, stamp, method, phot_res):
            self.progress_dlg.close()
            if found_xy:
                self.calibration_ref_xy = found_xy
                self.view.canvas.clear_patches()
                from matplotlib.patches import Circle
                self.view.canvas.add_patch(Circle(found_xy, 15, color='red', fill=False, lw=2))
            else:
                msg = "Could not find a source there."
                if method == 'not-found-in-crop':
                    msg = "No any source detected in localized stamp."
                QMessageBox.warning(self.view, "Not Found", msg)
        
        self.det_worker.finished.connect(on_done)
        self.det_worker.start()

    def _on_confirm_ref(self):
        if not self.calibration_ref_xy:
            QMessageBox.warning(self.view, "No Star", "Please click on a reference star first.")
            return
        
        mag, ok = QtWidgets.QInputDialog.getDouble(self.view, "Reference Mag", "Enter apparent magnitude of this star:", 0.0, -50, 50, 4)
        if not ok: return
        
        self.confirm_cancel_buttons.hide()
        self.zeropoint_calibration_mode = False
        
        # Start a bulk photometry run for this reference star
        self._run_calibration_bulk(self.calibration_ref_xy, mag)

    def _on_cancel_calib(self):
        self.confirm_cancel_buttons.hide()
        self.zeropoint_calibration_mode = False
        self.view.canvas.clear_patches()

    def _run_calibration_bulk(self, xy, ref_mag, replace_index=None):
        if is_worker_busy(self, 'worker', view=self.view,
                          busy_msg="A bulk photometry or calibration run is already in progress."):
            return
        # We reuse BulkPhotometryWorker, and it must be handed the SAME settings
        # a normal bulk run gets. A zeropoint is only transferable to another
        # star if both were measured the same way, so every knob in Advanced
        # Settings -- radius rule, annulus coefficients, FWHM, threshold,
        # stamps, and above all the exposure times -- has to reach this run too.
        # (The annulus coefficients used to be omitted here, so a user who set
        # 2.5/4.0 still had their zeropoints measured through the default
        # 2.0/3.0 sky annulus.)
        exptime_map, ok = self._resolve_exptimes()
        if not ok:
            return
        self.worker = BulkPhotometryWorker(
            self.model.files, self.model._current_index, xy,
            fwhm=self.model.fwhm, threshold_sigma=self.model.threshold,
            detection_stamp_size=self.model.stamp_size,
            aperture_override=self.model.aperture if self.model.fixed_aperture else None,
            aperture_policy=self.model.aperture_policy,
            growth_fraction=self.model.growth_fraction,
            fwhm_factor=self.model.fwhm_factor,
            aperture_correct=self.model.aperture_correct,
            inner_coef=self.model.inner_coef,
            outer_coef=self.model.outer_coef,
            # Same exposure-time resolution as start_bulk: per-frame headers
            # first, the resolved map as fallback, never one global value.
            exptime_override=None,
            exptime_map=exptime_map,
            gain=self.model.gain,
            read_noise=self.model.read_noise,
            dark_noise=self.model.dark_noise,
            zeropoint=0.0, # We want instrumental mag for calibration
            zeropoint_map={} # Clear map for calibration run
        )
        self.progress_dlg = QProgressDialog("Calibrating Zeropoints...", "Cancel", 0, 100, self.view)
        self.progress_dlg.show() # Show immediately
        self.worker.progress.connect(self.on_bulk_progress)
        self.worker.done.connect(
            lambda results: self._on_calibration_done(results, ref_mag, xy, replace_index))
        self.progress_dlg.canceled.connect(self.worker.requestAbort)
        self.worker.start()

    def _on_calibration_done(self, results, ref_mag, xy=None, replace_index=None):
        self.progress_dlg.close()

        res_dict = {r['index']: r for r in results if r['success']}
        entry = {
            'ref_mag': ref_mag,
            'results': res_dict,
            'xy': tuple(xy) if xy else None,
            'policy': self._aperture_policy_label(),
            'name': f"Ref {len(self.calibration_history)+1}"
        }
        if replace_index is None:
            self.calibration_history.append(entry)
        else:
            entry['name'] = self.calibration_history[replace_index].get('name', entry['name'])
            self.calibration_history[replace_index] = entry

        # When a rule change is being applied to several reference stars, keep
        # going until every one has been re-measured, then show the dialog once.
        # A zeropoint built from an uncorrected magnitude in a "+apcorr" run is
        # off by the whole correction (~0.2-0.3 mag at 1 FWHM); collect such
        # frames over every reference star of this round and say so once.
        if not hasattr(self, '_calib_no_apcorr'):
            self._calib_no_apcorr = set()
        self._calib_no_apcorr.update(self._apcorr_unavailable(results))

        if getattr(self, '_repolicy_queue', None):
            nxt = self._repolicy_queue.pop(0)
            self._run_calibration_bulk(nxt['xy'], nxt['ref_mag'], replace_index=nxt['i'])
            return

        if self._calib_no_apcorr:
            QMessageBox.warning(self.view, "Aperture correction missing",
                                self._apcorr_unavailable_text(sorted(self._calib_no_apcorr)))
            self._calib_no_apcorr = set()
        self._show_calibration_dialog()

    def _rerun_calibration_with_policy(self, policy, param):
        """Re-measure every stored reference star under a different aperture rule."""
        usable = [(i, h) for i, h in enumerate(self.calibration_history) if h.get('xy')]
        if not usable:
            QMessageBox.information(
                self.view, 'Cannot re-measure',
                'These reference stars were calibrated before positions were '
                'recorded. Add the reference star again to use a different rule.')
            return False
        self.model.aperture_policy = policy
        if policy == 'frame_fwhm':
            self.model.fwhm_factor = float(param)
        elif policy == 'growth':
            self.model.growth_fraction = float(param)
        self.view.update_params(self.model)
        self._repolicy_queue = [{'i': i, 'xy': h['xy'], 'ref_mag': h['ref_mag']}
                                for i, h in usable]
        first = self._repolicy_queue.pop(0)
        self._run_calibration_bulk(first['xy'], first['ref_mag'], replace_index=first['i'])
        return True

    def _show_calibration_dialog(self):
        # We recreate the dialog with updated history
        dlg = ZeropointCalibrationDialog(self.view, self.model.files, self.calibration_history)
        
        # Connect signals
        dlg.request_add_ref.connect(lambda: self._on_request_add_ref(dlg))
        dlg.request_review.connect(lambda idx: self._on_request_review_ref(idx, dlg))
        dlg.request_repolicy.connect(
            lambda pol, prm: self._on_repolicy_requested(dlg, pol, prm))
        dlg.set_rule(self.model.aperture_policy,
                     self.model.fwhm_factor if self.model.aperture_policy == 'frame_fwhm'
                     else self.model.growth_fraction)
        
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            zps = dlg.get_results()
            policy = self._aperture_policy_label()
            for idx, zp in zps.items():
                fname = os.path.basename(self.model.files[idx])
                # Record what this zeropoint was built from, so it can be
                # audited (or distrusted) later from the Manage Zeropoints view.
                refs = []
                for h in self.calibration_history:
                    r = (h.get('results') or {}).get(idx)
                    if not r:
                        continue
                    instr = None
                    try:
                        instr = float(r['aperture_result']['instr_mag'])
                    except Exception:
                        try:
                            instr = float(r.get('mag'))
                        except Exception:
                            instr = None
                    xy = r.get('picked_full') or (float('nan'), float('nan'))
                    refs.append({'name': h.get('name', 'ref'),
                                 'ref_mag': float(h.get('ref_mag', float('nan'))),
                                 'instr_mag': instr,
                                 'zp': (float(h['ref_mag']) - instr) if instr is not None else float('nan'),
                                 'x': float(xy[0]), 'y': float(xy[1])})
                self.model.set_zeropoint_for_file(fname, zp,
                                                  meta={'refs': refs, 'policy': policy})
            QMessageBox.information(self.view, "Success", f"Applied zeropoints to {len(zps)} frames.")
            
            # Update global zeropoint to Mean? Or just let per-file ZP handle it.
            # To be safe, we might want to update the displayed ZP to average, but user wants Sync.
            # The per-file ZP handling in _measure_frame handles the sync.

    def _on_repolicy_requested(self, dlg, policy, param):
        dlg.close()
        if not self._rerun_calibration_with_policy(policy, param):
            self._show_calibration_dialog()

    def _on_request_add_ref(self, dlg):
        dlg.close()
        # Enter calibration mode again
        QMessageBox.information(self.view, "Add Reference", "Please click on the NEW reference star in the current frame, then click Confirm Reference.")
        self.zeropoint_calibration_mode = True
        self._setup_calibration_ui() # Show buttons again if hidden
    
    def _review_params(self):
        """
        The measurement settings the review dialog must start from.

        A frame re-measured by hand has to be measured the way the rest of the
        run was, or its instrumental magnitude -- and therefore its zeropoint --
        sits on a different radius rule and a different sky annulus than every
        other frame. So the review dialog is seeded from Advanced Settings
        rather than from its own hard-coded defaults.
        """
        return {
            'stamp_size': self.model.stamp_size,
            'fwhm': self.model.fwhm,
            'threshold': self.model.threshold,
            'inner_coef': self.model.inner_coef,
            'outer_coef': self.model.outer_coef,
            'aperture': self.model.aperture,
            'fixed_aperture': self.model.fixed_aperture,
            'aperture_policy': self.model.aperture_policy,
            'growth_fraction': self.model.growth_fraction,
            'fwhm_factor': self.model.fwhm_factor,
            'aperture_correct': self.model.aperture_correct,
            'gain': self.model.gain,
            'read_noise': self.model.read_noise,
            'dark_noise': self.model.dark_noise,
            'exptime_override': self.model.exptime_override,
        }

    def _on_request_review_ref(self, idx, dlg=None):
        from views.components.calibration_review_dialog import CalibrationReviewDialog
        rd = CalibrationReviewDialog(self.view, self.model.files, self.calibration_history, idx,
                                     params=self._review_params())
        rd.exec_()
        # A manual re-measure rewrites that frame's instrumental magnitude in
        # place, so the zeropoint table behind this dialog is now stale.
        # on_calculate() keeps any reference magnitude the user has typed but
        # not yet applied, then repopulates.
        if dlg is not None:
            try:
                dlg.on_calculate()
            except Exception as e:
                log.warning("Could not refresh the zeropoint table after review: %s", e)

    @QtCore.pyqtSlot()
    def save_csv(self):
        path, _ = QFileDialog.getSaveFileName(self.view, "Save CSV", "", "CSV (*.csv)")
        if not path: return
        import csv
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                # The first nine columns are unchanged so older scripts that read
                # this file by name keep working; everything after Zeropoint is
                # measurement provenance, needed to audit or reproduce a light
                # curve (which aperture, which camera mode / gain / read noise).
                # AperturePolicy records HOW the radius was chosen. Without it an
                # error bar cannot be interpreted: a per-star SNR optimum and a
                # frame-PSF radius give error bars that differ in meaning, and
                # magnitudes measured through different radii are not comparable
                # between stars (r=20 vs r=27 px is a 62 mmag offset of pure
                # aperture loss). ApertureCorrection records whether the reported
                # magnitude is a total magnitude or the flux inside the aperture
                # -- the two sit on zeropoint scales that differ by ~0.2 mag at a
                # one-FWHM radius, so the distinction cannot be left implicit.
                policy = self.model.aperture_policy
                if self.model.fixed_aperture:
                    policy_str = f"fixed:{self.model.aperture:g}px"
                elif policy == "frame_fwhm":
                    policy_str = f"frame_fwhm:x{self.model.fwhm_factor:g}"
                elif policy == "growth":
                    policy_str = f"growth:E{self.model.growth_fraction:g}"
                else:
                    policy_str = "snr"
                writer.writerow(["Index", "File", "X", "Y", "Mag", "Err", "Flux", "SNR", "Zeropoint",
                                 "Aperture", "R_best", "InnerCoef", "OuterCoef",
                                 "BkgMean", "ApertureArea", "Exptime",
                                 "CameraMode", "Gain", "ReadNoise", "DarkNoise",
                                 "AperturePolicy", "ApertureCorrection",
                                 "PeakADU", "Saturation", "StarFWHM", "PSFCheck"])
                # Per row: "on", "off", or "unavailable" (asked for, but the
                # frame's curve of growth could not be measured, so that row is
                # an aperture magnitude). Rows measured before the status was
                # recorded fall back to the run-wide setting.
                run_ap_corr = "on" if self.model.aperture_correct else "off"
                status_to_csv = {"applied": "on", "off": "off", "unavailable": "unavailable"}
                for row in self.model.get_photometry_table():
                    ap_corr_str = status_to_csv.get(row.aperture_correction_status, run_ap_corr)
                    writer.writerow([row.index, row.filename, row.x, row.y, row.mag, row.mag_err, row.flux, row.snr, row.zeropoint,
                                     row.aperture, row.r_best, row.inner_coef, row.outer_coef,
                                     row.bkg_mean, row.aperture_area, row.exptime,
                                     row.camera_mode, row.gain, row.read_noise, row.dark_noise,
                                     policy_str, ap_corr_str,
                                     row.peak_adu, row.saturation,
                                     row.star_fwhm, row.psf_check])
        except PermissionError:
            QMessageBox.warning(self.view, "Cannot save CSV",
                f"Permission denied writing:\n{path}\n\n"
                "The file is probably open in another program (e.g. Excel). "
                "Close it and try again, or choose a different name.")
            return
        except OSError as e:
            QMessageBox.warning(self.view, "Cannot save CSV",
                f"Could not save the CSV:\n{path}\n\n{e}")
            return
        QMessageBox.information(self.view, "Saved", f"Results saved to {path}")

    @QtCore.pyqtSlot()
    def sort_by_time(self):
        """Order the frames by observation time.

        Frames whose time cannot be read keep their relative order and go to
        the END, and the user is told how many. They used to be given t = 0 by
        a bare `except: pass` and sorted to the FRONT, indistinguishable from
        the earliest real epoch.

        The time itself comes from ext_coeff.get_datetime_from_header, which
        knows DATE is UT on these headers and DATE-OBS is local -- a constant
        offset does not change the ORDER, but using one source keeps the two
        paths from disagreeing.
        """
        from astropy.io import fits
        from core.ext_coeff import get_datetime_from_header

        timed, untimed = [], []
        for i, p in enumerate(self.model.files):
            t = None
            try:
                dt = get_datetime_from_header(fits.getheader(p))
                if dt:
                    t = Time(dt).jd
            except Exception as e:
                log.warning("Cannot read an observation time from %s: %s", p, e)
            (timed if t is not None else untimed).append((i, p, t))

        order = [x[1] for x in sorted(timed, key=lambda x: x[2])]             + [x[1] for x in untimed]
        self.model.sort_by_time(order)

        if untimed:
            QMessageBox.warning(
                self.view, "Some frames have no time",
                f"{len(untimed)} of {len(self.model.files)} frame(s) carry no "
                f"readable observation time. They were left in their original "
                f"order at the END of the list rather than being treated as the "
                f"earliest epoch.")


    @QtCore.pyqtSlot()
    def save_snr(self):
        path, _ = QFileDialog.getSaveFileName(self.view, "Save SNR Graph", "", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            try:
                save_publication_figure(self.view.fig_snr, path, dpi=300)
            except PermissionError:
                QMessageBox.warning(self.view, "Cannot save graph",
                    f"Permission denied writing:\n{path}\n\n"
                    "The file is probably open in another program. "
                    "Close it and try again, or choose a different name.")
                return
            except OSError as e:
                QMessageBox.warning(self.view, "Cannot save graph",
                    f"Could not save the graph:\n{path}\n\n{e}")
                return
            QMessageBox.information(self.view, "Saved", f"Graph saved to {path}")
