import os
import sys
import json
import numpy as np
from datetime import datetime
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QProgressDialog

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
from core.multi_photometry_worker import BulkPhotometryWorker
from utils.gui_helpers import plot_profile_and_snr, ExtinctionDialog, get_output_dir
from utils.image_cache import ImageDisplayCache
from utils.exptime_utils import get_exptime_seconds
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
                # 2. Measurement optimization (SNR/Profile)
                # Radial profile
                r_used = self.p['aperture'] if self.p['fixed_ap'] else 10.0
                rp_rad, rp_prof = pc.compute_radial_profile(self.img, found_xy, max_radius=max(30, r_used*3))
                
                # SNR vs Radius
                radii, snrs, r_best, mag_err = pc.compute_snr_vs_radius(self.img, found_xy, fwhm=self.p['fwhm'], gain=16.5)
                
                # Photometry
                if not self.p['fixed_ap']: r_used = r_best
                
                res = pc.perform_aperture_photometry(self.img, found_xy, r_used,
                                                    inner_coef=self.p['inner'],
                                                    outer_coef=self.p['outer'],
                                                    zeropoint=self.p['zp'],
                                                    exptime=self.p['exptime'])
                
                phot_res = {
                    'r_used': r_used,
                    'radii': radii, 'snrs': snrs, 'r_best': r_best,
                    'rp_rad': rp_rad, 'rp_prof': rp_prof,
                    'phot': res
                }
                
            self.finished.emit(found_xy, params, stamp, method, phot_res)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(None, None, None, 'error', {})

class InitialLoadWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)
    done = QtCore.pyqtSignal()
    
    def __init__(self, files, cache, max_dim):
        super().__init__()
        self.files = files
        self.cache = cache
        self.max_dim = max_dim
        
    def run(self):
        for i, p in enumerate(self.files):
            try:
                data = mfc.load_fits(p)
                # This populates the cache
                self.cache.get_display_data(p, data)
            except: pass
            self.progress.emit(i + 1)
        self.done.emit()

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
        
        self.refresh_all()

    def refresh_all(self):
        self.on_files_changed()
        self.on_selection_changed(self.model._current_index)

    @QtCore.pyqtSlot()
    def on_files_changed(self):
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
                radii, snrs, r_best, _ = pc.compute_snr_vs_radius(self.current_img, xy)
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
        
        self.view.set_results(data.mag, data.mag_err, data.flux, data.snr)
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
        zp_val = self.model.zeropoints.get(data.fname, self.model.zeropoint)
        self.view.spin_zp.blockSignals(True)
        self.view.spin_zp.setValue(zp_val)
        self.view.spin_zp.blockSignals(False)
        
        # Update Plots
        plot_profile_and_snr(self.view.fig_profile, self.view.fig_snr,
                             data.rp_radii, data.rp_profile,
                             data.radii, data.snrs, data.r_best)
        self.view.canvas_profile.draw_idle()
        self.view.canvas_snr.draw_idle()
        
        # Update Canvas patches
        self.view.canvas.clear_patches()
        if data.selected_source:
            from matplotlib.patches import Circle
            x, y = data.selected_source
            r = data.aperture_radius
            self.view.canvas.add_patch(Circle((x,y), r, color='cyan', fill=False, lw=1.5))
            self.view.canvas.add_patch(Circle((x,y), r*self.model.inner_coef, color='yellow', fill=False, lw=1, ls='--'))
            self.view.canvas.add_patch(Circle((x,y), r*self.model.outer_coef, color='yellow', fill=False, lw=1, ls='--'))

    @QtCore.pyqtSlot()
    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self.view, "Add FITS files", "", "FITS (*.fits *.fit *.fit.gz)")
        if not paths: return
        
        self.model.add_files(paths)
        
        # 1. Metadata extraction (Fast)
        from astropy.io import fits
        exptimes = []
        # Identify which indices were just added
        num_before = len(self.model.files) - len(paths)
        for i in range(num_before, len(self.model.files)):
            p = self.model.files[i]
            try:
                with fits.open(p) as hdul:
                    hdr = hdul[0].header
                    
                    # Use unified exposure time detection
                    et = get_exptime_seconds(hdr)
                    
                    date_ut = hdr.get('DATE')
                    date_obs = f"{hdr.get('DATE-OBS', '')}".replace('NOGPS', '').strip()
                    if not date_obs: date_obs = None
                    
                    self.model.update_frame_data(i, exptime=et, date_ut=date_ut, date_obs=date_obs)
                    if et is not None: exptimes.append(et)
            except: pass
        
        # Update override if we found something in headers
        if exptimes and self.model.exptime_override is None:
            med_et = float(np.median(exptimes))
            self.model.exptime_override = med_et
            self.view.spin_exptime.blockSignals(True)
            self.view.spin_exptime.setValue(med_et)
            self.view.spin_exptime.blockSignals(False)

        if self.model.exptime_override is None:
            # We only prompt if discovery from headers failed AND no previous override exists.
            val, ok = QtWidgets.QInputDialog.getDouble(self.view, "Exposure Time", 
                                                      "Exposure time missing from headers.\nPlease enter exposure time (s):", 
                                                      1.0, 0, 1e6, 2)
            if ok:
                self.model.exptime_override = val
                self.view.spin_exptime.blockSignals(True)
                self.view.spin_exptime.setValue(val)
                self.view.spin_exptime.blockSignals(False)
                # Update all frames added in THIS batch that failed to find exptime
                for i in range(num_before, len(self.model.files)):
                    fd = self.model._frame_data.get(i)
                    if fd and fd.exptime is None:
                         self.model.update_frame_data(i, exptime=val)
            
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
        
        # Prepare params for worker
        params = {
            'stamp_size': self.model.stamp_size,
            'fwhm': self.model.fwhm,
            'threshold': self.model.threshold,
            'fixed_ap': self.model.fixed_aperture,
            'aperture': self.model.aperture,
            'inner': self.model.inner_coef,
            'outer': self.model.outer_coef,
            'zp': self.model.zeropoint,
            'exptime': exptime_to_use
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
                    aperture_result=res
                )
            else:
                msg = "No star detected in target area."
                if method == 'not-found-in-crop':
                    msg = f"No any source detected in localized stamp ({self.model.stamp_size} px)."
                QMessageBox.warning(self.view, "Not Found", msg)
        
        self.det_worker.finished.connect(on_done)
        self.det_worker.start()

    def _measure_frame(self, index, xy, fast_mode=False):
        img = self.current_img
        if img is None: return
        
        data = self.model._frame_data.get(index)
        
        # Radius optimization if not fixed
        if not self.model.fixed_aperture:
            if not fast_mode or data.snrs is None:
                radii, snrs, r_best, mag_err = pc.compute_snr_vs_radius(img, xy, fwhm=self.model.fwhm, gain=16.5)
            else:
                radii, snrs, r_best = data.radii, data.snrs, data.r_best
            r_used = r_best
        else:
            radii, snrs, r_best = data.radii if data else None, data.snrs if data else None, data.r_best if data else None
            r_used = self.model.aperture
            
        # Photometry
        # Prefer specific zeropoint for this file if available
        fname = self.model._frame_data[index].fname
        zp = self.model.zeropoints.get(fname, self.model.zeropoint)
        
        res = pc.perform_aperture_photometry(img, xy, r_used,
                                            inner_coef=self.model.inner_coef,
                                            outer_coef=self.model.outer_coef,
                                            zeropoint=zp,
                                            exptime=self.model.exptime_override)
        
        # Profile - Only if not fast mode or missing
        if not fast_mode or data.rp_profile is None:
            rp_rad, rp_prof = pc.compute_radial_profile(img, xy, max_radius=max(30, r_used*3))
        else:
            rp_rad, rp_prof = data.rp_radii, data.rp_profile
        
        mag = res['mag'] # Use calibrated mag from core
        
        self.model.update_frame_data(index, 
            selected_source=xy,
            aperture_radius=r_used,
            radii=radii, snrs=snrs, r_best=r_best,
            rp_radii=rp_rad, rp_profile=rp_prof,
            mag=mag, mag_err=res['mag_err'], flux=res['flux'], snr=res['snr'],
            aperture_result=res
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
        self.model.aperture = int(self.view.spin_aperture.value())
        self.model.fixed_aperture = self.view.chk_fixed_aperture.isChecked()
        self.model.stamp_size = int(self.view.spin_stamp.value())
        self.model.tracking_radius = int(self.view.spin_tracking.value())
        self.model.fwhm = self.view.spin_fwhm.value()
        self.model.threshold = self.view.spin_thresh.value()
        self.model.inner_coef = self.view.spin_inner.value()
        self.model.outer_coef = self.view.spin_outer.value()
        self.model.zeropoint = self.view.spin_zp.value()
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
                  fname = self.model.files[self.model._current_index]
                  self.model.zeropoints[fname] = val
        
        self.view.blockSignals(False)
        
        # If parameters affecting photometry result changed (ZP or Exptime), update ALL frames with existing results
        if param_name in ["zeropoint", "exptime"]:
            self._update_all_frame_results()

        # Always re-measure current frame if source selected (Live Update)
        data = self.model.get_current_data()
        if data and data.selected_source:
            # Re-measure with fast_mode
            self._measure_frame(self.model._current_index, data.selected_source, fast_mode=True)
            self.show_current_data() # Update UI
        else:
            # If no current source, still trigger update so table refreshes if we updated all frames
            self.model.trigger_update()

    def _update_all_frame_results(self):
        """Recalculate mag for all frames that have aperture results, based on new ZP.
        
        CRITICAL: When updating ZP/Exptime, we must ONLY update frames where the user
        explicitly changed these values. We NEVER overwrite per-frame exptime values
        with the global exptime_override, as that would cause all measured frames to
        lose their individual exposure times.
        """
        import math
        zp = self.model.zeropoint
        
        for idx, data in self.model._frame_data.items():
            if data.aperture_result:
                # Use the frame's OWN exptime for magnitude calculation
                # If frame has no per-frame exptime, use global override as fallback
                et = data.exptime if (data.exptime is not None and data.exptime > 0) else self.model.exptime_override
                
                # Recalculate instr_mag if exptime available
                flux = data.aperture_result.get('flux', 0)
                if flux > 0 and et and et > 0:
                    instr_mag = -2.5 * math.log10(flux / et)
                else:
                    instr_mag = None
                
                # Update magnitude only - NEVER overwrite the frame's exptime
                new_mag = (instr_mag + zp) if instr_mag is not None else None
                
                # Note: perform_aperture_photometry returns more complex dict. 
                # We are just patching mag here for speed. We do NOT update exptime
                # to avoid losing per-frame values.
                
                self.model.update_frame_data(idx, mag=new_mag)
                # Note: exptime is intentionally NOT updated here to preserve per-frame values

    @QtCore.pyqtSlot()
    def start_bulk(self):
        data = self.model.get_current_data()
        if not data or not data.selected_source:
            QMessageBox.warning(self.view, "No Source", "Select a star in the current frame first.")
            return

        # For bulk photometry, ALWAYS read exposure from each frame's header (fresh read)
        # Do NOT use any cached/stored exptime values or exptime_override
        # This ensures each frame uses its own exposure time for magnitude calculation
        exptime_final = None  # Force reading from headers for each frame
        
        # Only use exptime_override if explicitly provided by user dialog
        # (This is for when headers lack EXPTIME and user provides a manual value)
        
        # Check if any frame header truly lacks EXPTIME
        needs_exp = False
        from astropy.io import fits
        for path in self.model.files:
            try:
                hdr = fits.getheader(path)
                # Check if EXPTIME key exists and has valid value
                if 'EXPTIME' not in hdr:
                    needs_exp = True
                    break
                et = hdr.get('EXPTIME')
                if et is None or float(et) <= 0:
                    needs_exp = True
                    break
            except:
                needs_exp = True
                break
        
        if needs_exp:
            val, ok = QtWidgets.QInputDialog.getDouble(self.view, "Exposure Time", 
                                                      "Some frames lack EXPTIME in header.\nEnter manual EXPTIME (s) for all frames:", 
                                                      1.0, 0.001, 1e5, 3)
            if not ok: return
            exptime_final = val

        # Start worker with fresh header reading enabled (exptime_override=None)
        self.worker = BulkPhotometryWorker(
            self.model.files, self.model._current_index, data.selected_source,
            fwhm=self.model.fwhm, threshold_sigma=self.model.threshold,
            search_stamp_size=self.model.tracking_radius,
            detection_stamp_size=self.model.stamp_size, # Pass the precision stamp
            aperture_override=self.model.aperture if self.model.fixed_aperture else None,
            inner_coef=self.model.inner_coef,
            outer_coef=self.model.outer_coef,
            zeropoint=self.model.zeropoint,
            zeropoint_map=self.model.zeropoints, # Sync per-frame ZPs
            exptime_override=exptime_final  # Only set if user manually provided a value
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
                self.model.update_frame_data(idx,
                    selected_source=r['picked_full'],
                    aperture_radius=r['aperture_used'],
                    radii=np.array(r['radii']), snrs=np.array(r['snrs']), r_best=r['r_best'],
                    mag=r['mag'], mag_err=r['mag_err'], flux=r['flux'], snr=r['snr'],
                    aperture_result=r['aperture_result'],
                    exptime=r['exptime']
                )
                if r['exptime']: exptimes.append(r['exptime'])
            else:
                failed += 1
                failed_frames.append((r['file'], r['msg']))
        
        if exptimes:
            med_et = float(np.median(exptimes))
            self.view.spin_exptime.setValue(med_et)
            self.model.exptime_override = med_et
        
        # Build user-friendly summary message
        summary = f"Bulk Photometry Complete.\n\nSuccessful: {successful}\nFailed: {failed}"
        
        if failed_frames:
            # Show detailed error information
            detail_msg = "Failed Frames:\n" + "\n".join([f"  • {fname}: {msg}" for fname, msg in failed_frames[:10]])
            if len(failed_frames) > 10:
                detail_msg += f"\n  ... and {len(failed_frames) - 10} more"
            
            QMessageBox.warning(self.view, "Bulk Photometry Results", f"{summary}\n\n{detail_msg}")
        else:
            QMessageBox.information(self.view, "Bulk Photometry Results", summary)
        
        self.model.trigger_update()

    def _read_header_datetime(self, path):
        """Robust parse DATE-OBS variant."""
        from astropy.io import fits
        try:
            hdr = fits.getheader(path)
            for key in ('DATE-OBS', 'DATE'):
                if key in hdr:
                    val = hdr[key]
                    try: return Time(val)
                    except: pass
        except: pass
        return None

    @QtCore.pyqtSlot()
    def calc_extinction(self):
        rows = self.model.get_photometry_table()
        if len(rows) < 2:
            QMessageBox.warning(self.view, "Insufficient Data", "Need at least 2 photometry points in the table.")
            return
            
        dlg = ExtinctionDialog(self.view)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            vals = dlg.values()
            ra_txt = vals['ra']; dec_txt = vals['dec']; filter_name = vals['flt'] or "unknown"
            lat = vals['lat']; lon = vals['lon']
            
            # Gather valid data
            mags, mag_errs, times, fnames = [], [], [], []
            for r in rows:
                if r.mag is None or not np.isfinite(r.mag): continue
                t = self._read_header_datetime(self.model.files[r.index])
                if t:
                    mags.append(r.mag); mag_errs.append(r.mag_err or 0.1)
                    times.append(t); fnames.append(r.filename)
            
            if len(mags) < 2:
                QMessageBox.warning(self.view, "No Time Info", "Missing DATE-OBS in headers for selected frames."); return
                
            try:
                # Heuristic: if colon in RA or 'h' in RA -> HourAngle, else Degrees
                if ':' in ra_txt or 'h' in ra_txt.lower():
                    sky = SkyCoord(ra=ra_txt, dec=dec_txt, unit=(u.hourangle, u.deg))
                else:
                    sky = SkyCoord(ra=float(ra_txt), dec=float(dec_txt), unit=(u.deg, u.deg))
            except Exception as e:
                QMessageBox.critical(self.view, "Parse Error", f"Unable to parse RA/Dec: {e}"); return
                
            obstimes = Time([t.iso for t in times])
            location = EarthLocation(lat=lat*u.deg, lon=lon*u.deg)
            altaz = sky.transform_to(AltAz(obstime=obstimes, location=location))
            airmass = 1.0 / np.cos(np.deg2rad(90.0 - altaz.alt.deg))
            
            # Fit
            mags = np.array(mags); airmass = np.array(airmass); errs = np.array(mag_errs)
            p, cov = np.polyfit(airmass, mags, 1, w=1.0/errs, cov=True)
            k = p[0]; intercept = p[1]; k_err = np.sqrt(cov[0,0])
            
            # Save JSON
            outdir = os.path.join(get_output_dir(), "k_results")
            os.makedirs(outdir, exist_ok=True)
            outpath = os.path.join(outdir, f"extinction_k_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
            
            res = {
                "created_at": datetime.utcnow().isoformat(), "target": {"ra": ra_txt, "dec": dec_txt},
                "fit": {"k": float(k), "k_err": float(k_err), "intercept": float(intercept)},
                "measurements": [{"file": f, "airmass": float(a), "mag": float(m)} for f,a,m in zip(fnames, airmass, mags)]
            }
            with open(outpath, "w") as f: json.dump(res, f, indent=2)
            
            QMessageBox.information(self.view, "Success", f"k = {k:.5f} \u00b1 {k_err:.5f}\nSaved to {outpath}")
            
            # Advanced Plot
            p_dlg = ExtinctionPlotDialog(self.view, airmass, mags, errs, k, intercept, k_err)
            p_dlg.exec_()

    @QtCore.pyqtSlot()
    def review_table(self):
        rows = self.model.get_photometry_table()
        # Mix in the phot_table from model? 
        dlg = PhotometryTableDialog(rows, self.view)
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
            self.view.fig_profile.savefig(path, bbox_inches='tight', dpi=300)
            QMessageBox.information(self.view, "Saved", f"Profile graph saved to {path}")

    @QtCore.pyqtSlot()
    def add_current_to_table(self):
        data = self.model.get_current_data()
        if not data or data.selected_source is None:
            QMessageBox.warning(self.view, "No measurement", "Perform photometry on the current frame first.")
            return
        
        row = PhotometryRow(
            index=self.model._current_index,
            filename=data.fname,
            x=data.selected_source[0],
            y=data.selected_source[1],
            mag=data.mag,
            mag_err=data.mag_err,
            flux=data.flux,
            snr=data.snr,
            zeropoint=self.model.zeropoint
        )
        self.model.add_to_table(row)
        QMessageBox.information(self.view, "Added", f"Added measurement for {data.fname} to results table.")

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
    def calibrate_zeropoints(self):
        if not self.model.files:
            QMessageBox.warning(self.view, "No Files", "Load files first.")
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
            self.btn_confirm_ref.setStyleSheet("background: #28a745; color: white;")
            self.btn_cancel_ref.setStyleSheet("background: #dc3545; color: white;")
            l.addWidget(self.btn_confirm_ref)
            l.addWidget(self.btn_cancel_ref)
            
            # Insert calibration bar into center layout
            if hasattr(self.view, 'center_layout'):
                self.view.center_layout.insertWidget(0, self.confirm_cancel_buttons)
            
            self.btn_confirm_ref.clicked.connect(self._on_confirm_ref)
            self.btn_cancel_ref.clicked.connect(self._on_cancel_calib)
        
        self.confirm_cancel_buttons.show()

    def _handle_calibration_click(self, x, y):
        self.progress_dlg = QProgressDialog("Detecting Calibration Star...", "Cancel", 0, 0, self.view)
        self.progress_dlg.setWindowModality(QtCore.Qt.WindowModal)
        self.progress_dlg.show()
        
        params = {
            'stamp_size': self.model.stamp_size,
            'fwhm': self.model.fwhm,
            'threshold': self.model.threshold,
            'fixed_ap': self.model.fixed_aperture,
            'aperture': self.model.aperture,
            'inner': self.model.inner_coef,
            'outer': self.model.outer_coef,
            'zp': self.model.zeropoint,
            'exptime': self.model.exptime_override
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

    def _run_calibration_bulk(self, xy, ref_mag):
        # We reuse BulkPhotometryWorker
        self.worker = BulkPhotometryWorker(
            self.model.files, self.model._current_index, xy,
            fwhm=self.model.fwhm, threshold_sigma=self.model.threshold,
            search_stamp_size=self.model.tracking_radius,
            detection_stamp_size=self.model.stamp_size,
            aperture_override=self.model.aperture if self.model.fixed_aperture else None,
            exptime_override=self.model.exptime_override,
            zeropoint=0.0, # We want instrumental mag for calibration
            zeropoint_map={} # Clear map for calibration run
        )
        self.progress_dlg = QProgressDialog("Calibrating Zeropoints...", "Cancel", 0, 100, self.view)
        self.progress_dlg.show() # Show immediately
        self.worker.progress.connect(self.on_bulk_progress)
        self.worker.done.connect(lambda results: self._on_calibration_done(results, ref_mag))
        self.worker.start()

    def _on_calibration_done(self, results, ref_mag):
        self.progress_dlg.close()
        
        # Append to history
        res_dict = {r['index']: r for r in results if r['success']}
        self.calibration_history.append({
            'ref_mag': ref_mag,
            'results': res_dict,
            'name': f"Ref {len(self.calibration_history)+1}"
        })
        
        self._show_calibration_dialog()

    def _show_calibration_dialog(self):
        # We recreate the dialog with updated history
        dlg = ZeropointCalibrationDialog(self.view, self.model.files, self.calibration_history)
        
        # Connect signals
        dlg.request_add_ref.connect(lambda: self._on_request_add_ref(dlg))
        dlg.request_review.connect(lambda idx: self._on_request_review_ref(idx))
        
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            zps = dlg.get_results()
            for idx, zp in zps.items():
                fname = os.path.basename(self.model.files[idx])
                self.model.set_zeropoint_for_file(fname, zp)
            QMessageBox.information(self.view, "Success", f"Applied zeropoints to {len(zps)} frames.")
            
            # Update global zeropoint to Mean? Or just let per-file ZP handle it.
            # To be safe, we might want to update the displayed ZP to average, but user wants Sync.
            # The per-file ZP handling in _measure_frame handles the sync.

    def _on_request_add_ref(self, dlg):
        dlg.close()
        # Enter calibration mode again
        QMessageBox.information(self.view, "Add Reference", "Please click on the NEW reference star in the current frame, then click Confirm Reference.")
        self.zeropoint_calibration_mode = True
        self._setup_calibration_ui() # Show buttons again if hidden
    
    def _on_request_review_ref(self, idx):
        from views.components.calibration_review_dialog import CalibrationReviewDialog
        rd = CalibrationReviewDialog(self.view, self.model.files, self.calibration_history, idx)
        rd.exec_()

    @QtCore.pyqtSlot()
    def save_csv(self):
        path, _ = QFileDialog.getSaveFileName(self.view, "Save CSV", "", "CSV (*.csv)")
        if not path: return
        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Index", "File", "X", "Y", "Mag", "Err", "Flux", "SNR", "Zeropoint"])
            for row in self.model.get_photometry_table():
                writer.writerow([row.index, row.filename, row.x, row.y, row.mag, row.mag_err, row.flux, row.snr, row.zeropoint])
        QMessageBox.information(self.view, "Saved", f"Results saved to {path}")

    @QtCore.pyqtSlot()
    def sort_by_time(self):
        # Heuristic sort using DATE-OBS from headers
        from astropy.io import fits
        files_with_time = []
        for p in self.model.files:
            t = 0
            try:
                hdr = fits.getheader(p)
                for key in ('DATE-OBS', 'DATE'):
                    if key in hdr:
                        try: t = Time(hdr[key]).jd; break
                        except: pass
            except: pass
            files_with_time.append((p, t))
            
        sorted_files = [x[0] for x in sorted(files_with_time, key=lambda x: x[1])]
        # Preserve all data by using model's sort_by_time
        self.model.sort_by_time(sorted_files)


    @QtCore.pyqtSlot()
    def save_snr(self):
        path, _ = QFileDialog.getSaveFileName(self.view, "Save SNR Graph", "", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self.view.fig_snr.savefig(path, bbox_inches='tight', dpi=300)
            QMessageBox.information(self.view, "Saved", f"Graph saved to {path}")
