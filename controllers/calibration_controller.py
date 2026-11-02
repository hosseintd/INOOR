import os
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QObject, pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
from astropy.io import fits

# Adjust path to find root
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from models.file_set import FileSet
from utils.exptime_utils import get_exptime_seconds
from utils.image_cache import ImageDisplayCache
from utils.logging_config import get_logger
from utils.thread_guard import is_worker_busy
from views.calibration_view import CalibrationView

log = get_logger(__name__)
from views.components.review_dialog import ReviewDialog
from views.components.calibration_summary_dialog import CalibrationSummaryDialog
from views.components.flat_screen_dialog import FlatScreenDialog
from core import masterFrame_creator as mfc
from controllers.workers import CalibrationWorker, FlatScreenWorker, ImagePreloadWorker

class CalibrationController(QObject):
    def __init__(self, model, view: CalibrationView):
        super().__init__()
        self.model = model
        self.view = view
        
        self.current_img_data = None
        self.metadata_cache = {} # path -> {XBINNING, NAXIS1, CCD-TEMP, EXPTIME}
        self.auto_binning = False
        # Downsampled previews + ZScale limits, warmed in the background after
        # files are added so clicking through the tree does not re-read a
        # 4096x4096 frame on the GUI thread every time.
        # 512 px is ample for a 360 px-wide preview panel and keeps each cached
        # frame at ~1 MB, so a whole night's worth of calibration frames fits.
        self.image_cache = ImageDisplayCache(max_display_dim=512)
        self.image_cache.max_cache_size = 200
        self.preload_worker = None
        
        # Debounce timer for tree updates to prevent rapid re-entrance crashes
        self.tree_update_timer = QtCore.QTimer()
        self.tree_update_timer.timeout.connect(self._do_tree_update)
        self.tree_update_timer.setSingleShot(True)
        self.pending_tree_update = False
        
        # Connect View signals
        self.view.browse_files_clicked.connect(self.browse_and_add_files)
        self.view.remove_item_clicked.connect(self.remove_selected_item)
        self.view.file_selection_changed.connect(self.select_file)
        self.view.set_selection_changed.connect(self.select_set)
        self.view.bad_frame_toggled.connect(self.toggle_bad_frame)
        self.view.review_set_clicked.connect(self.review_set)
        self.view.screen_flats_clicked.connect(self.screen_flats)
        self.view.start_calibration_clicked.connect(self.start_calibration)
        self.view.param_changed.connect(self.update_params_from_view)
        self.view.hist_update_range_clicked.connect(self.on_hist_range_changed)
        
        # Connect Model signals
        self.model.sets_changed.connect(self.on_sets_changed)
        self.model.data_changed.connect(self.on_data_changed)
        
        self.refresh_all()

    def refresh_all(self):
        self._ensure_metadata_cache()
        self.on_sets_changed()
        self.on_data_changed()

    def _ensure_metadata_cache(self):
        # Scan all files in all sets and fill cache if missing
        for s in self.model.get_sets():
            for f in s.files:
                if f not in self.metadata_cache:
                    try:
                        hdr = fits.getheader(f)
                        self.metadata_cache[f] = {
                            'XBINNING': hdr.get('XBINNING', 1),
                            'YBINNING': hdr.get('YBINNING', 1),
                            'NAXIS1': hdr.get('NAXIS1', 0),
                            'NAXIS2': hdr.get('NAXIS2', 0),
                            'CCD-TEMP': hdr.get('CCD-TEMP', 0),
                            'EXPTIME': hdr.get('EXPTIME', 0)
                        }
                    except Exception as e:
                        # A file whose header cannot be read drops out of the
                        # shape/binning decisions -- must not vanish silently.
                        log.warning("Failed to read header of %s: %s", f, e)

    def _validate_frame_shapes(self, sets):
        """Validate that all frames in each set have the same shape. Return True if valid."""
        for s in sets:
            if not s.files:
                continue
            
            shapes = set()
            unreadable = []
            for f in s.files:
                try:
                    hdr = fits.getheader(f)
                    ny = hdr.get('NAXIS2', 0)
                    nx = hdr.get('NAXIS1', 0)
                    shapes.add((ny, nx))
                except Exception as e:
                    log.warning("Shape validation: cannot read header of %s: %s", f, e)
                    unreadable.append(os.path.basename(f))

            if unreadable:
                QMessageBox.warning(self.view, "Unreadable Frames",
                                    f"{len(unreadable)} frame(s) in the {s.set_type} set could not "
                                    f"be read and were EXCLUDED from shape validation:\n"
                                    + "\n".join(unreadable[:6]))

            if len(shapes) > 1:
                QMessageBox.critical(self.view, "Shape Mismatch in Frame Set",
                                   f"All frames in {s.set_type} set must have the same shape.\n"
                                   f"Detected shapes: {sorted(list(shapes))}\n\n"
                                   f"Please ensure all {s.set_type} frames are taken at the same binning.\n"
                                   f"You can load frames of different shapes in separate sets.")
                return False
        
        return True

    def _exposure_times(self, fset):
        """Distinct exposure times (seconds) among the good frames of a set."""
        if not fset:
            return []
        bad = getattr(fset, 'bad_indices', set()) or set()
        seconds = set()
        for i, f in enumerate(fset.files):
            if i in bad:
                continue
            try:
                seconds.add(get_exptime_seconds(fits.getheader(f), default=None))
            except Exception as e:
                log.warning("Cannot read EXPTIME from %s: %s", f, e)
        seconds.discard(None)
        return sorted(seconds)

    def _validate_exposure_times(self, sets):
        """Warn when a set that gets combined into one master mixes exposure
        times. Stacking 10 s and 60 s darks into a single master silently
        produces a dark that matches neither, and the exposure-scaling factor
        is then taken from whichever frame happened to be read first."""
        for set_type in ('Dark', 'Flat', 'Bias'):
            fset = next((s for s in sets if s.set_type == set_type), None)
            times = self._exposure_times(fset)
            if len(times) > 1:
                res = QMessageBox.warning(
                    self.view, f"Mixed Exposure Times in {set_type} Set",
                    f"The {set_type} set contains frames with different exposure "
                    f"times: {', '.join(f'{t:g}s' for t in times)}.\n\n"
                    f"They will be combined into a SINGLE master {set_type.lower()}, "
                    f"which will not match any of them. Load each exposure time as "
                    f"a separate run instead.\n\nContinue anyway?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if res != QMessageBox.Yes:
                    return False
        return True

    def _confirm_no_bias(self, has_flats):
        """Explain what running without bias actually does -- in particular how
        the flat still gets its pedestal removed, so the sensor seam does not
        survive flat-fielding. Returns True to continue."""
        box = QMessageBox(self.view)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("No Bias Frames")
        box.setTextFormat(QtCore.Qt.RichText)
        box.setText("<b>No Bias frames were loaded. Continue without bias?</b>")

        light_part = (
            "<p><b>Light frames</b> &mdash; nothing is lost. A master dark already "
            "contains the bias, so subtracting it removes both at once:</p>"
            "<pre>    calibrated = light - master_dark</pre>"
        )

        if has_flats:
            flat_part = (
                "<p><b>Flat frames &mdash; this is the part that matters.</b> "
                "A raw flat is not just the pixel response:</p>"
                "<pre>    raw_flat = bias + dark_flat + response x illumination</pre>"
                "<p>If the bias is left inside the flat, dividing by it stamps the "
                "bias structure &mdash; on the KL4040 the quadrant-seam '+' lines "
                "&mdash; back onto every science frame, <i>inverted</i>, with "
                "amplitude:</p>"
                "<pre>    residual = seam_step / flat_level</pre>"
                "<p>That is why the seam disappears when the dark is subtracted "
                "from the light, but comes back as soon as an uncorrected flat is "
                "applied.</p>"
                "<p>Because you have no bias, INOOR will use the "
                "<b>master dark as the flat's pedestal</b> instead &mdash; it "
                "contains the same bias structure:</p>"
                "<pre>    flat_used = master_flat - master_dark</pre>"
                "<p>Each flat is then scaled to unity by its own median before "
                "combining, so <b>the flat will not re-stamp the seam</b>.</p>"
                "<p><i>Small caveat:</i> this also subtracts the dark's own dark "
                "current, which differs from the flat's if their exposure times "
                "differ. For short flats that is far below 0.1% and harmless.</p>"
                "<p><b>The seam can still come back by another route.</b> If "
                "dark scaling is on, the scale factor multiplies the pedestal "
                "inside the dark &mdash; seam included &mdash; and leaves a "
                "residual of about <tt>seam_step x (scale - 1)</tt>. Measured on "
                "INO340 i-band frames (97 s lights, 29 s darks, scale 3.33) the "
                "seam went from <b>-0.11% with a bias to -1.14% without one</b>. "
                "The next prompt lets you turn scaling off.</p>"
            )
        else:
            flat_part = (
                "<p>No flat frames are loaded, so there is no flat pedestal to "
                "worry about.</p>"
            )

        box.setInformativeText(light_part + flat_part)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.Yes)
        return box.exec_() == QMessageBox.Yes

    def _confirm_unbiased_dark_scaling(self, sets):
        """Dark scaling multiplies the WHOLE master dark, pedestal included,
        when no bias is available. Get informed consent rather than silently
        injecting a bias*(scale-1) offset. Returns True to keep scaling on."""
        lights = next((s for s in sets if s.set_type == 'Light'), None)
        darks = next((s for s in sets if s.set_type == 'Dark'), None)
        light_times = self._exposure_times(lights)
        dark_times = self._exposure_times(darks)
        if not light_times or not dark_times:
            return True
        # Nothing to scale if the exposures already match.
        if max(abs(lt / dark_times[0] - 1.0) for lt in light_times) <= 0.001:
            return True

        res = QMessageBox.question(
            self.view, "Dark Scaling Without Bias",
            "Dark scaling is enabled but no Bias frames were provided.\n\n"
            "A master dark is bias + dark current. Without a bias frame the "
            "scaling factor multiplies the bias pedestal as well, which adds a "
            "spurious offset of roughly bias x (scale - 1) ADU to every "
            "calibrated pixel. Note also that sensor response is not perfectly "
            "linear with exposure time, so scaled darks are an approximation.\n\n"
            "The pedestal is not flat: each KL4040 quadrant sits at its own "
            "level, so scaling it also multiplies the quadrant seam and puts "
            "the '+' back across the middle of every calibrated frame. On "
            "INO340 i-band data (97 s lights, 29 s darks) the seam measured "
            "-0.11% with a bias loaded and -1.14% without one.\n\n"
            f"Light exposure(s): {', '.join(f'{t:g}s' for t in light_times)}\n"
            f"Dark exposure: {dark_times[0]:g}s\n"
            f"Scale factor: x{light_times[0] / dark_times[0]:.3f}\n\n"
            "Scale the dark anyway?\n"
            "(No = keep the dark unscaled. With this camera's dark current that "
            "is usually the better trade: at -10 C the median pixel accrues "
            "under 1 ADU in 30 s, so the scaling mostly amplifies noise and the "
            "seam.)",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return res == QMessageBox.Yes

    @pyqtSlot()
    def on_sets_changed(self):
        sets = self.model.get_sets()
        self.view.update_tree(sets, self.metadata_cache)
    
    def schedule_tree_update(self):
        """Schedule a tree update with debounce to prevent rapid re-entrance crashes."""
        self.pending_tree_update = True
        if not self.tree_update_timer.isActive():
            self.tree_update_timer.start(100)  # 100ms debounce
    
    def _do_tree_update(self):
        """Actually perform the tree update."""
        if self.pending_tree_update:
            self.pending_tree_update = False
            self.on_sets_changed()

    @pyqtSlot()
    def on_data_changed(self):
        s = self.model.get_current_set()
        if s:
            self.view.update_params(s)
        # Use debounced update to prevent crashes from rapid changes
        self.schedule_tree_update()

    @pyqtSlot(str)
    def browse_and_add_files(self, set_type):
        from astropy.io import fits
        paths, _ = QFileDialog.getOpenFileNames(self.view, f"Select {set_type} files", "", "FITS (*.fits *.fit)")
        if not paths: return
        
        if set_type == "HotPixelMap":
            if len(paths) > 1:
                QMessageBox.warning(self.view, "Load Limit", "Please load only one Hot Pixel Mask. Using the first selected file.")
                paths = [paths[0]]
            try:
                data = mfc.load_fits(paths[0])
                # Check for binary values (allowing small float tolerances)
                valid = np.all((np.abs(data - 0) < 1e-3) | (np.abs(data - 1) < 1e-3))
                if not valid:
                    QMessageBox.critical(self.view, "Validation Error", 
                                       "The selected Hot Pixel Map is not binary.\n"
                                       "It must contain only 0 and 1 values (where 1 indicates a hot pixel).")
                    return
            except Exception as e:
                QMessageBox.critical(self.view, "Error", f"Could not read mask file:\n{e}")
                return
        
        new_metas = []
        for p in paths:
            if p not in self.metadata_cache:
                try:
                    hdr = fits.getheader(p)
                    self.metadata_cache[p] = {
                        'XBINNING': hdr.get('XBINNING', 1),
                        'YBINNING': hdr.get('YBINNING', 1),
                        'NAXIS1': hdr.get('NAXIS1', 0),
                        'NAXIS2': hdr.get('NAXIS2', 0),
                        'CCD-TEMP': hdr.get('CCD-TEMP', 0),
                        'EXPTIME': hdr.get('EXPTIME', 0)
                    }
                except Exception as e:
                    log.warning("Failed to read header of %s: %s", p, e)
            if p in self.metadata_cache:
                new_metas.append(self.metadata_cache[p])

        # Check for binning mismatch in these new files
        bins = set(m['XBINNING'] for m in new_metas)
        # Also check against existing sets
        for s in self.model.get_sets():
            for f in s.files:
                if f in self.metadata_cache:
                    bins.add(self.metadata_cache[f]['XBINNING'])
        
        if len(bins) > 1 and not self.auto_binning:
            res = QMessageBox.question(self.view, "Binning Mismatch",
                                    f"Detected different XBINNING values: {sorted(list(bins))}. "
                                    "Do you want to enable auto-binning to match the smallest image size?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if res == QMessageBox.Yes:
                self.auto_binning = True
            else:
                QMessageBox.warning(self.view, "Mismatch", "Note: Calibration may fail if image sizes do not match.")

        # Find if set of this type already exists, or create new
        existing_set = None
        for s in self.model.get_sets():
            if s.set_type == set_type:
                existing_set = s
                break
        
        if existing_set:
            if set_type == "HotPixelMap":
                existing_set.files = paths # Replace instead of extend
            else:
                new_files = [p for p in paths if p not in existing_set.files]
                existing_set.files.extend(new_files)
        else:
            new_set = FileSet(set_type=set_type, files=paths)
            # Light frames should have sigma clipping disabled by default
            # User can manually enable it if desired
            if set_type == "Light":
                new_set.do_sigma_clip = False
            self.model.add_set(new_set)

        self.model.trigger_update()
        self._start_preload()

    @pyqtSlot()
    def remove_selected_item(self):
        item = self.view.tree_data.currentItem()
        if not item: return
        
        if not item.parent():
            # Parent item: Remove whole set
            set_type = item.text(0)
            sets = self.model.get_sets()
            for i, s in enumerate(sets):
                if s.set_type == set_type:
                    self.model.remove_set(i)
                    break
        else:
            # Child item: Remove single file
            parent = item.parent()
            set_type = parent.text(0)
            file_idx = parent.indexOfChild(item)
            
            for s in self.model.get_sets():
                if s.set_type == set_type:
                    if 0 <= file_idx < len(s.files):
                        s.files.pop(file_idx)
                        if file_idx in s.bad_indices:
                            s.bad_indices.remove(file_idx)
                        # Re-shift bad indices
                        new_bad = set()
                        for bi in s.bad_indices:
                            if bi > file_idx: new_bad.add(bi - 1)
                            else: new_bad.add(bi)
                        s.bad_indices = new_bad
                    break
        
        # Always trigger update regardless of what was removed
        self.model.trigger_update()

    @pyqtSlot(str, int)
    def select_file(self, set_type, index):
        print(f"DEBUG: select_file called with set_type={set_type}, index={index}")
        # Set this as current set in model
        sets = self.model.get_sets()
        for i, s in enumerate(sets):
            if s.set_type == set_type:
                self.model.set_current_set_index(i)
                
                # Load Preview
                if 0 <= index < len(s.files):
                    self._show_preview(s.files[index])
                break

    def _show_preview(self, path):
        """Render the preview + histogram for one frame.

        Served from ImageDisplayCache: on a cache hit nothing touches the disk
        and no ZScale is recomputed, so switching frames is instant. On a miss
        (file added but preload still running) it falls back to reading the
        frame directly, exactly as before.
        """
        try:
            entry = self.image_cache.cache.get(path)
            if entry is None:
                entry = self.image_cache.get_display_data(path, mfc.load_fits(path))

            img = entry['display_data']
            self.current_img_data = img
            self.view.canvas_image.show_image(img)
            self.view.lbl_fname.setText(os.path.basename(path))

            arr = img.ravel()
            arr = arr[np.isfinite(arr)]
            counts, bins = np.histogram(arr, bins=2000)
            self.view.canvas_hist.show_hist(counts, bins)

            self.view.canvas_hist.set_limits(entry['vmin'], entry['vmax'])
        except Exception as e:
            log.warning("Preview failed for %s: %s", path, e)

    def _start_preload(self):
        """Warm the display cache for every loaded frame, in the background."""
        paths = [f for s in self.model.get_sets() for f in s.files
                 if f not in self.image_cache.cache]
        if not paths:
            return
        if self.preload_worker is not None and self.preload_worker.isRunning():
            self.preload_worker.requestInterruption()
            self.preload_worker.wait(2000)

        self.view.lbl_status.setText(f"Pre-loading {len(paths)} frame(s)...")
        self.preload_worker = ImagePreloadWorker(paths, self.image_cache)
        self.preload_worker.done.connect(
            lambda: self.view.lbl_status.setText("Ready"))
        self.preload_worker.start()
    
    @pyqtSlot(str)
    def select_set(self, set_type):
        """Handle selection of a set (parent item in tree)."""
        print(f"DEBUG: select_set called with set_type={set_type}")
        sets = self.model.get_sets()
        for i, s in enumerate(sets):
            if s.set_type == set_type:
                self.model.set_current_set_index(i)
                # Update the parameter controls to show this set's parameters
                self.view.update_params(s)
                # Don't rebuild tree here - it clears the selection and breaks remove button
                # Selection and highlighting are handled by _on_tree_selection_changed()
                break

    @pyqtSlot(str, int, bool)
    def toggle_bad_frame(self, set_type, index, is_bad):
        for s in self.model.get_sets():
            if s.set_type == set_type:
                if is_bad: s.bad_indices.add(index)
                else: s.bad_indices.discard(index)
                # Trigger tree update to show visual changes (red color) immediately
                self.model.trigger_update()
                break

    @pyqtSlot(float, float)
    def on_hist_range_changed(self, mn, mx):
        if mn == -1 and mx == -1:
            # Reset
            if self.current_img_data is not None:
                vmin, vmax = mfc.zscale_limits(self.current_img_data)
                self.view.canvas_image.set_clim(vmin, vmax)
                self.view.canvas_hist.set_limits(vmin, vmax)
        else:
            self.view.canvas_image.set_clim(mn, mx)

    @pyqtSlot()
    def update_params_from_view(self):
        s = self.model.get_current_set()
        if not s: return
        s.method = self.view.combo_method.currentText()
        
        # Allow sigma clipping for any set type, but Light frames default to disabled
        # User can enable it if desired
        s.do_sigma_clip = self.view.chk_sigma.isChecked()
        
        s.sigma_lower = self.view.spin_sigma_lo.value()
        s.sigma_upper = self.view.spin_sigma_hi.value()
        
        # Trigger tree update to reflect changes immediately
        self.model.trigger_update()

    @pyqtSlot()
    def review_set(self):
        s = self.model.get_current_set()
        if not s or not s.files: return
        
        dlg = ReviewDialog(s.files, initial_bad=s.bad_indices, parent=self.view)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            s.bad_indices = dlg.bad_indices
            self.model.trigger_update()

    @pyqtSlot()
    def screen_flats(self):
        """Measure the loaded flats and offer to exclude the bad ones.

        Screening only ADVISES: the operator sees every measurement and decides.
        A flat folder routinely holds frames that are not usable flats -- past
        the linear range, saturated, full of stars, or affected by a sensor
        glitch -- and combining those quietly corrupts every science frame
        divided by the master.
        """
        if is_worker_busy(self, 'screen_worker', view=self.view,
                          busy_msg="Flat screening is already running."):
            return

        flat_sets = [s for s in self.model.get_sets() if s.set_type == 'Flat' and s.files]
        if not flat_sets:
            QMessageBox.information(
                self.view, "No Flats",
                "Load Flat frames first -- there is nothing to check.")
            return
        if len(flat_sets) > 1:
            QMessageBox.warning(
                self.view, "Several Flat Sets",
                "More than one Flat set is loaded. Only the first is screened; "
                "the sets are combined into one master anyway, so screen them "
                "as a single set.")
        self._screen_target = flat_sets[0]

        files = list(self._screen_target.files)
        self.screen_progress = QProgressDialog(
            f"Checking {len(files)} flat frame(s)...", "Cancel", 0, 100, self.view)
        self.screen_progress.setWindowModality(QtCore.Qt.WindowModal)
        self.screen_worker = FlatScreenWorker(files)
        self.screen_progress.canceled.connect(self.screen_worker.requestInterruption)
        self.screen_worker.progress.connect(
            lambda f, m: (self.screen_progress.setValue(int(f * 100)),
                          self.screen_progress.setLabelText(m)))
        self.screen_worker.finished.connect(self._on_screen_finished)
        self.screen_progress.show()
        self.screen_worker.start()

    def _on_screen_finished(self, ok, result, message):
        self.screen_progress.close()
        if not ok:
            if message != "Cancelled by user":
                QMessageBox.critical(self.view, "Screening Failed", message)
            return

        dlg = FlatScreenDialog(result, parent=self.view)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            self.view.lbl_status.setText(message + " Nothing changed.")
            return

        # Positions came from the file list as it was when screening started;
        # re-match by path so a list edited meanwhile cannot mark the wrong row.
        failed_paths = {r['path'] for r in result['frames'] if not r['ok']}
        fset = self._screen_target
        marked = {i for i, f in enumerate(fset.files) if f in failed_paths}
        fset.bad_indices |= marked
        self.model.trigger_update()
        self.view.lbl_status.setText(
            f"Excluded {len(marked)} flat frame(s) after screening")

    @pyqtSlot()
    def start_calibration(self):
        if is_worker_busy(self, 'worker', view=self.view,
                          busy_msg="A calibration run is already in progress."):
            return
        sets = self.model.get_sets()
        has_lights = any(s.set_type == 'Light' for s in sets)
        if not has_lights:
            QMessageBox.warning(self.view, "No Lights", "You must load Light frames first.")
            return
        
        # VALIDATE: Check for Dark frames
        has_darks = any(s.set_type == 'Dark' for s in sets)
        if not has_darks:
            QMessageBox.critical(self.view, "No Dark Frames", 
                               "Dark frames are mandatory for calibration.\n"
                               "Please load at least one set of Dark frames before proceeding.")
            return
        
        has_flats = any(s.set_type == 'Flat' for s in sets)

        # VALIDATE: Check shape uniformity per set
        if not self._validate_frame_shapes(sets):
            return

        # VALIDATE: a master is only meaningful if its frames share an exposure
        if not self._validate_exposure_times(sets):
            return

        out_dir = QFileDialog.getExistingDirectory(self.view, "Select Output Directory")
        if not out_dir: return
        
        has_bias = any(s.set_type == 'Bias' for s in sets)
        
        allow_auto = False
        if not has_flats:
            res = QMessageBox.question(self.view, "No Flats",
                                     "No Flat frames were loaded. Create auto-flats per Light?\n\n"
                                     "Yes: each Light is divided by an auto-flat made from that frame itself.\n"
                                     "No: calibration does not start. Load Flat frames first.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if res == QMessageBox.Yes:
                allow_auto = True
            else:
                QMessageBox.information(self.view, "Calibration Not Started",
                                        "Calibration was not started. Load Flat frames, "
                                        "then press START CALIBRATION again.")
                return

        skip_bias = False
        if not has_bias:
            if not self._confirm_no_bias(has_flats):
                return
            skip_bias = True

        # Post-processing options
        scale_dark = self.view.chk_scale_dark.isChecked() if has_darks else False
        if scale_dark and skip_bias and not self._confirm_unbiased_dark_scaling(sets):
            scale_dark = False
        remove_cosmics = self.view.chk_cosmic.isChecked()
        oc_sigma = self.view.spin_outlier_sigma.value()
        oc_radius = self.view.spin_outlier_radius.value()
        
        bin_str = self.view.combo_binning.currentText()
        # Map: "1x1 (None)" -> 1, "2x2" -> 2, etc.
        bin_map = {"1x1 (None)": 1, "2x2": 2, "4x4": 4, "8x8": 8}
        bin_factor = bin_map.get(bin_str, 1)

        # Find Hot Pixel Mask
        hot_mask_path = None
        for s in sets:
            if s.set_type == 'HotPixelMap' and s.files:
                hot_mask_path = s.files[0]
                break

        # Generate and show summary dialog
        summary_data = self._generate_summary_data(sets, out_dir, skip_bias, allow_auto, 
                                                  scale_dark, remove_cosmics, oc_sigma, oc_radius, 
                                                  bin_factor, hot_mask_path)
        
        dlg = CalibrationSummaryDialog(summary_data, parent=self.view)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        # All validations passed, proceed with calibration
        self.view.btn_start.setEnabled(False)
        self.view.lbl_status.setText("Processing...")

        self.worker = CalibrationWorker(sets, out_dir, auto_flat=allow_auto, 
                                        skip_bias=skip_bias, auto_bin=True,
                                        remove_cosmics=remove_cosmics, bin_factor=bin_factor,
                                        scale_dark=scale_dark, outlier_sigma=oc_sigma,
                                        outlier_radius=oc_radius, hot_mask_path=hot_mask_path,
                                        metadata_cache=self.metadata_cache)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        
        self.progress_dlg = QProgressDialog("Calibrating frames...", "Cancel", 0, 100, self.view)
        self.progress_dlg.setWindowModality(QtCore.Qt.WindowModal)
        self.progress_dlg.canceled.connect(self.worker.requestInterruption)
        self.progress_dlg.show()
        
        self.worker.start()

    def _generate_summary_data(self, sets, out_dir, skip_bias, allow_auto, scale_dark, 
                               remove_cosmics, oc_sigma, oc_radius, bin_factor, hot_mask_path):
        """Generate summary data for the calibration preview dialog."""
        
        # Frame inventory
        inventory = {}
        lights = next((s for s in sets if s.set_type == 'Light'), None)
        darks = next((s for s in sets if s.set_type == 'Dark'), None)
        flats = next((s for s in sets if s.set_type == 'Flat'), None)
        bias = next((s for s in sets if s.set_type == 'Bias'), None)
        hpm = next((s for s in sets if s.set_type == 'HotPixelMap'), None)
        
        def get_shape_info(fset):
            if not fset or not fset.files:
                return "None"
            try:
                hdr = fits.getheader(fset.files[0])
                ny = hdr.get('NAXIS2', 0)
                nx = hdr.get('NAXIS1', 0)
                return f"{len(fset.files)} files ({nx}x{ny})"
            except:
                return f"{len(fset.files)} files"
        
        inventory['lights_summary'] = get_shape_info(lights)
        inventory['darks_summary'] = get_shape_info(darks)
        inventory['flats_summary'] = get_shape_info(flats)
        inventory['bias_summary'] = get_shape_info(bias)
        inventory['hpm_summary'] = "Yes" if hpm and hpm.files else "None"
        
        # Detect min shape
        all_shapes = []
        for s in sets:
            if s.files:
                try:
                    hdr = fits.getheader(s.files[0])
                    ny = hdr.get('NAXIS2', 0)
                    nx = hdr.get('NAXIS1', 0)
                    all_shapes.append((ny, nx))
                except:
                    pass
        
        min_shape = min(all_shapes) if all_shapes else (0, 0)
        min_shape_str = f"{min_shape[1]}x{min_shape[0]}"
        
        # Processing strategy
        processing = {'min_shape': min_shape_str, 'masters': {}}
        
        def create_master_info(fset, name):
            if not fset or not fset.files:
                return None
            
            try:
                hdr = fits.getheader(fset.files[0])
                ny = hdr.get('NAXIS2', 0)
                nx = hdr.get('NAXIS1', 0)
                shape = (ny, nx)
                
                binning_needed = shape != min_shape
                binning_info = f"{nx}x{ny}→{min_shape_str}" if binning_needed else f"{nx}x{ny} (no binning)"
                
                return {
                    'summary': f"Create from {len(fset.files)} frames",
                    'files': [os.path.basename(f) for f in fset.files],
                    'shape': shape,
                    'binning_needed': binning_needed,
                    'binning_info': binning_info,
                    'method': fset.method,
                    'sigma_clip': fset.do_sigma_clip,
                    'sigma_lower': fset.sigma_lower,
                    'sigma_upper': fset.sigma_upper,
                }
            except:
                return None
        
        if darks:
            processing['masters']['dark'] = create_master_info(darks, 'Dark')
        if flats:
            processing['masters']['flat'] = create_master_info(flats, 'Flat')
        if bias:
            processing['masters']['bias'] = create_master_info(bias, 'Bias')
        
        # Calibration settings. The exposure times are listed whether or not
        # scaling is on: an unscaled dark taken at a different exposure under-
        # subtracts the dark current, and that used to be invisible here.
        light_times = self._exposure_times(lights)
        dark_times = self._exposure_times(darks)

        def _fmt(times):
            if not times:
                return "unknown"
            return ", ".join(f"{t:g}s" for t in times)

        scaling_text = "Yes" if scale_dark else "No"
        if light_times and dark_times and dark_times[0]:
            ratio = light_times[0] / dark_times[0]
            if abs(ratio - 1.0) > 0.001:
                scaling_text += f"  (exposures differ by x{ratio:.2f})"

        calibration = {
            'light_exposure': _fmt(light_times),
            'dark_exposure': _fmt(dark_times),
            'dark_scaling': scaling_text,
            'skip_bias': "Yes" if skip_bias else "No",
            'auto_flat': "Yes" if allow_auto else "No",
        }
        
        # Light frame processing
        light_processing = {'binning_breakdown': [], 'sequence': []}
        
        if lights and lights.files:
            light_processing['summary'] = f"{len(lights.files)} light frames"
            
            # Breakdown by size
            shape_counts = {}
            for f in lights.files:
                try:
                    hdr = fits.getheader(f)
                    ny = hdr.get('NAXIS2', 0)
                    nx = hdr.get('NAXIS1', 0)
                    shape = (ny, nx)
                    shape_counts[shape] = shape_counts.get(shape, 0) + 1
                except:
                    pass
            
            for shape, count in sorted(shape_counts.items()):
                if shape == min_shape:
                    light_processing['binning_breakdown'].append(f"{count} frames @ {shape[1]}x{shape[0]} - No binning")
                else:
                    light_processing['binning_breakdown'].append(f"{count} frames @ {shape[1]}x{shape[0]} - Bin to {min_shape_str}")
            
            # Processing sequence
            sequence = [
                "Load frame",
                f"Bin if necessary (to {min_shape_str})",
            ]
            # A master dark ALREADY contains the bias pedestal, so subtracting
            # the dark removes bias and dark current together and the bias is
            # NOT subtracted from the light as well. This list used to say it
            # was -- a description of a double subtraction the code has never
            # done (pinned by tests/test_calibration_output.py).
            if scale_dark:
                sequence.append("Subtract Dark, scaled by exp time "
                                "(removes bias + dark current together)")
            else:
                sequence.append("Subtract Dark "
                                "(removes bias + dark current together)")

            if not skip_bias:
                sequence.append("Bias used to scale the dark and as the flat "
                                "pedestal, not subtracted again")
            
            if flats and flats.files:
                sequence.append("Divide by Flat")
            else:
                sequence.append("Apply Auto-Flat division")
            
            if remove_cosmics:
                sequence.append(f"Remove cosmic rays (σ: {oc_sigma}, radius: {oc_radius}px)")
            
            if hpm and hpm.files:
                sequence.append("Apply hot pixel mask interpolation")
            
            if bin_factor > 1:
                sequence.append(f"Apply output binning ({bin_factor}x{bin_factor})")
            
            sequence.append("Save calibrated frame with history")
            
            light_processing['sequence'] = sequence
        
        # Output settings - format bin_factor into display string
        if bin_factor == -1:
            bin_str = "Auto (shape-matched)"
        elif bin_factor == 1:
            bin_str = "1x1 (None)"
        else:
            bin_str = f"{bin_factor}x{bin_factor}"
        
        output = {
            'binning': bin_str,
            'directory': out_dir,
            'format': 'FITS 16-bit',
            'preserve_headers': 'Yes',
            'add_history': 'Yes',
        }
        
        return {
            'inventory': inventory,
            'processing': processing,
            'calibration': calibration,
            'light_processing': light_processing,
            'output': output,
        }

    def on_progress(self, frac, msg):
        self.progress_dlg.setValue(int(frac*100))
        self.progress_dlg.setLabelText(msg)
        self.view.lbl_status.setText(msg)

    def on_finished(self, success, msg):
        self.view.btn_start.setEnabled(True)
        self.progress_dlg.close()
        if success:
            # A run that silently stored thousands of pixels as 0 is not a
            # plain "Success": give it the warning icon so the text is read.
            if getattr(self.worker, 'clamp_report', None):
                QMessageBox.warning(self.view, "Completed With Warnings", msg)
                self.view.lbl_status.setText("Done -- with clamped pixels")
            else:
                QMessageBox.information(self.view, "Success", msg)
                self.view.lbl_status.setText("Done")
        else:
            QMessageBox.critical(self.view, "Error", f"Calibration failed: {msg}")
            self.view.lbl_status.setText("Error")
