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
from views.calibration_view import CalibrationView
from views.components.review_dialog import ReviewDialog
from views.components.calibration_summary_dialog import CalibrationSummaryDialog
from core import masterFrame_creator as mfc
from controllers.workers import CalibrationWorker

class CalibrationController(QObject):
    def __init__(self, model, view: CalibrationView):
        super().__init__()
        self.model = model
        self.view = view
        
        self.current_img_data = None
        self.metadata_cache = {} # path -> {XBINNING, NAXIS1, CCD-TEMP, EXPTIME}
        self.auto_binning = False
        
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
                    except:
                        pass

    def _validate_frame_shapes(self, sets):
        """Validate that all frames in each set have the same shape. Return True if valid."""
        for s in sets:
            if not s.files:
                continue
            
            shapes = set()
            for f in s.files:
                try:
                    hdr = fits.getheader(f)
                    ny = hdr.get('NAXIS2', 0)
                    nx = hdr.get('NAXIS1', 0)
                    shapes.add((ny, nx))
                except:
                    pass
            
            if len(shapes) > 1:
                QMessageBox.critical(self.view, "Shape Mismatch in Frame Set",
                                   f"All frames in {s.set_type} set must have the same shape.\n"
                                   f"Detected shapes: {sorted(list(shapes))}\n\n"
                                   f"Please ensure all {s.set_type} frames are taken at the same binning.\n"
                                   f"You can load frames of different shapes in separate sets.")
                return False
        
        return True

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
                except:
                    pass
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
                    path = s.files[index]
                    try:
                        img = mfc.load_fits(path)
                        self.current_img_data = img
                        self.view.canvas_image.show_image(img)
                        self.view.lbl_fname.setText(os.path.basename(path))
                        
                        # Hist (Speed optimized deterministic sampling)
                        arr = img.ravel()
                        arr = arr[np.isfinite(arr)]
                        sample_size = 200000
                        if arr.size > sample_size:
                            step = arr.size // sample_size
                            sample = arr[::step]
                        else:
                            sample = arr
                        counts, bins = np.histogram(sample, bins=2000)
                        self.view.canvas_hist.show_hist(counts, bins)
                        
                        # Set initial ZScale limits for visual lines
                        vmin, vmax = mfc.zscale_limits(img)
                        self.view.canvas_hist.set_limits(vmin, vmax)
                    except Exception as e:
                        print(f"Preview error: {e}")
                break
    
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
    def start_calibration(self):
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
        
        # VALIDATE: Check mutual exclusivity of Flat vs GainTable
        has_flats = any(s.set_type == 'Flat' for s in sets)
        has_gain = any(s.set_type == 'GainTable' for s in sets)
        if has_flats and has_gain:
            QMessageBox.critical(self.view, "Conflicting Calibration Frames",
                               "You cannot use both Flat frames and Gain Table simultaneously.\n"
                               "Please remove either all Flat frames or the Gain Table, then try again.")
            return
        
        # VALIDATE: Check shape uniformity per set
        if not self._validate_frame_shapes(sets):
            return
        
        out_dir = QFileDialog.getExistingDirectory(self.view, "Select Output Directory")
        if not out_dir: return
        
        has_bias = any(s.set_type == 'Bias' for s in sets)
        
        allow_auto = False
        if not has_flats and not has_gain:
            res = QMessageBox.question(self.view, "No Flats", 
                                     "No Flat frames or GainTable provided. Create auto-flats per Light?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if res == QMessageBox.Yes:
                allow_auto = True
            else: return

        skip_bias = False
        if not has_bias:
            res = QMessageBox.question(self.view, "No Bias",
                                     "No Bias frames provided. Do you want to calibrate without bias (subtract Dark only)?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if res == QMessageBox.Yes:
                skip_bias = True
            else:
                return

        # Post-processing options
        scale_dark = self.view.chk_scale_dark.isChecked() if has_darks else False
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
        gain = next((s for s in sets if s.set_type == 'GainTable'), None)
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
        inventory['gain_summary'] = "Yes" if gain and gain.files else "None"
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
        
        # Calibration settings
        calibration = {
            'dark_scaling': "Yes" if scale_dark else "No",
            'skip_bias': "Yes" if skip_bias else "No",
            'auto_flat': "Yes" if allow_auto else "No",
            'use_gain': "Yes" if gain and gain.files else "No",
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
            if scale_dark:
                sequence.append("Subtract Dark (scaled by exp time)")
            else:
                sequence.append("Subtract Dark")
            
            if not skip_bias:
                sequence.append("Subtract Bias")
            
            if flats and flats.files:
                sequence.append("Divide by Flat")
            elif gain and gain.files:
                sequence.append("Multiply by Gain Table")
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
            QMessageBox.information(self.view, "Success", msg)
            self.view.lbl_status.setText("Done")
        else:
            QMessageBox.critical(self.view, "Error", f"Calibration failed: {msg}")
            self.view.lbl_status.setText("Error")
