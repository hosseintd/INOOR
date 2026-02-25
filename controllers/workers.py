import numpy as np
from PyQt5 import QtCore
from core import masterFrame_creator as mfc
from core import calibration as calib
import os
from astropy.io import fits

# ---------- Histogram Worker ----------
class HistogramWorker(QtCore.QThread):
    histogram_ready = QtCore.pyqtSignal(int, object, object, object)
    progress = QtCore.pyqtSignal(float)
    done = QtCore.pyqtSignal()

    def __init__(self, file_list, sample_max_pixels=500_000, bins=2000, preview_max_dim=1024, parent=None):
        super().__init__(parent=parent)
        self.file_list = file_list
        self.sample_max_pixels = sample_max_pixels
        self.bins = bins
        self.preview_max_dim = preview_max_dim
        self._is_killed = False

    def make_preview(self, img):
        ny, nx = img.shape
        max_dim = max(ny, nx)
        if max_dim <= self.preview_max_dim:
            return img.astype('float32')
        step = int(np.ceil(max_dim / self.preview_max_dim))
        preview = img[::step, ::step]
        return preview.astype('float32')

    def run(self):
        n = len(self.file_list)
        for i, path in enumerate(self.file_list):
            if self._is_killed:
                break
            try:
                img = mfc.load_fits(path)
                preview = self.make_preview(img)
                arr = img.ravel()
                arr = arr[np.isfinite(arr)]
                if arr.size > self.sample_max_pixels:
                    step = arr.size // self.sample_max_pixels
                    arr_sample = arr[::step]
                else:
                    arr_sample = arr
                counts, bin_edges = np.histogram(arr_sample, bins=self.bins)
                self.histogram_ready.emit(i, counts, bin_edges, preview)
            except Exception:
                self.histogram_ready.emit(i, np.array([]), np.array([]), None)
            self.progress.emit((i + 1) / max(1, n))
        self.done.emit()

    def kill(self):
        self._is_killed = True

# ---------- Create Master Worker ----------
class CreateMasterWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(float, str)
    finished = QtCore.pyqtSignal(bool, object, object, str) # success, data_or_path, header, msg
    
    def __init__(self, out_file, file_list, method, do_clip, lower, upper, excludes, kernel=5, auto_sigma=False):
        super().__init__()
        self.out_file = out_file # If None, will return (data, header)
        self.file_list = file_list
        self.method = method
        self.do_clip = do_clip
        self.lower = lower
        self.upper = upper
        self.excludes = excludes
        self.kernel = kernel
        self.auto_sigma = auto_sigma
        
    def run(self):
        try:
            def progress_cb(fraction, msg):
                if self.isInterruptionRequested():
                    raise Exception("Cancelled")
                self.progress.emit(fraction, msg)

            result = mfc.create_master(self.out_file, self.file_list, method=self.method,
                                      do_sigma_clip=self.do_clip, sigma_lower=self.lower,
                                      sigma_upper=self.upper, exclude_indices=self.excludes,
                                      progress_callback=progress_cb, kernel=self.kernel,
                                      auto_sigma=self.auto_sigma)
            
            if self.out_file:
                # result is just the path
                self.finished.emit(True, result, None, "Master created successfully.")
            else:
                # result is (master_array, header)
                data, hdr = result
                self.finished.emit(True, data, hdr, "Master ready for review.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, None, None, str(e))

# ---------- Fit Worker ----------
class FitWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(float, str)
    done = QtCore.pyqtSignal(object, object, object)
    failed = QtCore.pyqtSignal(str)
    
    def __init__(self, master_array, degree=2, max_samples=500_000):
        super().__init__()
        self.master_array = master_array
        self.degree = degree
        self.max_samples = max_samples
        
    def run(self):
        try:
            def progress_cb(frac, msg):
                if self.isInterruptionRequested():
                    raise Exception("Cancelled")
                self.progress.emit(frac, msg)
            z_fitted, gain_table, coeffs = mfc.create_gain_table_from_master(
                self.master_array, degree=self.degree, fit_max_samples=self.max_samples, progress_callback=progress_cb)
            self.done.emit(z_fitted, gain_table, coeffs)
        except Exception as e:
            self.failed.emit(str(e))

# ---------- Calibration Worker ----------
class CalibrationWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(float, str)
    finished = QtCore.pyqtSignal(bool, str)

    def __init__(self, sets, output_dir, auto_flat=False, auto_flat_params=None, skip_bias=False, auto_bin=False, 
                 remove_cosmics=False, bin_factor=1, scale_dark=True, outlier_sigma=5.0, outlier_radius=1, hot_mask_path=None, metadata_cache=None):
        super().__init__()
        self.sets = sets # List of FileSet objects or dicts
        self.output_dir = output_dir
        self.auto_flat = bool(auto_flat)
        self.auto_flat_params = auto_flat_params or {}
        self.skip_bias = bool(skip_bias)
        self.auto_bin = bool(auto_bin)
        self.remove_cosmics = bool(remove_cosmics)
        self.bin_factor = bin_factor # -1 for auto smart-matching, else fixed factor
        self.scale_dark = bool(scale_dark)
        self.outlier_sigma = outlier_sigma
        self.outlier_radius = outlier_radius
        self.hot_mask_path = hot_mask_path
        self.metadata_cache = metadata_cache or {}
        self.target_shape = None
        self.fm_auto = None # Cache for auto-generated flat
        self.shape_match_log = []  # Log of shape matching decisions

    def bin_image(self, img, target_shape):
        """Downsample image to match target_shape (ny, nx) by pixel averaging."""
        ny_tgt, nx_tgt = target_shape
        ny_src, nx_src = img.shape
        
        fy = ny_src // ny_tgt
        fx = nx_src // nx_tgt
        
        factor = min(fx, fy)
        if factor <= 1:
            # If they don't match by integer factor, just crop/pad?
            # For now, if no integer factor, return as is (safeguard handled later)
            if (ny_src, nx_src) == (ny_tgt, nx_tgt): return img
            return img[:ny_tgt, :nx_tgt] # Simple crop
            
        # 2D binning
        img_view = img[:ny_tgt*factor, :nx_tgt*factor].reshape(ny_tgt, factor, nx_tgt, factor)
        return img_view.mean(axis=(1, 3))

    def run(self):
        try:
            # Grouping sets by type
            type_map = {}
            all_files = []
            for s in self.sets:
                t = s.set_type if hasattr(s, 'set_type') else s.get('type')
                type_map.setdefault(t, []).append(s)
                all_files.extend(s.files if hasattr(s, 'files') else s.get('files', []))

            # Phase 1: Detect MINIMUM shape across ALL sets (not per-set)
            self.progress.emit(0.01, "Detecting frame shapes...")
            min_shape = self._detect_minimum_shape(all_files)
            min_shape_str = f"{min_shape[1]}x{min_shape[0]}"
            print(f"[CALIB] Target size (minimum): {min_shape_str}")
            self.progress.emit(0.02, f"Target resolution: {min_shape_str}")

            def merge_files_for(typ):
                items = type_map.get(typ, [])
                files = []
                bads = set()
                method = 'median'
                do_clip = False
                l, u = 3.0, 3.0
                current_offset = 0
                for it in items:
                    it_files = it.files if hasattr(it, 'files') else it.get('files', [])
                    it_bads = it.bad_indices if hasattr(it, 'bad_indices') else it.get('bad_indices', set())
                    it_method = it.method if hasattr(it, 'method') else it.get('method', method)
                    it_clip = it.do_sigma_clip if hasattr(it, 'do_sigma_clip') else it.get('do_sigma_clip', do_clip)
                    it_l = it.sigma_lower if hasattr(it, 'sigma_lower') else it.get('sigma_lower', l)
                    it_u = it.sigma_upper if hasattr(it, 'sigma_upper') else it.get('sigma_upper', u)
                    
                    files.extend(it_files)
                    for bi in it_bads:
                        bads.add(bi + current_offset)
                    
                    current_offset += len(it_files)
                    method = it_method
                    do_clip = it_clip
                    l = it_l
                    u = it_u
                return files, bads, method, do_clip, l, u

            flats, flat_bads, flat_method, flat_clip, flat_l, flat_u = merge_files_for('Flat')
            darks, dark_bads, dark_method, dark_clip, dark_l, dark_u = merge_files_for('Dark')
            biases, bias_bads, bias_method, bias_clip, bias_l, bias_u = merge_files_for('Bias')
            lights, light_bads, light_method, light_clip, light_l, light_u = merge_files_for('Light')
            
            # Helper to get EXPTIME from first valid file in a set
            def get_set_exptime(files, bads):
                for i, f in enumerate(files):
                    if i not in bads:
                        try:
                            hdr = fits.getheader(f)
                            return mfc.get_exptime(hdr)
                        except: pass
                return 1.0

            dark_et = get_set_exptime(darks, dark_bads)
            if darks and dark_et == 1.0:
                 try:
                     h0 = fits.getheader([f for i,f in enumerate(darks) if i not in dark_bads][0])
                     if 'EXPTIME' not in h0 and 'EXP_TIME' not in h0 and 'EXPOSURE' not in h0:
                         print("[CALIB] Warning: Dark frames missing EXPTIME keyword. Scaling may be inaccurate.")
                 except: pass
            
            gain_sets = type_map.get('GainTable', [])
            gain_arr = None
            if gain_sets:
                gfiles = gain_sets[0].files if hasattr(gain_sets[0], 'files') else gain_sets[0].get('files', [])
                if gfiles:
                    gain_arr = mfc.load_fits(gfiles[0])
                    # Bin gain table if needed
                    if gain_arr.shape != min_shape:
                        print(f"[CALIB] Binning GainTable {gain_arr.shape} -> {min_shape}")
                        gain_arr = self.bin_image(gain_arr, min_shape)

            def make_master_or_none(files, bads, method, do_clip, l, u, role_name):
                """Create master frame - bin individual frames BEFORE stacking."""
                if not files: return None, []
                
                self.progress.emit(0.05, f"Creating {role_name} master (binning if needed)...")
                
                valid_files = [f for i,f in enumerate(files) if i not in bads]
                valid_basenames = [os.path.basename(f) for f in valid_files]
                if not valid_files: return None, []
                
                # Check if first file needs binning
                try:
                    hdr = fits.getheader(valid_files[0])
                    ny = hdr.get('NAXIS2', 0)
                    nx = hdr.get('NAXIS1', 0)
                    file_shape = (ny, nx)
                except:
                    file_shape = min_shape
                
                if file_shape != min_shape:
                    print(f"[CALIB] Binning {role_name} frames {file_shape} -> {min_shape} before stacking...")
                    stack = []
                    for f in valid_files:
                        img = mfc.load_fits(f)
                        if do_clip:
                            img = mfc.sigclip(img, l, u)
                        # Bin BEFORE stacking
                        if img.shape != min_shape:
                            img = self.bin_image(img, min_shape)
                        stack.append(img)
                    if method == 'median': arr = np.median(stack, axis=0)
                    else: arr = np.mean(stack, axis=0)
                else:
                    arr = calib.create_master_from_list(valid_files, method=method, do_sigma_clip=do_clip,
                                                        lower_sigma=l, upper_sigma=u, exclude_indices=bads)
                return arr, valid_basenames

            bias_master, bias_files = make_master_or_none(biases, bias_bads, bias_method, bias_clip, bias_l, bias_u, "Bias")
            dark_master, dark_files = make_master_or_none(darks, dark_bads, dark_method, dark_clip, dark_l, dark_u, "Dark")
            
            flat_master, flat_files = None, []
            if gain_arr is None and flats:
                flat_master, flat_files = make_master_or_none(flats, flat_bads, flat_method, flat_clip, flat_l, flat_u, "Flat")

            if not lights: raise RuntimeError("No Light frames.")

            # Load Hot Pixel Mask if provided
            hp_mask = None
            if self.hot_mask_path and os.path.exists(self.hot_mask_path):
                try:
                    hp_mask = mfc.load_fits(self.hot_mask_path).astype('float32')
                    # Bin hot pixel mask if needed
                    if hp_mask.shape != min_shape:
                        print(f"[CALIB] Binning HotPixelMap {hp_mask.shape} -> {min_shape}")
                        hp_mask = self.bin_image(hp_mask, min_shape)
                except Exception as e:
                    print(f"[CALIB] Error loading HotPixelMap: {self.hot_mask_path} - {e}")

            total_lights = len([i for i in range(len(lights)) if i not in light_bads])
            processed_count = 0

            for i, lfpath in enumerate(lights):
                if self.isInterruptionRequested(): break
                if i in light_bads: continue
                
                processed_count += 1
                hdr_l = fits.getheader(lfpath)
                light_et = mfc.get_exptime(hdr_l)
                ny_light = hdr_l.get('NAXIS2', 0)
                nx_light = hdr_l.get('NAXIS1', 0)
                light_shape = (ny_light, nx_light)
                
                print(f"[CALIB] [{processed_count}/{total_lights}] {os.path.basename(lfpath)} | Shape: {nx_light}x{ny_light}")
                
                progress_frac = 0.1 + 0.85 * (processed_count - 1) / max(1, total_lights)
                self.progress.emit(progress_frac, f"[{processed_count}/{total_lights}] Calibrating {os.path.basename(lfpath)}...")
                
                lf = mfc.load_fits(lfpath)
                autobinned = False
                
                # Bin light frame if needed
                if light_shape != min_shape:
                    print(f"[CALIB]   Binning light frame {light_shape} -> {min_shape}")
                    lf = self.bin_image(lf, min_shape)
                    autobinned = True
                
                ny_l, nx_l = lf.shape
                
                # Masters should already be at min_shape, so no matching needed
                bm = bias_master
                dm = dark_master
                
                if bm is None: bm = np.zeros_like(lf)
                if dm is None: dm = np.zeros_like(lf)
                
                # Dark scaling
                if self.scale_dark and darks:
                    if dark_et <= 0:
                        print("[CALIB]   Warning: Dark exposure time is 0 or unknown. Skipping scaling.")
                        scaling = 1.0
                    else:
                        scaling = float(light_et) / float(dark_et)
                    
                    if abs(scaling - 1.0) > 0.001:
                        print(f"[CALIB]   Scaling Dark by {scaling:.4f}x")
                        if self.skip_bias:
                            dm = dm * scaling
                        else:
                            dm = (dm - bm) * scaling + bm
                
                # Calibration
                if self.skip_bias:
                    calibrated = (lf.astype('float32') - dm.astype('float32'))
                    
                    if gain_arr is not None:
                        ga = gain_arr if gain_arr.shape == (ny_l, nx_l) else np.ones_like(lf)
                        calibrated *= ga
                    elif flat_master is not None:
                        fm = flat_master if flat_master.shape == (ny_l, nx_l) else np.ones_like(lf)
                        calibrated /= (fm / np.mean(fm))
                    else:
                        if self.fm_auto is None:
                            self.progress.emit(progress_frac, "Creating Auto-Flat...")
                            auto = calib.create_auto_flat_from_light(lf, master_dark=dm, master_bias=bm)
                            self.fm_auto = auto['model']
                        calibrated /= (self.fm_auto / np.mean(self.fm_auto))
                    np.clip(calibrated, 0, None, out=calibrated)
                else:
                    if gain_arr is not None:
                        ga = gain_arr if gain_arr.shape == (ny_l, nx_l) else np.ones_like(lf)
                        calibrated = calib.calibrate_by_gaintable(lf, dm, bm, ga)
                    else:
                        fm = flat_master
                        if fm is None:
                            if self.fm_auto is None:
                                self.progress.emit(progress_frac, "Creating Auto-Flat...")
                                auto = calib.create_auto_flat_from_light(lf, master_dark=dm, master_bias=bm)
                                self.fm_auto = auto['model']
                            fm = self.fm_auto
                        calibrated = calib.calibrate(lf, dm, bm, fm)

                # POST-PROCESSING
                if self.remove_cosmics:
                    print(f"[CALIB]   Removing cosmic rays...")
                    bsize = self.outlier_radius * 2 + 1
                    calibrated = calib.remove_outliers(calibrated, sigma=self.outlier_sigma, box_size=bsize)
                
                # Hot Pixel Mask
                if hp_mask is not None:
                    if hp_mask.shape == (ny_l, nx_l):
                        print(f"[CALIB]   Applying hot pixel mask...")
                        hpm_binary = (hp_mask > 0.5).astype('float32')
                        calibrated = calib.apply_hot_pixel_mask(calibrated, hpm_binary, box_size=3)
                    else:
                        print(f"[CALIB]   Warning: Hot pixel mask shape mismatch. Skipping.")
                
                # Output binning (only if user selected > 1)
                applied_bin = self.bin_factor
                if applied_bin > 1:
                    print(f"[CALIB]   Applying output binning {applied_bin}x{applied_bin}...")
                    calibrated = calib.bin_ndarray(calibrated, applied_bin)

                # Save with enhanced header
                outname = os.path.basename(lfpath)
                outpath = os.path.join(self.output_dir, f"Calibrated_{outname}")
                orig_hdr = fits.getheader(lfpath) if os.path.exists(lfpath) else None
                
                if orig_hdr is not None:
                    orig_hdr = orig_hdr.copy()
                    
                    # Add comprehensive calibration history
                    orig_hdr.add_history("===== CALIBRATION HISTORY =====")
                    orig_hdr.add_history(f"Calibration Date: {self._get_timestamp()}")
                    orig_hdr.add_history("")
                    
                    if autobinned:
                        orig_hdr.add_history(f"Pre-calibration binning: {light_shape[1]}x{light_shape[0]} -> {min_shape_str}")
                        orig_hdr.add_history("Binning reason: Match calibration frames (minimum size)")
                        orig_hdr.add_history("")
                    
                    # Master files used
                    if dark_files:
                        orig_hdr.add_history(f"Dark Master: {len(dark_files)} frames (Method: {dark_method}, Sigma-clip: {dark_method != 'mean'})")
                        for dfile in dark_files:
                            orig_hdr.add_history(f"  {dfile}")
                    
                    if flat_files:
                        orig_hdr.add_history(f"Flat Master: {len(flat_files)} frames (Method: {flat_method}, Sigma-clip: {flat_method != 'mean'})")
                        for ffile in flat_files:
                            orig_hdr.add_history(f"  {ffile}")
                    
                    if bias_files:
                        orig_hdr.add_history(f"Bias Master: {len(bias_files)} frames (Method: {bias_method}, Sigma-clip: {bias_method != 'mean'})")
                        for bfile in bias_files:
                            orig_hdr.add_history(f"  {bfile}")
                    
                    if gain_arr is not None:
                        orig_hdr.add_history("Flat field: Applied via Gain Table")
                    
                    orig_hdr.add_history("")
                    
                    # Calibration operations
                    if self.scale_dark and darks:
                        orig_hdr.add_history(f"Dark Scaling: ENABLED (Light exp: {light_et}s, Dark exp: {dark_et}s)")
                    else:
                        orig_hdr.add_history("Dark Scaling: DISABLED")
                    
                    if self.remove_cosmics:
                        orig_hdr.add_history(f"Cosmic Ray Removal: ENABLED (sigma: {self.outlier_sigma}, radius: {self.outlier_radius}px)")
                    
                    if hp_mask is not None and hp_mask.shape == (ny_l, nx_l):
                        orig_hdr.add_history("Hot Pixel Mask: APPLIED")
                    
                    if applied_bin > 1:
                        orig_hdr.add_history(f"Output Binning: {applied_bin}x{applied_bin}")
                    
                    orig_hdr.add_history("===== END CALIBRATION HISTORY =====")
                    
                calib.save_fits_16bit(outpath, calibrated, header=orig_hdr, clip_max=4095, overwrite=True)

            print("[CALIB] Calibration complete!")
            self.finished.emit(True, "Calibration complete")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e))

    def _detect_minimum_shape(self, all_files):
        """Detect the minimum shape across all files."""
        min_ny, min_nx = 1e9, 1e9
        for f in all_files:
            try:
                hdr = fits.getheader(f)
                min_ny = min(min_ny, hdr.get('NAXIS2', 1e9))
                min_nx = min(min_nx, hdr.get('NAXIS1', 1e9))
            except:
                pass
        return (int(min_ny), int(min_nx))

    def _get_timestamp(self):
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat(timespec='seconds')
