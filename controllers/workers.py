import numpy as np
from PyQt5 import QtCore
from core import masterFrame_creator as mfc
from core import calibration as calib
import os
from astropy.io import fits

from utils.exptime_utils import get_exptime_seconds
from utils.logging_config import get_logger

log = get_logger(__name__)

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

# ---------- Image Preload Worker ----------
class ImagePreloadWorker(QtCore.QThread):
    """Warm an ImageDisplayCache in the background.

    Clicking a frame otherwise costs a full 4096x4096 read plus a ZScale pass
    ON THE GUI THREAD, freezing the UI for about a second per click. Loading
    once up front makes switching between frames instant.
    """
    progress = QtCore.pyqtSignal(int)
    done = QtCore.pyqtSignal()

    def __init__(self, files, cache, parent=None):
        super().__init__(parent=parent)
        self.files = list(files)
        self.cache = cache

    def run(self):
        for i, p in enumerate(self.files):
            if self.isInterruptionRequested():
                break
            try:
                self.cache.get_display_data(p, mfc.load_fits(p))
            except Exception as e:
                log.warning("Preload failed for %s: %s", p, e)
            self.progress.emit(i + 1)
        self.done.emit()


# ---------- Create Master Worker ----------
class CreateMasterWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(float, str)
    finished = QtCore.pyqtSignal(bool, object, object, str) # success, data_or_path, header, msg
    
    def __init__(self, out_file, file_list, method, do_clip, lower, upper, excludes, kernel=5, auto_sigma=False,
                 scale_levels=False):
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
        self.scale_levels = scale_levels
        
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
                                      auto_sigma=self.auto_sigma,
                                      scale_to_common_level=self.scale_levels)
            
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

# ---------- Calibration Worker ----------

# Progress budget for CalibrationWorker. Master creation loads and stacks every
# calibration frame, so it gets a real slice of the bar rather than a single
# frozen tick; the light-frame loop owns the rest.
SHAPE_PHASE_END = 0.03
MASTER_PHASE_START = SHAPE_PHASE_END
MASTER_PHASE_END = 0.45
LIGHT_PHASE_START = MASTER_PHASE_END
LIGHT_PHASE_END = 0.99


class _Cancelled(Exception):
    """Raised inside a progress callback when the user cancels."""


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
        # No auto-flat cache: the surface is refitted per light frame.
        self.shape_match_log = []  # Log of shape matching decisions
        self.clamp_report = []     # (filename, n_below_zero, n_nonfinite, n_pixels)
        self._float_trigger = None  # (filename, z) of the frame that made the run float

    def bin_image(self, img, target_shape):
        """Downsample image to match target_shape (ny, nx) by pixel averaging.

        Each axis is binned by its OWN integer factor. The old code applied
        min(fx, fy) to both axes, so an anisotropic mismatch left one axis
        under-binned and the frame was then silently cropped to a subregion
        of the field of view.
        """
        ny_tgt, nx_tgt = target_shape
        ny_src, nx_src = img.shape

        fy = max(1, ny_src // ny_tgt)
        fx = max(1, nx_src // nx_tgt)

        if fy == 1 and fx == 1:
            if (ny_src, nx_src) == (ny_tgt, nx_tgt):
                return img
            # No integer binning possible: crop (and warn -- data is lost)
            print(f"[CALIB] Warning: shape {img.shape} not an integer multiple "
                  f"of target {target_shape}; cropping edges.")
            return img[:ny_tgt, :nx_tgt]

        if fy != fx:
            print(f"[CALIB] Warning: anisotropic binning factors (fy={fy}, fx={fx}) "
                  f"for {img.shape} -> {target_shape}.")

        out = img[:ny_tgt * fy, :nx_tgt * fx].reshape(ny_tgt, fy, nx_tgt, fx).mean(axis=(1, 3))
        return out

    def run(self):
        try:
            # Grouping sets by type
            type_map = {}
            shape_files = []
            for s in self.sets:
                t = s.set_type if hasattr(s, 'set_type') else s.get('type')
                type_map.setdefault(t, []).append(s)
                # HotPixelMap is an auxiliary product: it gets RESAMPLED to the
                # target shape, it must not DEFINE it. A mask saved at a
                # different size used to drag every science frame down to its
                # resolution.
                if t in ('Light', 'Dark', 'Bias', 'Flat'):
                    shape_files.extend(s.files if hasattr(s, 'files') else s.get('files', []))

            # Phase 1: Detect MINIMUM shape across the science/calibration sets
            self.progress.emit(0.01, "Detecting frame shapes...")
            min_shape = self._detect_minimum_shape(shape_files)
            min_shape_str = f"{min_shape[1]}x{min_shape[0]}"
            print(f"[CALIB] Target size (minimum): {min_shape_str}")
            self.progress.emit(SHAPE_PHASE_END, f"Target resolution: {min_shape_str}")

            def merge_files_for(typ):
                """Flatten every set of this type into one file list.

                Stacking parameters come from the FIRST set; previously the
                loop overwrote them on each pass, so with several sets of the
                same type the LAST set silently decided the method and clipping
                for all of them. Disagreements are logged rather than hidden.
                """
                items = type_map.get(typ, [])
                files = []
                bads = set()
                params = None
                current_offset = 0
                for it in items:
                    it_files = it.files if hasattr(it, 'files') else it.get('files', [])
                    it_bads = it.bad_indices if hasattr(it, 'bad_indices') else it.get('bad_indices', set())
                    it_params = (
                        it.method if hasattr(it, 'method') else it.get('method', 'median'),
                        it.do_sigma_clip if hasattr(it, 'do_sigma_clip') else it.get('do_sigma_clip', False),
                        it.sigma_lower if hasattr(it, 'sigma_lower') else it.get('sigma_lower', 3.0),
                        it.sigma_upper if hasattr(it, 'sigma_upper') else it.get('sigma_upper', 3.0),
                    )

                    files.extend(it_files)
                    for bi in it_bads:
                        bads.add(bi + current_offset)
                    current_offset += len(it_files)

                    if params is None:
                        params = it_params
                    elif it_params != params:
                        log.warning("%s sets disagree on stacking parameters %s vs %s; "
                                    "using the first set's.", typ, params, it_params)

                if params is None:
                    params = ('median', False, 3.0, 3.0)
                method, do_clip, l, u = params
                return files, bads, method, do_clip, l, u

            flats, flat_bads, flat_method, flat_clip, flat_l, flat_u = merge_files_for('Flat')
            darks, dark_bads, dark_method, dark_clip, dark_l, dark_u = merge_files_for('Dark')
            biases, bias_bads, bias_method, bias_clip, bias_l, bias_u = merge_files_for('Bias')
            lights, light_bads, light_method, light_clip, light_l, light_u = merge_files_for('Light')
            
            def get_set_exptime(files, bads):
                """Exposure time of a whole set, read from EVERY frame in it.

                This used to return on the FIRST readable frame, so a set with
                mixed exposures silently scaled by whichever file happened to
                come first, and a set with no EXPTIME at all fell back to 1.0 s
                with nothing said. Returns (seconds or None, note).
                """
                seen = []
                for i, f in enumerate(files):
                    if i in bads:
                        continue
                    try:
                        et = get_exptime_seconds(fits.getheader(f), default=None)
                    except Exception as e:
                        log.warning("Cannot read header of %s: %s", f, e)
                        continue
                    if et is not None:
                        seen.append(float(et))
                if not seen:
                    return None, "no frame carries an exposure time"
                lo, hi = min(seen), max(seen)
                if hi - lo > 1e-3:
                    return (float(np.median(seen)),
                            f"mixed exposures {lo:g}-{hi:g}s; using the median")
                return lo, ""

            dark_et, dark_et_note = get_set_exptime(darks, dark_bads)
            if darks and dark_et_note:
                log.warning("Dark set: %s", dark_et_note)
                self.progress.emit(0.0, f"Warning -- dark set: {dark_et_note}")
            if darks and dark_et is None and self.scale_dark:
                raise ValueError(
                    "Dark scaling was requested but no dark frame carries an "
                    "exposure time. Turn dark scaling off, or use darks whose "
                    "headers have EXPTIME.")
            if dark_et is None:
                dark_et = 0.0

            # ---- Progress budget -------------------------------------------
            # Master creation is the slowest phase but used to sit at a frozen
            # 5%. Give the masters a real share of the bar, weighted by how
            # many frames each one has to load and stack.
            def _valid(files, bads):
                return [f for i, f in enumerate(files) if i not in bads]

            master_jobs = [
                ('Bias', _valid(biases, bias_bads), bias_method, bias_clip, bias_l, bias_u),
                ('Dark', _valid(darks, dark_bads), dark_method, dark_clip, dark_l, dark_u),
            ]
            if flats:
                master_jobs.append(('Flat', _valid(flats, flat_bads), flat_method,
                                    flat_clip, flat_l, flat_u))

            total_master_frames = sum(len(j[1]) for j in master_jobs)
            master_share = (MASTER_PHASE_END - MASTER_PHASE_START) if total_master_frames else 0.0
            master_cursor = MASTER_PHASE_START

            def make_master_or_none(role_name, valid_files, method, do_clip, l, u, span,
                                    per_frame=None):
                """Create master frame via the single canonical stacking path.

                Frames whose shape differs from min_shape are binned per-frame
                (inside the same create_master pipeline) so the combination and
                sigma-clipping algorithm is IDENTICAL whether or not binning
                was needed -- the algorithm must never depend on frame shape.

                `per_frame` runs after binning, on every frame, before the
                combination: the flat path uses it to remove the pedestal and
                scale each frame to unity.
                """
                if not valid_files:
                    return None, []

                valid_basenames = [os.path.basename(f) for f in valid_files]

                def transform(img):
                    if img.shape != min_shape:
                        img = self.bin_image(img, min_shape)
                    if per_frame is not None:
                        img = per_frame(img)
                    return img

                def sub_progress(frac, msg):
                    # Cancel now works during master creation too; previously
                    # interruption was only checked in the light-frame loop, so
                    # Cancel did nothing during the slowest phase.
                    if self.isInterruptionRequested():
                        raise _Cancelled()
                    frac = min(max(float(frac), 0.0), 1.0)
                    self.progress.emit(master_cursor + span * frac,
                                       f"{role_name} master: {msg}")

                arr = calib.create_master_from_list(valid_files, method=method, do_sigma_clip=do_clip,
                                                    lower_sigma=l, upper_sigma=u,
                                                    progress_callback=sub_progress,
                                                    frame_transform=transform)
                return arr, valid_basenames

            masters = {}
            spans = {}
            for role_name, valid_files, method, do_clip, l, u in master_jobs:
                spans[role_name] = (master_share * (len(valid_files) / total_master_frames)
                                    if total_master_frames else 0.0)

            # Bias and Dark must exist BEFORE the flat is combined: the flat
            # needs one of them as its pedestal.
            for role_name, valid_files, method, do_clip, l, u in master_jobs:
                if role_name == 'Flat':
                    continue
                masters[role_name] = make_master_or_none(role_name, valid_files, method,
                                                         do_clip, l, u, spans[role_name])
                master_cursor += spans[role_name]

            bias_master, bias_files = masters.get('Bias', (None, []))
            dark_master, dark_files = masters.get('Dark', (None, []))

            # ---- Flat pedestal ---------------------------------------------
            # A raw flat is bias + dark_f + F*I. Left in, the bias structure
            # (the KL4040 quadrant seam) is re-imprinted INVERTED on every
            # science frame with amplitude bias_step/flat_level. A master dark
            # already contains the bias, so it works as the pedestal when no
            # bias frames were loaded.
            if bias_master is not None:
                flat_pedestal, flat_pedestal_name = bias_master, "Master Bias"
            elif dark_master is not None:
                flat_pedestal, flat_pedestal_name = dark_master, "Master Dark"
            else:
                flat_pedestal, flat_pedestal_name = None, "none"

            flat_job = next((j for j in master_jobs if j[0] == 'Flat'), None)
            if flat_job is not None:
                _, valid_files, method, do_clip, l, u = flat_job
                if flat_pedestal is None:
                    print("[CALIB] Warning: no Bias and no Dark -- the master flat keeps "
                          "its pedestal and the sensor seam will survive flat-fielding.")
                else:
                    print(f"[CALIB] Flat pedestal: subtracting {flat_pedestal_name} "
                          f"from each flat frame")

                def prep_flat(img):
                    return calib.prepare_flat_frame(img, pedestal=flat_pedestal,
                                                    scale_to_unity=True)

                masters['Flat'] = make_master_or_none('Flat', valid_files, method,
                                                      do_clip, l, u, spans['Flat'],
                                                      per_frame=prep_flat)
                master_cursor += spans['Flat']

            flat_master, flat_files = masters.get('Flat', (None, []))

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

            def calibrate_light(lfpath, processed_count):
                """Calibrate one light frame; returns (outpath, calibrated, header)."""
                hdr_l = fits.getheader(lfpath)
                # default=None, not 1.0: a light frame with no EXPTIME used to
                # be scaled as though it were a 1-second exposure, with no
                # warning anywhere.
                light_et = get_exptime_seconds(hdr_l, default=None)
                if light_et is None:
                    # dark_master, not dm: dm is not assigned until further
                    # down the loop, so this read used to raise
                    # UnboundLocalError on the first such frame and consult the
                    # PREVIOUS frame's dark on any later one.
                    if self.scale_dark and dark_master is not None:
                        raise ValueError(
                            f"{os.path.basename(lfpath)} has no exposure time, so "
                            f"the dark cannot be scaled to it. Turn dark scaling "
                            f"off, or fix the header.")
                    light_et = 0.0
                ny_light = hdr_l.get('NAXIS2', 0)
                nx_light = hdr_l.get('NAXIS1', 0)
                light_shape = (ny_light, nx_light)
                
                print(f"[CALIB] [{processed_count}/{total_lights}] {os.path.basename(lfpath)} | Shape: {nx_light}x{ny_light}")
                
                progress_frac = LIGHT_PHASE_START + (LIGHT_PHASE_END - LIGHT_PHASE_START) * \
                    (processed_count - 1) / max(1, total_lights)
                self.progress.emit(progress_frac, f"[{processed_count}/{total_lights}] Calibrating {os.path.basename(lfpath)}...")
                
                lf = mfc.load_fits(lfpath)
                autobinned = False
                
                # Bin light frame if needed
                if light_shape != min_shape:
                    print(f"[CALIB]   Binning light frame {light_shape} -> {min_shape}")
                    lf = self.bin_image(lf, min_shape)
                    autobinned = True
                
                ny_l, nx_l = lf.shape
                
                # Masters are already at min_shape. Keep them as None when
                # absent rather than substituting zeros: calib.subtract_pedestal
                # needs to know whether a dark exists, because a master dark
                # already carries the bias and a bias-only run must subtract the
                # bias instead.
                bm = None if self.skip_bias else bias_master
                dm = dark_master

                # Dark scaling
                dark_scale_applied = None
                if self.scale_dark and dm is not None:
                    if dark_et <= 0:
                        print("[CALIB]   Warning: Dark exposure time is 0 or unknown. Skipping scaling.")
                        scaling = 1.0
                    else:
                        scaling = float(light_et) / float(dark_et)

                    if abs(scaling - 1.0) > 0.001:
                        dark_scale_applied = scaling
                        print(f"[CALIB]   Scaling Dark by {scaling:.4f}x")
                        if bm is not None:
                            # Only the dark CURRENT scales with exposure time;
                            # the bias pedestal does not. Remove it, scale, put
                            # it back.
                            dm = (dm - bm) * scaling + bm
                        else:
                            # No bias available: the pedestal inside the dark is
                            # scaled too, which injects a bias*(scaling-1) offset.
                            # The user explicitly accepted this in the controller.
                            print("[CALIB]   Warning: no bias -- dark pedestal is being "
                                  "scaled as well (user-confirmed).")
                            dm = dm * scaling

                # Calibration
                used_auto_flat = False
                fm = flat_master
                if fm is not None and fm.shape != (ny_l, nx_l):
                    # This used to set fm = None, which dropped straight
                    # through to auto-flat: the operator's real flats were
                    # silently replaced by a polynomial model of each
                    # frame's own sky, with nothing in the header to say so.
                    raise ValueError(
                        f"Master flat is {fm.shape}, light frame is "
                        f"{(ny_l, nx_l)}. These do not match -- check the "
                        f"binning of the flats and the lights.")
                if fm is None:
                    if not self.auto_flat:
                        # The consent flag existed but was never read.
                        raise ValueError(
                            "No flat field is available for this light set "
                            "and auto-flat was not authorised. Provide flats, "
                            "or re-run and accept the auto-flat prompt.")
                    # Auto-flat is derived from the frame's OWN sky, so it
                    # must be refitted per frame: the vignetting pattern
                    # tracks the sky level, airmass and any rotation, and
                    # this data set drifts ~350 px and rotates ~1.8 deg per
                    # night. Reusing the first frame's surface for the whole
                    # run imprinted a stale gradient on every later frame.
                    self.progress.emit(progress_frac,
                                       f"[{processed_count}/{total_lights}] Auto-flat...")
                    auto = calib.create_auto_flat_from_light(
                        lf, master_dark=dm, master_bias=bm,
                        polydeg=int(self.auto_flat_params.get('polydeg', 3)),
                        star_sigma=float(self.auto_flat_params.get('star_sigma', 4.0)))
                    fm = auto['model']
                    used_auto_flat = True
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
                        # The clip flag is the ACTUAL setting. It used to be
                        # derived from the combine method's name, so every
                        # median master was recorded as clipped whether or not
                        # clipping had run, and every mean one as unclipped.
                        orig_hdr.add_history(
                            f"Dark Master: {len(dark_files)} frames, {dark_method}, "
                            + (f"clip {dark_l:g}/{dark_u:g} sigma" if dark_clip else "no clip"))
                        for dfile in dark_files:
                            orig_hdr.add_history(f"  {dfile}")
                    
                    if flat_files:
                        # The clip flag is the ACTUAL setting. It used to be
                        # derived from the combine method's name, so every
                        # median master was recorded as clipped whether or not
                        # clipping had run, and every mean one as unclipped.
                        orig_hdr.add_history(
                            f"Flat Master: {len(flat_files)} frames, {flat_method}, "
                            + (f"clip {flat_l:g}/{flat_u:g} sigma" if flat_clip else "no clip"))
                        orig_hdr.add_history(f"  Flat pedestal subtracted: {flat_pedestal_name}")
                        orig_hdr.add_history("  Each flat scaled to unity by its own median "
                                             "before combining")
                        for ffile in flat_files:
                            orig_hdr.add_history(f"  {ffile}")
                    
                    if bias_files:
                        # The clip flag is the ACTUAL setting. It used to be
                        # derived from the combine method's name, so every
                        # median master was recorded as clipped whether or not
                        # clipping had run, and every mean one as unclipped.
                        orig_hdr.add_history(
                            f"Bias Master: {len(bias_files)} frames, {bias_method}, "
                            + (f"clip {bias_l:g}/{bias_u:g} sigma" if bias_clip else "no clip"))
                        for bfile in bias_files:
                            orig_hdr.add_history(f"  {bfile}")
                    
                    if used_auto_flat:
                        # A frame flat-fielded by a model of its own sky must not
                        # be indistinguishable from one flat-fielded by real flats.
                        orig_hdr.add_history(
                            "Flat field: AUTO-FLAT -- polynomial surface fitted to "
                            "THIS frame's own sky, not measured flats "
                            f"(polydeg={self.auto_flat_params.get('polydeg', 3)}, "
                            f"star_sigma={self.auto_flat_params.get('star_sigma', 4.0)})")

                    orig_hdr.add_history("")
                    
                    # Calibration operations
                    if dark_scale_applied is not None:
                        # The APPLIED factor. "ENABLED" alone was misleading:
                        # scaling is skipped when |factor-1| <= 0.001, so the
                        # header could claim it was on when nothing was scaled.
                        orig_hdr.add_history(
                            f"Dark Scaling: APPLIED x{dark_scale_applied:.6f} "
                            f"(Light exp: {light_et}s, Dark exp: {dark_et}s)")
                    elif self.scale_dark and darks:
                        orig_hdr.add_history(
                            f"Dark Scaling: not needed (Light exp: {light_et}s, "
                            f"Dark exp: {dark_et}s)")
                    else:
                        orig_hdr.add_history("Dark Scaling: DISABLED")
                    
                    if self.remove_cosmics:
                        orig_hdr.add_history(f"Cosmic Ray Removal: ENABLED (sigma: {self.outlier_sigma}, radius: {self.outlier_radius}px)")
                    
                    if hp_mask is not None and hp_mask.shape == (ny_l, nx_l):
                        orig_hdr.add_history("Hot Pixel Mask: APPLIED")
                    
                    if applied_bin > 1:
                        orig_hdr.add_history(f"Output Binning: {applied_bin}x{applied_bin}")
                    
                    orig_hdr.add_history("===== END CALIBRATION HISTORY =====")

                    # Free-text HISTORY cannot be queried. These can.
                    orig_hdr['CALSTAT'] = (
                        ('D' if darks else '') + ('B' if bias_files else '')
                        + ('A' if used_auto_flat else ('F' if flat_files else '')),
                        'D=dark B=bias F=flat A=auto-flat')
                    orig_hdr['CALSOFT'] = ('INOOR', 'Reduction software')
                    if darks:
                        orig_hdr['MASTERD'] = (len(dark_files), 'Dark frames combined')
                    if flat_files:
                        orig_hdr['MASTERF'] = (len(flat_files), 'Flat frames combined')
                    if bias_files:
                        orig_hdr['MASTERB'] = (len(bias_files), 'Bias frames combined')
                    if dark_scale_applied is not None:
                        orig_hdr['DARKSCAL'] = (float(dark_scale_applied),
                                                'Dark scaling factor applied')
                    orig_hdr['AUTOFLAT'] = (bool(used_auto_flat),
                                            'Flat modelled from this frame own sky')
                    
                return outpath, calibrated, orig_hdr

            # (lfpath, number, z, sky, sigma, written_as_float) of every light written
            done = []
            for i, lfpath in enumerate(lights):
                if self.isInterruptionRequested(): break
                if i in light_bads: continue

                processed_count += 1
                outpath, calibrated, orig_hdr = calibrate_light(lfpath, processed_count)
                z, sky, sigma = calib.sky_clip_margin(calibrated)
                if self._float_trigger is None and z < calib.FLOAT_STORAGE_Z:
                    self._float_trigger = (os.path.basename(lfpath), z)
                done.append((lfpath, processed_count, z, sky, sigma, self._float_trigger is not None))
                if self._float_trigger is not None:
                    self._save_float(outpath, calibrated, orig_hdr, z, sky, sigma)
                    continue
                frame_stats = {}
                calib.save_fits_16bit(outpath, calibrated, header=orig_hdr,
                                      saturate_level=4095, overwrite=True,
                                      stats_out=frame_stats)
                self._tally_clamping(os.path.basename(lfpath), frame_stats)

            storage_note = self._store_as_float_if_needed(done, calibrate_light)

            print("[CALIB] Calibration complete!")
            self.progress.emit(1.0, "Calibration complete")
            self.finished.emit(True, "Calibration complete." + storage_note + self._clamp_summary())
        except _Cancelled:
            print("[CALIB] Cancelled by user.")
            self.finished.emit(False, "Cancelled by user")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e))

    def _save_float(self, outpath, calibrated, orig_hdr, z, sky, sigma):
        name, z_trig = self._float_trigger
        calib.save_fits_float32(
            outpath, calibrated, header=orig_hdr, saturate_level=4095, overwrite=True,
            reason=(f"Sky {sky:.2f} +- {sigma:.2f} ADU, z = {z:.2f}; run stored as float "
                    f"because the sky of {name} has z = {z_trig:.2f} < {calib.FLOAT_STORAGE_Z:g}"))

    def _store_as_float_if_needed(self, done, calibrate_light):
        """
        One storage format per run. Frames are written as 16-bit integers (the
        raw frames' convention) until one has its sky less than
        calib.FLOAT_STORAGE_Z sigma above 0 ADU, where the 16-bit clamp erases
        the negative half of the sky noise; that frame and every later one are
        written as 32-bit float, and here the earlier ones are calibrated again
        and rewritten as float too. Mixed formats would make a faint star read
        differently on clamped and unclamped frames of one series.
        Returns a note for the completion dialog ('' when nothing changed).
        """
        if self._float_trigger is None:
            return ""
        lf_w, _, z_min, sky_w, sig_w, _ = min(done, key=lambda d: d[2])
        why = (f"the sky of {os.path.basename(lf_w)} sits only {z_min:.1f} sigma "
               f"above 0 ADU ({sky_w:.2f} +- {sig_w:.2f} ADU), so 16-bit integers would set "
               f"the negative half of its sky noise to 0 and bias faint stars")
        # The 16-bit files are replaced, and a float file clamps nothing.
        self.clamp_report = []
        todo = [d for d in done if not d[5]]
        for k, (lfpath, number, z, sky, sigma, _) in enumerate(todo):
            if self.isInterruptionRequested():
                return (f"\n\nCANCELLED: {len(todo) - k} of {len(done)} frame(s) are still "
                        f"16-bit and the rest are 32-bit float, because {why}. "
                        f"Re-run the calibration.")
            self.progress.emit(LIGHT_PHASE_START + (LIGHT_PHASE_END - LIGHT_PHASE_START) * k / len(todo),
                               f"[{k + 1}/{len(todo)}] Rewriting as 32-bit float...")
            outpath, calibrated, orig_hdr = calibrate_light(lfpath, number)
            self._save_float(outpath, calibrated, orig_hdr, z, sky, sigma)
        return (f"\n\nStored as 32-bit float: {why}. "
                f"All {len(done)} frame(s) of this run use the same format.")

    def _tally_clamping(self, name, stats):
        """Record how much of a frame could not be stored.

        The 16-bit output cannot hold a negative value, so an over-subtracted
        pixel is written as 0 -- it stops being data. On real INO340 frames
        this runs to thousands of pixels per frame (a scaled master dark drives
        hot pixels past what the 12-bit ADC could ever have recorded), and the
        only trace was a HISTORY line and a log warning that nobody reads.
        """
        lo = int(stats.get('clamped_low', 0))
        bad = int(stats.get('nonfinite', 0))
        if lo or bad:
            self.clamp_report.append((name, lo, bad, int(stats.get('size', 0))))

    def _clamp_summary(self):
        """One sentence for the completion dialog, empty when nothing clamped."""
        if not self.clamp_report:
            return ""
        total = sum(r[1] for r in self.clamp_report)
        nonfinite = sum(r[2] for r in self.clamp_report)
        worst = max(self.clamp_report, key=lambda r: r[1])
        pct = 100.0 * worst[1] / worst[3] if worst[3] else 0.0
        msg = (f"\n\nWARNING: {total:,} pixel(s) over {len(self.clamp_report)} "
               f"frame(s) were below 0 ADU and were stored as 0 -- they carry no "
               f"signal in the output. Worst frame: {worst[0]} "
               f"({worst[1]:,} px, {pct:.3f}%).\n"
               f"This usually means the dark over-subtracts: check DATAMIN and the "
               f"HISTORY of each file.")
        if nonfinite:
            msg += f"\n{nonfinite:,} non-finite pixel(s) were also stored as 0."
        return msg

    def _detect_minimum_shape(self, all_files):
        """Detect the minimum shape across all files. Files whose headers
        cannot be read are logged: they silently drop out of the target-shape
        decision, which changes how every other frame gets binned."""
        min_ny, min_nx = 1e9, 1e9
        for f in all_files:
            try:
                hdr = fits.getheader(f)
                min_ny = min(min_ny, hdr.get('NAXIS2', 1e9))
                min_nx = min(min_nx, hdr.get('NAXIS1', 1e9))
            except Exception as e:
                log.warning("Min-shape detection: cannot read header of %s: %s", f, e)
        return (int(min_ny), int(min_nx))

    def _get_timestamp(self):
        """Current UT timestamp for the calibration HISTORY.

        UT, not local time: every other time INOOR writes (the master's
        CRE_TIME) is UT, and the frames' own DATE keyword is UT while DATE-OBS
        is Iran local. A local stamp here put one run's products 3h30m apart
        in their own headers.
        """
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S') + ' UT'


# ---------- Flat Screening Worker ----------
class FlatScreenWorker(QtCore.QThread):
    """Measure a set of flat frames off the GUI thread.

    Screening reads every frame in full, which is ~0.5 s per 4096x4096 frame
    from the data drive -- ten seconds for a twenty-frame set, far too long to
    block the interface.
    """
    progress = QtCore.pyqtSignal(float, str)
    finished = QtCore.pyqtSignal(bool, object, str)   # ok, result dict, message

    def __init__(self, files, parent=None):
        super().__init__(parent=parent)
        self.files = list(files)

    def run(self):
        try:
            from core.flat_screening import screen_flat_frames

            def cb(frac, msg):
                if self.isInterruptionRequested():
                    raise _Cancelled()
                self.progress.emit(float(frac), msg)

            result = screen_flat_frames(self.files, mfc.load_fits,
                                        progress_callback=cb)
            self.finished.emit(True, result, result['summary'])
        except _Cancelled:
            self.finished.emit(False, None, "Cancelled by user")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, None, str(e))
