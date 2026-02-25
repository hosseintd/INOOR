# multi_photometry_worker.py
# Bulk photometry worker using robust alignment: ransac_align -> fallback_phase_align
# Reads per-frame EXPTIME from header for magnitude calculation and returns it in results.

import os
import numpy as np
from PyQt5 import QtCore
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
import astroalign as aa
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, warp

from . import photometry_core as pc
from .align_utils import ransac_align, fallback_phase_align, load_frame, find_star_in_frame, find_star_adaptive

class BulkPhotometryWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(str, float)
    done = QtCore.pyqtSignal(object)

    def __init__(self, files, reference_index, ref_source_xy,
                 fwhm=8.0, threshold_sigma=3.0,
                 inner_coef=2.0, outer_coef=3.0,
                 gain=16.5, read_noise=3.18, dark_noise=4.15,
                 max_radii_samples=200, search_stamp_size=300, detection_stamp_size=30,
                 zeropoint=0.0, zeropoint_map=None,
                 aperture_override=None, exptime_override=None, show_preview=False, parent=None):
        super().__init__(parent=parent)
        self.files = list(files)
        self.reference_index = int(reference_index)
        self.ref_source_xy = (float(ref_source_xy[0]), float(ref_source_xy[1]))
        self.fwhm = float(fwhm)
        self.threshold_sigma = float(threshold_sigma)
        self.inner_coef = float(inner_coef)
        self.outer_coef = float(outer_coef)
        self.gain = float(gain)
        self.read_noise = float(read_noise)
        self.dark_noise = float(dark_noise)
        self.max_radii_samples = int(max_radii_samples)
        # SEARCH stamp size (half-size) used for tracking area
        self.search_stamp_size = int(search_stamp_size)
        # DETECTION stamp size (half-size) used for precision refinement
        self.detection_stamp_size = int(detection_stamp_size)
        self.zeropoint = float(zeropoint)
        # aperture_override (if provided) will be used for final photometry (magnitude)
        self.aperture_override = float(aperture_override) if aperture_override is not None else None
        # exptime_override: if provided, worker will use it; otherwise worker reads header
        self.exptime_override = float(exptime_override) if exptime_override is not None else None
        
        # zeropoint_map: filename -> float
        self.zeropoint_map = zeropoint_map if zeropoint_map is not None else {}
        
        self.show_preview = bool(show_preview)
        self._abort = False

    def requestAbort(self):
        self._abort = True

    def _wcs_usable(self, hdr):
        try:
            w = WCS(hdr)
            if getattr(w.wcs, 'naxis', 0) < 2:
                return False
            crval = getattr(w.wcs, 'crval', None)
            if crval is None:
                return False
            if np.all(np.isnan(np.atleast_1d(crval))):
                return False
            return True
        except Exception:
            return False

    def _align(self, data, hdr, ref_data, ref_hdr):
        """Return (aligned_array, method_string). Tries ransac_align -> reproject -> astroalign -> phase_xcorr."""
        # try robust multi-star ransac translation first (fast, rotation-free)
        try:
            registered, method = ransac_align(data, ref_data, fwhm=max(4.0, self.fwhm), threshold_sigma=max(3.0, self.threshold_sigma), max_sources=600, match_radius=8.0)
            return registered, method
        except Exception:
            pass
        # prefer WCS reprojection when both headers have working WCS
        if self._wcs_usable(hdr) and self._wcs_usable(ref_hdr):
            try:
                arr, footprint = reproject_interp((data, hdr), ref_hdr, shape_out=ref_data.shape)
                return arr, 'reproject'
            except Exception:
                pass
        # try astroalign next
        try:
            registered, footprint = aa.register(data, ref_data, fill_value=np.nan)
            return registered.astype(np.float32), 'astroalign'
        except Exception:
            pass
        # final fallback: phase_cross_correlation
        try:
            return fallback_phase_align(data, ref_data), 'phase_xcorr'
        except Exception:
            # will raise below in caller if no method succeeded
            raise RuntimeError("All alignment methods failed")

    def _header_exptime(self, path):
        """Read EXPTIME header and convert 10-microsecond ticks if necessary."""
        try:
            hdr = fits.getheader(path)
        except Exception:
            return None
        for key in ('EXPTIME', 'EXPOSURE', 'EXPT', 'TEXPTIME', 'EXPOS'):
            if key in hdr:
                val = hdr.get(key)
                if val is None:
                    continue
                try:
                    v = float(val)
                    # Heuristic for 10us units (968992 -> ~9.7s)
                    # Often the comment explicitly says "10us"
                    comment = ""
                    try: comment = str(hdr.comments[key]).lower()
                    except: pass
                    if v > 1000 and ("10us" in comment or v > 1e5):
                        return round(v / 100000.0, 6)
                    return round(v, 6)
                except Exception:
                    continue
        return None

    def run(self):
        results = []
        N = len(self.files)
        if N == 0:
            self.done.emit([dict(success=False, msg="No files provided")])
            return
        if self.reference_index < 0 or self.reference_index >= N:
            self.reference_index = 0

        ref_path = self.files[self.reference_index]
        ref_hdul = fits.open(ref_path, memmap=False)
        ref_data = ref_hdul[0].data.astype(np.float32)
        ref_hdr = ref_hdul[0].header
        ref_hdul.close()

        # try to refine reference coordinate using local detection (stamp only)
        try:
            refined_xy, params, _ = find_star_in_frame(ref_data, self.ref_source_xy,
                                                       stamp_radius=self.search_stamp_size,
                                                       fwhm_guess=self.fwhm)
            if refined_xy is not None:
                self.ref_source_xy = refined_xy
        except Exception:
            pass

        preview_frames = [ref_data.copy()]

        # Track previous detection for adaptive search across frames
        previous_detected_xy = None

        for i, path in enumerate(self.files):
            if self._abort:
                self.progress.emit("Aborted", 1.0)
                break
            frac = (i / max(1, N-1)) * 0.9
            self.progress.emit(f"Processing {os.path.basename(path)} ({i+1}/{N})", frac)

            try:
                hd = fits.open(path, memmap=False)
                data = hd[0].data.astype(np.float32)
                hdr = hd[0].header
                hd.close()
            except Exception as e:
                results.append(dict(index=i, success=False, 
                                  msg=f"Failed to load file: {str(e)[:80]}",
                                  method='load_error', file=os.path.basename(path), exptime=None))
                continue

            # Determine current effective target position and result
            try:
                found2 = None
                align_method = 'unknown' # Initialize align_method

                if i == self.reference_index:
                    found2 = self.ref_source_xy
                    align_method = 'reference'
                    aligned = ref_data.copy() # Ensure aligned is set for the reference frame
                else:
                    # First, align the current frame to the reference frame
                    # Use adaptive detection: searches near previous position if available
                    found_local, fitp_local, stamp_local = find_star_adaptive(
                        data, 
                        self.ref_source_xy,
                        previous_detected_xy=previous_detected_xy,
                        stamp_radius=self.search_stamp_size,
                        fwhm_guess=self.fwhm,
                        max_search_radius=self.search_stamp_size + 200
                    )
                    if found_local is not None:
                        align_method = 'local-detect'
                        aligned = data
                    else:
                        # fall back to global alignment methods
                        try:
                            aligned, align_method = self._align(data, hdr, ref_data, ref_hdr)
                        except Exception:
                            # final fallback: phase_cross_correlation with normalization to avoid overflow
                            try:
                                t1 = ref_data - np.nanmedian(ref_data); t1 = t1 / (np.nanstd(t1) + 1e-12)
                                t2 = data - np.nanmedian(data); t2 = t2 / (np.nanstd(t2) + 1e-12)
                                shift, error, diff = phase_cross_correlation(t1, t2, upsample_factor=10)
                                dy, dx = shift
                                tform = AffineTransform(translation=(dx, dy))
                                warped = warp(data, inverse_map=tform.inverse, output_shape=ref_data.shape, preserve_range=True, cval=np.nan)
                                aligned = warped.astype(np.float32)
                                align_method = f'phase_xcorr shift={shift}'
                            except Exception:
                                aligned = np.full_like(ref_data, np.nan)
                                align_method = 'failed_alignment'

                    # --- Step 1: Approximate Candidate (Tracking Area) ---
                    # attempt to find star in the (aligned or original) image using the search_stamp_size (Tracking Area)
                    found_approx, fit_approx, stamp_approx = find_star_adaptive(
                        aligned, 
                        self.ref_source_xy,
                        previous_detected_xy=previous_detected_xy,
                        stamp_radius=self.search_stamp_size,
                        fwhm_guess=self.fwhm,
                        max_search_radius=self.search_stamp_size + 200
                    )

                    # --- Step 2: Strict Refinement (Localized Precision) ---
                    # If we found an approximate candidate, RE-RUN refinement but STRICTLY localized
                    # to the detection_stamp_size to ensure we don't grab a bright neighbor.
                    if found_approx is not None:
                        # Use detect_then_refine with strict stamp for final lock
                        found_final, params_final, stamp_final, method_final = pc.detect_then_refine(
                            aligned, found_approx,
                            crop_half_size=self.detection_stamp_size,
                            stamp_radius=self.detection_stamp_size,
                            fwhm=self.fwhm,
                            threshold_sigma=self.threshold_sigma,
                            expand_steps=(0,) # Strict. No jumping.
                        )
                        if found_final:
                            found2 = found_final
                            align_method = f"{align_method}+strict_localized"
                        else:
                            found2 = found_approx # Fallback to the approximate fit if strict refinement failed
                    
                    # If adaptive local detection fails, try global source detection in search area
                    if found2 is None:
                        try:
                            sources, crop_origin, crop = pc.detect_sources_in_crop(
                                aligned,
                                center_xy=self.ref_source_xy if previous_detected_xy is None else previous_detected_xy,
                                crop_half_size=self.search_stamp_size,
                                fwhm=self.fwhm,
                                threshold_sigma=max(2.5, self.threshold_sigma)
                            )
                            
                            if sources is not None and len(sources) > 0:
                                # Pick best source
                                res = pc.pick_best_source_crowded(sources, crop_origin, self.ref_source_xy, fwhm=self.fwhm)
                                src_xy = res[0]
                                if src_xy is not None:
                                    found2 = src_xy
                                    align_method = f"{align_method}+global_detect"
                        except Exception:
                            pass

                if found2 is None:
                    results.append(dict(index=i, success=False, msg=f"Not found (align={align_method})", method=align_method, file=os.path.basename(path), exptime=None))
                    preview_frames.append(aligned)
                    continue

                # Update tracking for next frame
                previous_detected_xy = found2

                picked_full = (float(found2[0]), float(found2[1]))

                # Validate aligned image before computing SNR - check if it's all NaN or has insufficient valid data
                valid_pixels = np.isfinite(aligned).sum()
                if valid_pixels == 0 or valid_pixels < aligned.size * 0.01:  # Less than 1% valid pixels
                    results.append(dict(index=i, success=False, msg=f"Invalid aligned image (all NaN or insufficient data)", method=align_method, file=os.path.basename(path), exptime=None))
                    preview_frames.append(aligned)
                    continue

                # compute SNR vs radius (for diagnostics) — fast-ish
                max_snr_r = max(40, 5 * self.fwhm)
                radii_diag = np.linspace(1, max_snr_r, self.max_radii_samples)
                radii, snrs, r_best, mag_err = pc.compute_snr_vs_radius(aligned, picked_full,
                                                                        fwhm=self.fwhm,
                                                                        radii=radii_diag,
                                                                        gain=self.gain)

                # decide exposure time for instr_mag calculation
                exptime = None
                if self.exptime_override is not None:
                    exptime = float(self.exptime_override)
                else:
                    exptime = self._header_exptime(path)

                # Choose final aperture to use for magnitude:
                # - if user provided an explicit aperture_override, use it for the reported mag
                # - else use r_best (SNR-optimized)
                aperture_for_phot = float(self.aperture_override) if self.aperture_override is not None else float(r_best)

                # Determine Zeropoint for this frame
                fname = os.path.basename(path)
                zp_to_use = float(self.zeropoint_map.get(fname, self.zeropoint))

                ap_res = pc.perform_aperture_photometry(aligned, picked_full, aperture_for_phot,
                                                       inner_coef=self.inner_coef, outer_coef=self.outer_coef,
                                                       gain=self.gain, read_noise=self.read_noise, dark_noise=self.dark_noise,
                                                       zeropoint=zp_to_use, exptime=exptime)
                
                results.append(dict(index=i,
                                    success=True,
                                    msg=f"OK (align={align_method})",
                                    method=align_method,
                                    file=os.path.basename(path),
                                    picked_full=picked_full,
                                    radii=np.asarray(radii).tolist(),
                                    snrs=np.asarray(snrs).tolist(),
                                    r_best=float(r_best),
                                    aperture_used=float(aperture_for_phot),
                                    mag=ap_res['mag'], # Use calibrated mag from core
                                    mag_err=ap_res['mag_err'],
                                    flux=ap_res['flux'],
                                    snr=ap_res['snr'],
                                    aperture_result=ap_res,
                                    exptime=(float(exptime) if exptime is not None else None)))
                preview_frames.append(aligned)
            except Exception as frame_error:
                # Gracefully handle any per-frame error without crashing the entire bulk operation
                # Build a user-friendly error message
                error_msg = str(frame_error)
                if "All-NaN" in error_msg:
                    error_msg = "Cannot measure: no valid data (possibly a bias frame)"
                elif len(error_msg) > 100:
                    error_msg = error_msg[:97] + "..."
                
                results.append(dict(index=i, success=False,
                                  msg=f"Processing failed: {error_msg}",
                                  method=align_method if 'align_method' in locals() else 'unknown',
                                  file=os.path.basename(path), exptime=None))
                # Try to add a placeholder preview frame if available
                try:
                    if 'aligned' in locals() and aligned is not None:
                        preview_frames.append(aligned)
                    else:
                        preview_frames.append(np.zeros((100, 100), dtype=np.float32))
                except:
                    preview_frames.append(np.zeros((100, 100), dtype=np.float32))

        self.progress.emit("Bulk photometry finished", 1.0)
        self.done.emit(results)
