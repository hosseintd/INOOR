# multi_photometry_worker.py
# Bulk photometry worker: the clicked star is located in every frame from the
# whole field's transform to the reference frame, then re-centred once.
# Reads per-frame EXPTIME from header for magnitude calculation and returns it in results.

import math
import os
import numpy as np
from PyQt5 import QtCore
from astropy.io import fits
from astropy.wcs import WCS

from . import photometry_core as pc
from .align_utils import load_frame, field_stars, field_transform, disk_centroid
from .camera_config import DEFAULT_MODE, detect_mode_or_none, frame_binning

# Start of the method string of a frame whose star was re-centred at its
# last measured position because the field could not be matched.
LAST_POSITION = "last position"

class BulkPhotometryWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(str, float)
    done = QtCore.pyqtSignal(object)

    def __init__(self, files, reference_index, ref_source_xy,
                 fwhm=8.0, threshold_sigma=3.0,
                 inner_coef=2.0, outer_coef=3.0,
                 gain=DEFAULT_MODE.gain, read_noise=DEFAULT_MODE.read_noise, dark_noise=DEFAULT_MODE.dark_noise,
                 per_frame_mode=True,
                 max_radii_samples=200, detection_stamp_size=30,
                 zeropoint=0.0, zeropoint_map=None,
                 aperture_override=None, exptime_override=None, exptime_map=None,
                 aperture_policy='snr', growth_fraction=0.90, fwhm_factor=1.0,
                 aperture_correct=False,
                 parent=None):
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
        # When True, detect each frame's saving mode (High/Low/Merge) from its
        # own header/filename and use that mode's conversion gain + read/dark
        # noise for that frame's SNR. Falls back to the gain/read_noise/
        # dark_noise above whenever a frame's mode can't be identified, so
        # mixed or unlabelled datasets behave exactly as before.
        self.per_frame_mode = bool(per_frame_mode)
        self.max_radii_samples = int(max_radii_samples)
        # The detection stamp half-size no longer steers the bulk run (see
        # _predict); kept so callers do not change.
        self.detection_stamp_size = int(detection_stamp_size)
        self.zeropoint = float(zeropoint)
        # aperture_override (if provided) will be used for final photometry (magnitude)
        self.aperture_override = float(aperture_override) if aperture_override is not None else None
        # How the radius is chosen when no explicit override is given. The
        # 'frame_fwhm' rule derives it from the frame's PSF, so a run on one
        # star and a run on another star over the same frames necessarily use
        # identical radii -- which is what makes their magnitudes comparable.
        self.aperture_policy = str(aperture_policy or 'snr')
        self.growth_fraction = float(growth_fraction)
        self.fwhm_factor = float(fwhm_factor)
        # Report a TOTAL magnitude by adding the frame's aperture correction,
        # measured once per frame from its own bright isolated stars.
        self.aperture_correct = bool(aperture_correct)
        # exptime_override: if provided, worker will use it; otherwise worker reads header
        self.exptime_override = float(exptime_override) if exptime_override is not None else None

        # exptime_map: filename -> float. Per-frame fallback used only when the
        # frame's own header has no usable EXPTIME. Lets the caller supply values
        # already resolved elsewhere (e.g. typed by the user when the files were
        # added) without re-asking, while headers still win for frames that have
        # one so differing per-frame exposures are preserved.
        self.exptime_map = exptime_map if exptime_map is not None else {}

        # zeropoint_map: filename -> float
        self.zeropoint_map = zeropoint_map if zeropoint_map is not None else {}
        
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

    def _predict(self, data, hdr, ref_hdr, ref_stars, last_xy):
        """Where the clicked star lies in this frame: the click carried through
        the transform of the whole field onto the reference frame, or through
        both frames' WCS when the stars cannot be matched.

        The tracker this replaces SEARCHED for the star: it fitted the
        brightest blob in a +-100 px stamp and accepted it up to 40 px away,
        widening to +-250 px (accepting 100 px). Any star brighter than the
        target within ~100 px took the measurement over, usually from the
        second frame to the last, and the run still reported success: on PSR
        J2215 High_80..89 all 5 stars with a 2x brighter neighbour that it
        followed on the raw frames were lost that way, and
        over the whole night (drift 40-120 px between frames) all 3 isolated
        faint stars tried. A prediction from hundreds of stars cannot be
        captured by one neighbour.

        When neither works (a field with too few stars, frames not
        plate-solved) the star's position in the previously measured frame is
        used, and the method says why; on_bulk_done lists those frames.

        Returns ((x, y), method).
        """
        why = []
        x, y = self.ref_source_xy
        try:
            tform, info = field_transform(field_stars(data, self.fwhm), ref_stars)
            px, py = tform.inverse(np.array([[x, y]]))[0]
            return (float(px), float(py)), (
                f"field {info['model']} n={info['n_inliers']} rms={info['rms']:.2f}px "
                f"rot={info['rotation_deg']:.3f}deg")
        except Exception as e:
            why.append(f"stars: {e}")
        if self._wcs_usable(hdr) and self._wcs_usable(ref_hdr):
            try:
                sky = WCS(ref_hdr).celestial.pixel_to_world(x, y)
                px, py = WCS(hdr).celestial.world_to_pixel(sky)
                return (float(px), float(py)), "wcs"
            except Exception as e:
                why.append(f"wcs: {e}")
        else:
            why.append("wcs: frames not plate-solved")
        if self._ref_stars_error:
            why.append(self._ref_stars_error)
        return tuple(last_xy), f"{LAST_POSITION} (no field match: {'; '.join(why)})"

    def _recentre(self, data, xy):
        """ONE pass of the light-weighted centre in a disk of radius Fit FWHM
        around the prediction, above the photometry annulus' sky. Returns
        ((x, y), note).

        One pass on purpose. The prediction is already good to ~0.2-1 px, so
        the disk only has to pull it onto the star's core; iterating lets it
        walk down a bright neighbour's wing instead. On the PSR faint star
        (5x brighter neighbour 35 px away) one pass landed 0.07-0.36 px from
        a joint two-star fit in all 10 frames; ten passes in a 1.5 FWHM disk
        ended on the neighbour.

        No light above the sky inside the disk (a target that has faded out)
        leaves the prediction in place, and the note says so.
        """
        r = float(self.fwhm)
        sky, _ = pc.annulus_sky(data, xy, self.inner_coef * r, self.outer_coef * r)
        c = disk_centroid(data, xy, r, sky) if np.isfinite(sky) else None
        if c is None:
            return xy, "no light above sky: measured at the prediction"
        return c, f"centroid moved {math.hypot(c[0] - xy[0], c[1] - xy[1]):.2f}px"

    def _header_exptime(self, path):
        """Read exposure time (seconds) from the file's header.
        Single source of truth: utils.exptime_utils (do not duplicate the
        10us-tick heuristic here -- it diverged once already)."""
        from utils.exptime_utils import get_exptime_seconds
        try:
            hdr = fits.getheader(path)
        except Exception:
            return None
        return get_exptime_seconds(hdr, default=None)

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

        # The reference position is the click, exactly: the click's own
        # detection already chose the star. It used to be re-fitted here with
        # the TRACKING stamp, which moved a click on a faint PSR J2215 star
        # onto a 5x brighter one 35 px away before any frame was measured.
        self._ref_stars_error = None
        try:
            ref_stars = field_stars(ref_data, self.fwhm)
        except Exception as e:
            ref_stars = np.empty((0, 2))
            self._ref_stars_error = f"reference frame: {e}"
        last_xy = self.ref_source_xy

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

            # Locate the star: predicted from the field, re-centred once, and
            # measured on the frame itself (no resampling).
            try:
                align_method = 'unknown'
                if i == self.reference_index:
                    predicted = picked_full = self.ref_source_xy
                    align_method = 'reference'
                else:
                    predicted, align_method = self._predict(data, hdr, ref_hdr, ref_stars, last_xy)
                    ny, nx = data.shape
                    if not (0.0 <= predicted[0] <= nx - 1 and 0.0 <= predicted[1] <= ny - 1):
                        results.append(dict(index=i, success=False,
                                            msg=(f"Star falls outside this frame (predicted at "
                                                 f"{predicted[0]:.0f}, {predicted[1]:.0f})"),
                                            method=align_method, file=os.path.basename(path),
                                            exptime=None))
                        continue
                    picked_full, note = self._recentre(data, predicted)
                    align_method = f"{align_method}; {note}"

                picked_full = (float(picked_full[0]), float(picked_full[1]))
                last_xy = picked_full

                # Check the frame has data before computing SNR (e.g. not all NaN)
                valid_pixels = np.isfinite(data).sum()
                if valid_pixels == 0 or valid_pixels < data.size * 0.01:  # Less than 1% valid pixels
                    results.append(dict(index=i, success=False, msg=f"Invalid image (all NaN or insufficient data)", method=align_method, file=os.path.basename(path), exptime=None))
                    continue

                # Decide exposure time for instr_mag calculation.
                # Precedence: explicit override > this frame's own header >
                # per-frame fallback map (values already resolved by the caller).
                if self.exptime_override is not None:
                    exptime = float(self.exptime_override)
                else:
                    exptime = self._header_exptime(path)
                    if exptime is None:
                        exptime = self.exptime_map.get(os.path.basename(path))

                # Per-frame noise model: use THIS frame's saving mode (High/Low/
                # Merge) when it can be identified from the header or filename,
                # else fall back to the batch-level values. The dark charge is
                # the mode's dark current times this frame's exposure.
                f_gain, f_read_noise, f_dark_noise = self.gain, self.read_noise, self.dark_noise
                f_linear_limit = None
                frame_mode = None
                if self.per_frame_mode:
                    fm = detect_mode_or_none(hdr, path)
                    if fm is not None:
                        f_gain, f_read_noise = fm.gain, fm.read_noise
                        f_dark_noise = fm.dark_charge(exptime)
                        frame_mode = fm.name
                        f_linear_limit = fm.linear_limit_adu

                # Determine Zeropoint for this frame
                fname = os.path.basename(path)
                zp_to_use = float(self.zeropoint_map.get(fname, self.zeropoint))

                # One PSF measurement per FRAME, not per star: this is the
                # whole point of the frame_fwhm policy, and it would be the
                # dominant cost if repeated per measurement.
                frame_fwhm = frame_spread = None
                if (self.aperture_override is None
                        and self.aperture_policy == 'frame_fwhm') or self.aperture_correct:
                    frame_fwhm, frame_spread = pc.estimate_frame_fwhm(
                        data, fwhm_guess=self.fwhm, return_spread=True)

                # Likewise the aperture correction is a property of the frame:
                # one curve of growth from its bright isolated stars serves every
                # measurement in it.
                frame_cog = None
                if self.aperture_correct:
                    frame_cog = pc.frame_growth_curve(
                        data, frame_fwhm if frame_fwhm else self.fwhm,
                        inner_coef=self.inner_coef, outer_coef=self.outer_coef)

                # SNR curve -> aperture -> radial profile -> photometry, via the
                # SAME shared routine the GUI's click and live-update paths use,
                # so a bulk run reproduces exactly what a single click reports.
                # The radial profile is computed here (not lazily in the GUI)
                # because otherwise on_selection_changed recomputes it on the GUI
                # thread the first time each frame is displayed.
                m = pc.measure_star(
                    data, picked_full,
                    fwhm=self.fwhm,
                    aperture_override=self.aperture_override,
                    aperture_policy=self.aperture_policy,
                    growth_fraction=self.growth_fraction,
                    fwhm_factor=self.fwhm_factor,
                    frame_fwhm=frame_fwhm,
                    aperture_correct=self.aperture_correct,
                    frame_cog=frame_cog,
                    inner_coef=self.inner_coef, outer_coef=self.outer_coef,
                    gain=f_gain, read_noise=f_read_noise, dark_noise=f_dark_noise,
                    zeropoint=zp_to_use, exptime=exptime,
                    n_radii=self.max_radii_samples,
                    linear_limit=f_linear_limit,
                    frame_fwhm_spread=frame_spread)

                radii, snrs, r_best = m['radii'], m['snrs'], m['r_best']
                aperture_for_phot = m['r_used']
                rp_radii, rp_profile = m['rp_radii'], m['rp_profile']
                ap_res = m['phot']

                results.append(dict(index=i,
                                    success=True,
                                    msg=f"OK (align={align_method})",
                                    method=align_method,
                                    file=os.path.basename(path),
                                    picked_full=picked_full,
                                    predicted_xy=(float(predicted[0]), float(predicted[1])),
                                    radii=np.asarray(radii).tolist(),
                                    snrs=np.asarray(snrs).tolist(),
                                    rp_radii=(np.asarray(rp_radii).tolist() if rp_radii is not None else None),
                                    rp_profile=(np.asarray(rp_profile).tolist() if rp_profile is not None else None),
                                    r_best=float(r_best),
                                    frame_fwhm=(float(frame_fwhm) if frame_fwhm else None),
                                    frame_fwhm_spread=frame_spread,
                                    aperture_used=float(aperture_for_phot),
                                    mag=ap_res['mag'], # Use calibrated mag from core
                                    mag_err=ap_res['mag_err'],
                                    flux=ap_res['flux'],
                                    snr=ap_res['snr'],
                                    aperture_result=ap_res,
                                    camera_mode=frame_mode,
                                    binning=frame_binning(hdr, path),
                                    gain_used=float(f_gain),
                                    read_noise_used=float(f_read_noise),
                                    dark_noise_used=float(f_dark_noise),
                                    exptime=(float(exptime) if exptime is not None else None)))
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

        self.progress.emit("Bulk photometry finished", 1.0)
        self.done.emit(results)
