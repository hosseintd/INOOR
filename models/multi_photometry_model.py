from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import os
import math
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from core.camera_config import DEFAULT_MODE

@dataclass
class PhotometryRow:
    """Represents a single row in the photometry results table."""
    index: int
    filename: str
    x: float
    y: float
    mag: Optional[float] = None
    mag_err: Optional[float] = None
    flux: float = 0.0
    snr: float = 0.0
    zeropoint: float = 0.0
    date_ut: Optional[str] = None
    date_obs: Optional[str] = None
    # --- measurement provenance ---------------------------------------------
    # Exported alongside the results so a light curve can be audited and
    # reproduced later: without the aperture actually used and the camera
    # parameters that fed the noise model, the reported SNR/Err cannot be
    # recomputed (the noise equation is degenerate in gain without n_ap).
    aperture: Optional[float] = None      # radius actually used, px
    r_best: Optional[float] = None        # SNR-optimised radius, px (diagnostic)
    inner_coef: float = 2.0               # sky annulus inner radius = coef * aperture
    outer_coef: float = 3.0
    bkg_mean: Optional[float] = None      # ADU/pixel from the annulus
    aperture_area: Optional[float] = None # n_ap, pixels
    exptime: Optional[float] = None       # seconds
    camera_mode: Optional[str] = None     # High / Low / Merge, or "manual"
    gain: Optional[float] = None          # e-/ADU used for THIS frame
    read_noise: Optional[float] = None    # e-
    dark_noise: Optional[float] = None    # e-
    # 'applied', 'unavailable' (asked for, frame curve of growth not
    # measurable) or 'off'; None for results made before this was recorded.
    aperture_correction_status: Optional[str] = None
    # brightest aperture pixel and 'ok' / 'nonlinear' / 'saturated'
    peak_adu: Optional[float] = None
    saturation: Optional[str] = None
    # the star's own FWHM and 'ok' / 'broad' / 'unchecked' against the frame PSF
    star_fwhm: Optional[float] = None
    psf_check: Optional[str] = None

@dataclass
class FramePhotometryData:
    """Detailed photometry data for a single frame."""
    index: int
    fname: str
    selected_source: Optional[Tuple[float, float]] = None
    aperture_radius: float = 10.0
    inner_coef: float = 2.0
    outer_coef: float = 3.0
    
    # SNR/Profile curves
    radii: Optional[np.ndarray] = None
    snrs: Optional[np.ndarray] = None
    r_best: Optional[float] = None
    rp_radii: Optional[np.ndarray] = None
    rp_profile: Optional[np.ndarray] = None
    
    # Results
    mag: Optional[float] = None
    mag_err: Optional[float] = None
    flux: float = 0.0
    snr: float = 0.0
    aperture_result: Dict = field(default_factory=dict)
    is_bad: bool = False
    
    exptime: Optional[float] = None
    date_ut: Optional[str] = None
    date_obs: Optional[str] = None

    # Camera parameters actually used for THIS frame's noise model. The bulk
    # worker fills these from the frame's own detected mode (per_frame_mode);
    # they stay None on the interactive path, where the UI-set values apply.
    camera_mode: Optional[str] = None
    gain_used: Optional[float] = None
    read_noise_used: Optional[float] = None
    dark_noise_used: Optional[float] = None

    # Selection and limits
    view_xlim: Optional[Tuple[float, float]] = None
    view_ylim: Optional[Tuple[float, float]] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None

class MultiPhotometryModel(QObject):
    """
    Model for Multi-Photometry tab.
    Manages a list of files and their associated measurements.
    """
    files_changed = pyqtSignal()
    data_changed = pyqtSignal() # Emitted when frame_data or table changes
    selection_changed = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self._files: List[str] = []
        self._current_index: int = -1
        
        # frame_index -> FramePhotometryData
        self._frame_data: Dict[int, FramePhotometryData] = {}
        
        # Parameters
        self.aperture: float = 10.0
        self.fixed_aperture: bool = False
        # How the aperture radius is chosen when `fixed_aperture` is off.
        # "frame_fwhm" is the default: it derives the radius from the frame's
        # PSF alone, so two separate runs on two stars get exactly the same
        # radius (measured star-to-star mismatch 0.000 px, against 0.48-0.72 px
        # for "snr"), which light curves and zeropoints need. The per-star SNR
        # optimum ("snr", the old default) left PSR J2215 error bars 2.5-3.1x
        # too small, moved one GRB 251013C star's radius between 10.3 and
        # 16.1 px over 12 frames, and beside a 5x brighter star took its global
        # SNR maximum at 43-48 px, inside which the neighbour sat. It stays
        # available for single-frame inspection. See the note above
        # APERTURE_POLICIES in core/photometry_core.py for the measurements.
        self.aperture_policy: str = "frame_fwhm"
        self.growth_fraction: float = 0.90   # for "growth"
        self.fwhm_factor: float = 1.0        # for "frame_fwhm"; 1.0 minimised
                                             # the differential scatter on the
                                             # 139-frame PSR J2215 test
        # Report a TOTAL magnitude rather than the flux inside the aperture, by
        # adding the frame's aperture correction. Off by default: it changes the
        # zeropoint scale, and it is unnecessary for differential work where the
        # aperture loss already cancels between stars sharing a radius.
        self.aperture_correct: bool = False
        # This MUST match the spinbox default in views/multi_photometry_view.py
        # (spin_stamp). It previously did not: the view opened showing 50 while
        # the model held 100, so an untouched session measured with 100 even
        # though the UI said 50 -- and the first update_params() call then
        # silently jumped the displayed value.
        self.stamp_size: int = 50        # detection / precision stamp half-size
        self.fwhm: float = 12.0
        self.threshold: float = 3.0
        self.inner_coef: float = 2.0
        self.outer_coef: float = 3.0
        self.zeropoint: float = 0.0
        self.exptime_override: Optional[float] = None

        # Camera noise model (see core/camera_config.py for units and provenance).
        # gain is CONVERSION gain in e-/ADU, NOT the FITS-header GAIN setting.
        self.gain: float = DEFAULT_MODE.gain
        self.read_noise: float = DEFAULT_MODE.read_noise
        self.dark_noise: float = DEFAULT_MODE.dark_noise
        
        # Zeropoints: filename -> float
        self.zeropoints: Dict[str, float] = {}
        # filename -> {'refs': [{'name','ref_mag','instr_mag','zp','x','y'}, ...],
        #              'policy': str}. Provenance for each stored zeropoint.
        self.zeropoint_meta: Dict[str, dict] = {}
        
        # Display settings
        self.display_max_dim: int = 1536
        
        # Results table
        self._phot_table: List[PhotometryRow] = []

    def zeropoint_for(self, filename: str) -> float:
        """
        The zeropoint that applies to one frame: its own if it has one, else
        the global default.

        Keys are BASENAMES throughout. Five readers already assumed that while
        one writer stored a full path, so a zeropoint typed into the spin box
        went under a key nothing could ever look up. Going through this method
        keeps the convention in one place.
        """
        return self.zeropoints.get(os.path.basename(str(filename)), self.zeropoint)

    def magnitude_for(self, data) -> Optional[float]:
        """
        Calibrated magnitude of one measured frame, from ONE formula.

        There used to be two. `set_zeropoint_for_file` used the stored
        instrumental magnitude, which carries the frame's aperture correction.
        `_update_all_frame_results` instead recomputed
        `-2.5*log10(flux/exptime)` from the raw aperture flux, which does NOT
        carry it -- so touching the zeropoint or exposure box silently dropped
        the correction (~0.27 mag at 1 FWHM) from every frame, and applied the
        GLOBAL zeropoint on top, wiping any per-file calibration.

        Recomputing from flux is still needed so a corrected exposure time
        takes effect, so the aperture correction is added back explicitly.
        """
        ap = data.aperture_result or {}
        flux = ap.get('flux')
        et = data.exptime if (data.exptime is not None and data.exptime > 0)             else self.exptime_override
        if flux is None or not (flux > 0) or not et or not (et > 0):
            return None
        ap_corr = float(ap.get('aperture_correction', 0.0) or 0.0)
        instr = -2.5 * math.log10(float(flux) / float(et)) + ap_corr
        return instr + self.zeropoint_for(data.fname)

    def set_zeropoint_for_file(self, filename: str, zp: float, meta=None):
        """
        Store a zeropoint for one frame, optionally with the evidence behind it.

        `meta` records which reference star(s) produced the number (catalogue
        magnitude, measured instrumental magnitude, position, per-star ZP). A
        zeropoint with no provenance cannot be audited later -- you cannot tell
        whether it came from a saturated star, a blend, or a different aperture
        rule -- so the calibration path always fills it in.
        """
        self.zeropoints[filename] = zp
        if meta is not None:
            self.zeropoint_meta[filename] = meta

        # Also update any existing data in frame_data and phot_table.
        # A missing instr_mag must yield mag=None, NOT mag=zp: the old
        # `(... or 0) + zp` silently fabricated a plausible-looking magnitude
        # equal to the zeropoint whenever instrumental photometry had failed.
        for idx, data in self._frame_data.items():
            if data.fname == filename:
                instr_mag = data.aperture_result.get('instr_mag') if data.aperture_result else None
                data.mag = (instr_mag + zp) if instr_mag is not None else None
        self.data_changed.emit()

    def clear_zeropoints(self, filenames=None):
        """
        Drop stored zeropoints so magnitudes fall back to instrumental.
        `filenames=None` clears all of them. Frames whose zeropoint is removed
        get their magnitude reset to the instrumental value rather than left
        holding a stale calibrated number.
        """
        targets = list(self.zeropoints.keys()) if filenames is None else list(filenames)
        for fn in targets:
            self.zeropoints.pop(fn, None)
            self.zeropoint_meta.pop(fn, None)
        cleared = set(targets)
        for idx, data in self._frame_data.items():
            if data.fname in cleared:
                instr_mag = data.aperture_result.get('instr_mag') if data.aperture_result else None
                data.mag = instr_mag
        self.data_changed.emit()

    def add_to_table(self, row: PhotometryRow):
        # Unique by index - replace if exists
        self._phot_table = [r for r in self._phot_table if r.index != row.index]
        self._phot_table.append(row)
        self.data_changed.emit()

    def clear_table(self):
        self._phot_table = []
        self.data_changed.emit()

    @property
    def files(self): return self._files
    
    def add_files(self, paths: List[str]) -> List[int]:
        """Add files, skipping paths already loaded (duplicate full paths
        would alias onto a single FramePhotometryData after any
        remove/reorder/sort, silently linking two rows' measurements).

        Returns the list of indices that were actually added.
        """
        from os.path import basename
        existing = set(self._files)
        added_indices: List[int] = []
        for p in paths:
            if p in existing:
                continue
            existing.add(p)
            idx = len(self._files)
            self._files.append(p)
            self._frame_data[idx] = FramePhotometryData(index=idx, fname=basename(p))
            added_indices.append(idx)
        if self._current_index == -1 and self._files:
            self._current_index = 0
        self.files_changed.emit()
        return added_indices

    def _rebuild_from_old_indices(self, old_indices: List[int]):
        """Rebuild files and frame data so position i holds what old_indices[i] held.

        Carrying the data by INDEX, not by path. All three callers used to
        rebuild through a `{path: data}` dict, which collapses two equal paths
        onto one FramePhotometryData -- after which both rows referenced the
        SAME object and a measurement written to one appeared in the other.
        `add_files` de-duplicates by path, which is the only reason that never
        fired, but the hazard sat one bad call away. An index permutation
        cannot alias.
        """
        from os.path import basename

        old_files = self._files
        old_data = self._frame_data

        self._files = [old_files[i] for i in old_indices]
        self._frame_data = {}
        for new_i, old_i in enumerate(old_indices):
            d = old_data.get(old_i)
            if d is None:
                d = FramePhotometryData(index=new_i,
                                        fname=basename(self._files[new_i]))
            else:
                d.index = new_i
            self._frame_data[new_i] = d

    def remove_files(self, indices: List[int]):
        drop = {i for i in indices if 0 <= i < len(self._files)}
        if not drop:
            return
        keep = [i for i in range(len(self._files)) if i not in drop]

        # Where the selection ends up. It used to be clamped only when it ran
        # off the end, so removing a row BELOW the current one silently slid
        # the selection onto a different file.
        old_current = self._current_index
        self._rebuild_from_old_indices(keep)
        if old_current in drop or old_current < 0:
            self._current_index = min(old_current, len(self._files) - 1)
        else:
            self._current_index = keep.index(old_current)
        if not self._files:
            self._current_index = -1
            self.exptime_override = None

        self.files_changed.emit()
        self.data_changed.emit()

    def reorder_files(self, new_order_indices: List[int]):
        """Reorder files and their data based on a list of old indices."""
        n = len(self._files)
        if sorted(new_order_indices) != list(range(n)):
            raise ValueError(
                f"reorder_files needs a permutation of 0..{n - 1}, got "
                f"{len(new_order_indices)} indices")

        old_current = self._current_index
        self._rebuild_from_old_indices(list(new_order_indices))
        self._current_index = (new_order_indices.index(old_current)
                               if 0 <= old_current < n else -1)

        self.files_changed.emit()
        self.data_changed.emit()

    def sort_by_time(self, sorted_paths: List[str]):
        """Reorder to `sorted_paths`, which must be a permutation of the current files.

        It used to assign `self._files = sorted_paths` outright: a path that was
        not in the list produced a fresh empty record, and any path left out
        lost its measurements, both without a word.
        """
        remaining = {}
        for i, p in enumerate(self._files):
            remaining.setdefault(p, []).append(i)

        order = []
        for p in sorted_paths:
            slots = remaining.get(p)
            if not slots:
                raise ValueError(
                    f"sort_by_time was given a path that is not in the current "
                    f"file list (or is repeated): {p}")
            order.append(slots.pop(0))
        if len(order) != len(self._files):
            raise ValueError(
                f"sort_by_time got {len(sorted_paths)} paths for "
                f"{len(self._files)} files; every file must appear exactly once")

        self.reorder_files(order)


    def set_current_index(self, index: int):
        if -1 <= index < len(self._files):
            self._current_index = index
            self.selection_changed.emit(index)

    def get_current_data(self) -> Optional[FramePhotometryData]:
        return self._frame_data.get(self._current_index)

    def update_frame_data(self, index: int, **kwargs):
        if index in self._frame_data:
            data = self._frame_data[index]
            for k, v in kwargs.items():
                if hasattr(data, k):
                    setattr(data, k, v)
            self.data_changed.emit()

    def get_photometry_table(self) -> List[PhotometryRow]:
        rows = []
        for i, f in enumerate(self._files):
            data = self._frame_data.get(i)
            if data and data.selected_source:
                # Use per-file zeropoint if exists, else global
                zp = self.zeropoints.get(os.path.basename(f), self.zeropoint)
                ap_res = data.aperture_result or {}
                rows.append(PhotometryRow(
                    index=i,
                    filename=data.fname,
                    x=data.selected_source[0],
                    y=data.selected_source[1],
                    mag=data.mag,
                    mag_err=data.mag_err,
                    flux=data.flux,
                    snr=data.snr,
                    zeropoint=zp,
                    date_ut=data.date_ut,
                    date_obs=data.date_obs,
                    aperture=data.aperture_radius,
                    r_best=data.r_best,
                    # the annulus coefficients are a global setting, not per frame
                    inner_coef=self.inner_coef,
                    outer_coef=self.outer_coef,
                    bkg_mean=ap_res.get('bkg_mean'),
                    aperture_area=ap_res.get('aperture_area'),
                    exptime=(data.exptime if data.exptime is not None
                             else self.exptime_override),
                    # per-frame values when the bulk worker detected the mode,
                    # otherwise whatever the UI was set to
                    camera_mode=(data.camera_mode or "manual"),
                    gain=(data.gain_used if data.gain_used is not None else self.gain),
                    read_noise=(data.read_noise_used if data.read_noise_used is not None
                                else self.read_noise),
                    dark_noise=(data.dark_noise_used if data.dark_noise_used is not None
                                else self.dark_noise),
                    aperture_correction_status=ap_res.get('aperture_correction_status'),
                    peak_adu=ap_res.get('peak_adu'),
                    saturation=ap_res.get('saturation'),
                    star_fwhm=ap_res.get('star_fwhm'),
                    psf_check=ap_res.get('psf_check'),
                ))
        return rows
    def remove_results(self, indices: List[int]):
        """Clear photometry results for specific frames (but keep the frames)."""
        changed = False
        for i in indices:
            if i in self._frame_data:
                d = self._frame_data[i]
                d.selected_source = None
                d.mag = None
                d.snr = 0.0
                d.flux = 0.0
                d.mag_err = None
                # Keep cached image data (rp_profile etc) or clear? 
                # User said "removed result be removed from the light curve". 
                # Clearing selected_source usually invalidates the result.
                changed = True
        if changed:
            self.data_changed.emit()

    def trigger_update(self):
        self.data_changed.emit()
