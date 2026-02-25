from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import os
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

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
        self.stamp_size: int = 100
        self.tracking_radius: int = 300
        self.fwhm: float = 12.0
        self.threshold: float = 3.0
        self.inner_coef: float = 2.0
        self.outer_coef: float = 3.0
        self.zeropoint: float = 0.0
        self.exptime_override: Optional[float] = None
        
        # Zeropoints: filename -> float
        self.zeropoints: Dict[str, float] = {}
        
        # Display settings
        self.display_max_dim: int = 1536
        
        # Results table
        self._phot_table: List[PhotometryRow] = []

    def set_zeropoint_for_file(self, filename: str, zp: float):
        self.zeropoints[filename] = zp
        # Also update any existing data in frame_data and phot_table
        for idx, data in self._frame_data.items():
            if data.fname == filename:
                data.mag = (data.aperture_result.get('instr_mag', 0) or 0) + zp if data.aperture_result else None
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
    
    def add_files(self, paths: List[str]):
        new_start = len(self._files)
        self._files.extend(paths)
        for i, p in enumerate(paths):
            idx = new_start + i
            from os.path import basename
            self._frame_data[idx] = FramePhotometryData(index=idx, fname=basename(p))
        if self._current_index == -1 and self._files:
            self._current_index = 0
        self.files_changed.emit()

    def remove_files(self, indices: List[int]):
        # Map current files to their data projects
        path_to_data = {self._files[i]: d for i, d in self._frame_data.items() if i < len(self._files)}
        
        # Remove files from list
        indices = sorted(indices, reverse=True)
        for idx in indices:
            if 0 <= idx < len(self._files):
                p = self._files.pop(idx)
                if p in path_to_data: del path_to_data[p]
        
        # Re-build frame_data with new indices
        self._frame_data = {}
        for i, p in enumerate(self._files):
            d = path_to_data.get(p)
            if d:
                d.index = i
                self._frame_data[i] = d
            else:
                from os.path import basename
                self._frame_data[i] = FramePhotometryData(index=i, fname=basename(p))
                
        # Adjust current_index
        if self._current_index >= len(self._files):
            self._current_index = len(self._files) - 1
            
        if not self._files:
            self.exptime_override = None

        self.files_changed.emit()
        self.data_changed.emit()
        
    def reorder_files(self, new_order_indices: List[int]):
        """Reorder files and their data based on a list of old indices."""
        current_path = self._files[self._current_index] if (0 <= self._current_index < len(self._files)) else None
        
        new_files = [self._files[i] for i in new_order_indices]
        new_frame_data = {}
        path_to_data = {self._files[i]: d for i, d in self._frame_data.items() if i < len(self._files)}
        
        new_curr_idx = -1
        for i, p in enumerate(new_files):
            if current_path and p == current_path:
                new_curr_idx = i
            d = path_to_data.get(p)
            if d:
                d.index = i
                new_frame_data[i] = d
            else:
                from os.path import basename
                new_frame_data[i] = FramePhotometryData(index=i, fname=basename(p))
        
        self._files = new_files
        self._frame_data = new_frame_data
        self._current_index = new_curr_idx
        
        self.files_changed.emit()
        self.data_changed.emit()

    def sort_by_time(self, sorted_paths: List[str]):
        # Rebuild model based on new path order while preserving data if path matches
        current_path = self._files[self._current_index] if (0 <= self._current_index < len(self._files)) else None
        
        new_data = {}
        path_to_data = {self._files[i]: d for i, d in self._frame_data.items() if i < len(self._files)}
        
        self._files = sorted_paths
        new_idx = -1
        for i, p in enumerate(self._files):
            d = path_to_data.get(p)
            if d:
                d.index = i
                new_data[i] = d
            else:
                from os.path import basename
                new_data[i] = FramePhotometryData(index=i, fname=basename(p))
            
            if current_path and p == current_path:
                new_idx = i
                
        self._frame_data = new_data
        if new_idx != -1:
            self._current_index = new_idx
            
        self.files_changed.emit()
        self.data_changed.emit()

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
                    date_obs=data.date_obs
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
