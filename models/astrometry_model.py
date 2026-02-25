from dataclasses import dataclass, field
from PyQt5.QtCore import QObject, pyqtSignal
from typing import Optional, Dict, List

class AstrometryModel(QObject):
    data_changed = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.current_file: Optional[str] = None
        self.solved_file: Optional[str] = None
        self.job_id: Optional[str] = None
        self.submission_id: Optional[str] = None
        self.rdls_path: Optional[str] = None
        self.axy_path: Optional[str] = None
        
        # Catalog results
        self.catalog_data: List[Dict] = []
        
        # Astrometry settings (should persist or be loaded from config)
        self.api_key: str = "" 
        self.scale_preset: str = "default"
        self.scale_low: float = 2.0
        self.scale_high: float = 10.0
        self.timeout: int = 900
        self.downsample: int = 1

    def update_solve_result(self, result: Dict):
        self.solved_file = result.get('solved_fits')
        self.submission_id = result.get('submission_id')
        self.job_id = result.get('job_id')
        self.rdls_path = result.get('rdls_path')
        self.axy_path = result.get('axy_path')
        self.data_changed.emit()

    def set_current_file(self, path: str):
        self.current_file = path
        self.data_changed.emit()
