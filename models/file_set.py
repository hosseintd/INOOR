from dataclasses import dataclass, field
from typing import List, Set

@dataclass
class FileSet:
    """
    Represents a set of calibration frames (e.g. Flats, Darks) to be combined.
    """
    name: str = "New Set"
    set_type: str = "Flat"  # Flat, Dark, Bias, Light
    files: List[str] = field(default_factory=list)
    
    # Stacking parameters
    method: str = "median"  # 'median' or 'mean'
    do_sigma_clip: bool = True
    sigma_lower: float = 3.0
    sigma_upper: float = 3.0
    
    # Indices of files marked as bad by the user
    bad_indices: Set[int] = field(default_factory=set)
    
    # Gain table options (only for Flats)
    create_gain_table: bool = False
    gain_poly_degree: int = 2
