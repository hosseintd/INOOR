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

    # The 2D post-filter replaces spatial outliers with the local block
    # median, which on a master bias/dark erases exactly the stable hot
    # pixels the master exists to subtract. Off by default; see
    # apply_2d_sigma_clip for the measured numbers.
    do_sigma_clip: bool = False
    sigma_lower: float = 3.0
    sigma_upper: float = 3.0

    # Put every frame on a common level before combining. Right for sky
    # flats, which fade during the sequence; wrong for bias and dark, whose
    # absolute level is the signal. Off unless the frames look like flats.
    scale_to_common_level: bool = False
    
    # Indices of files marked as bad by the user
    bad_indices: Set[int] = field(default_factory=set)
