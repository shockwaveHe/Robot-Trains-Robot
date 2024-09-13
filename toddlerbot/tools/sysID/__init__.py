from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SysIDSpecs:
    amplitude_ratio: float = 0.75
    initial_frequency: float = 0.1
    final_frequency: float = 1.5
    decay_rate: float = 0.2
    direction: float = 1
    kp_list: Optional[List[float]] = None
    warm_up_angles: Optional[Dict[str, float]] = None
