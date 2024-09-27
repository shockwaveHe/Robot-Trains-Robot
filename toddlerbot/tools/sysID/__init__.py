from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SysIDSpecs:
    amplitude_list: List[float]
    initial_frequency: float = 0.1
    final_frequency: float = 10.0
    decay_rate: float = 0.1
    direction: float = 1
    kp_list: Optional[List[float]] = None
    warm_up_angles: Optional[Dict[str, float]] = None
