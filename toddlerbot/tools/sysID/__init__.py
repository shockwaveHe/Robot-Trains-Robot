from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SysIDSpecs:
    amplitude_ratio: float = 0.75
    initial_frequency: float = 0.1
    final_frequency: float = 0.5
    warm_up_angles: Dict[str, float] = field(default_factory=lambda: {})
    direction: float = 1
    kp_list: List[float] = field(default_factory=lambda: [])
