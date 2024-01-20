from dataclasses import dataclass

import numpy as np


@dataclass
class WalkingConfig:
    """Data class to hold walking parameters."""

    max_stride: np.ndarray = None  # x, y, theta
    plan_period: float = 0.0
    width: float = 0.0


walking_config = WalkingConfig(
    max_stride=np.array([0.05, 0.03, 0.2]), plan_period=0.34, width=0.06
)
