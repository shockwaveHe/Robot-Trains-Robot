from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class ZMPWalkConfig:
    """Data class to hold walking parameters."""

    plan_max_stride: List[float] = field(
        default_factory=lambda: [0.05, 0.01, np.pi / 8]
    )
    plan_t_step: float = 1.2
    control_dt: float = 0.01
    control_t_preview: float = 1.0
    control_t_filter: float = 0.5
    zmp_control_cost_Q: float = 1.0
    zmp_control_cost_R: float = 1e-6
    # lqr_control_cost_Q: float = 1.0
    # lqr_control_cost_R: float = 1e-6
    target_pose_init: List[float] = field(default_factory=lambda: [0.3, 0.0, 0.0])
    foot_step_height: float = 0.04
    squat_time: float = 1.0
    squat_height: float = 0.01
    y_offset_zmp: float = 0.034
    filter_dynamics: bool = False
    rotate_torso: bool = False
    speed_factor: float = 1.0
