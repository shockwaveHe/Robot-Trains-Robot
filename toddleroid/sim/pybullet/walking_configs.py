from dataclasses import dataclass

import numpy as np


@dataclass
class WalkingConfig:
    """Data class to hold walking parameters."""

    max_stride: np.ndarray = None  # x, y, theta
    plan_period: float = 0.0
    width: float = 0.0
    control_dt: float = 0.0
    control_period: float = 0.0
    control_cost_Q_val: float = 0.0
    control_cost_R_val: float = 0.0
    target_pos_init: np.ndarray = None  # x, y, theta
    sim_step_interval: int = 0


walking_config = WalkingConfig(
    max_stride=np.array([0.05, 0.03, 0.2]),
    plan_period=0.34,
    width=0.06,
    control_dt=0.01,
    control_period=1.0,
    control_cost_Q_val=1e8,
    control_cost_R_val=1.0,
    target_pos_init=np.array([0.4, 0.0, 0.5]),
    sim_step_interval=10,
)
