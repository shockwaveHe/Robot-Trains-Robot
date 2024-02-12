from dataclasses import dataclass

import numpy as np


@dataclass
class WalkingConfig:
    """Data class to hold walking parameters."""

    max_stride: np.ndarray = None  # x, y, theta
    plan_period: float = 0.0
    control_dt: float = 0.0
    control_period: float = 0.0
    control_cost_Q_val: float = 0.0
    control_cost_R_val: float = 0.0
    target_pos_init: np.ndarray = None  # x, y, theta
    sim_step_interval: int = 0
    foot_step_height: float = 0.0
    y_offset_com_to_foot: float = 0.0


sustaina_op_pybullet_walking_config = WalkingConfig(
    max_stride=np.array([0.05, 0.03, 0.2]),
    plan_period=0.34,
    control_dt=0.01,
    control_period=1.0,
    control_cost_Q_val=1e8,
    control_cost_R_val=1.0,
    target_pos_init=np.array([0.4, 0.0, 0.5]),
    sim_step_interval=10,
    foot_step_height=0.06,
    y_offset_com_to_foot=0.06,
)

robotis_op3_pybullet_walking_config = WalkingConfig(
    max_stride=np.array([0.05, 0.03, 0.2]),
    plan_period=0.34,
    control_dt=0.01,
    control_period=1.0,
    control_cost_Q_val=1e8,
    control_cost_R_val=1.0,
    target_pos_init=np.array([0.4, 0.0, 0.5]),
    sim_step_interval=10,
    foot_step_height=0.06,
    y_offset_com_to_foot=0.05,
)

robotis_op3_mujoco_walking_config = WalkingConfig(
    max_stride=np.array([0.05, 0.01, 0.2]),
    plan_period=0.6,  # This is proabably the most important parameter to change
    control_dt=0.01,
    control_period=1.0,
    control_cost_Q_val=1e8,
    control_cost_R_val=1.0,
    target_pos_init=np.array([0.4, 0.0, 0.5]),
    sim_step_interval=10,
    foot_step_height=0.06,
    y_offset_com_to_foot=0.05,
)

walking_configs = {
    "sustaina_op_pybullet": sustaina_op_pybullet_walking_config,
    "robotis_op3_pybullet": robotis_op3_pybullet_walking_config,
    "robotis_op3_mujoco": robotis_op3_mujoco_walking_config,
}
