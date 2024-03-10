from dataclasses import dataclass

import numpy as np


@dataclass
class WalkingConfig:
    """Data class to hold walking parameters."""

    plan_max_stride: np.ndarray = None  # x, y, theta
    plan_t_step: float = 0.0
    control_dt: float = 0.0
    control_t_preview: float = 0.0
    control_cost_Q_val: float = 0.0
    control_cost_R_val: float = 0.0
    target_pose_init: np.ndarray = None  # x, y, theta
    actuator_steps: int = 0
    foot_step_height: float = 0.0
    squat_height: float = 0.0
    y_offset_zmp: float = 0.0


sustaina_op_pybullet_walking_config = WalkingConfig(
    plan_max_stride=np.array([0.05, 0.03, 0.2]),
    plan_t_step=0.4,
    control_dt=0.01,
    control_t_preview=1.0,
    control_cost_Q_val=1.0,
    control_cost_R_val=1e-6,
    target_pose_init=np.array([0.4, 0.0, 0.5]),
    actuator_steps=10,
    foot_step_height=0.06,
    squat_height=0.03,
    y_offset_zmp=0.06,
)

robotis_op3_pybullet_walking_config = WalkingConfig(
    plan_max_stride=np.array([0.05, 0.01, 0.2]),
    plan_t_step=0.4,
    control_dt=0.01,
    control_t_preview=1.0,
    control_cost_Q_val=1.0,
    control_cost_R_val=1e-6,
    target_pose_init=np.array([0.4, 0.0, 0.5]),
    actuator_steps=10,
    foot_step_height=0.06,
    squat_height=0.03,
    y_offset_zmp=0.06,
)

robotis_op3_mujoco_walking_config = WalkingConfig(
    plan_max_stride=np.array([0.05, 0.01, np.pi / 8]),
    plan_t_step=0.6,
    control_dt=0.01,
    control_t_preview=1.0,
    control_cost_Q_val=1.0,
    control_cost_R_val=1e-6,
    target_pose_init=np.array([0.1, 0.0, 0.785]),
    actuator_steps=10,
    foot_step_height=0.04,
    squat_height=0.03,
    y_offset_zmp=0.06,
)

base_mujoco_walking_config = WalkingConfig(
    plan_max_stride=np.array([0.05, 0.01, np.pi / 8]),
    plan_t_step=0.6,
    control_dt=0.01,
    control_t_preview=1.0,
    control_cost_Q_val=1.0,
    control_cost_R_val=1e-6,
    target_pose_init=np.array([0.1, 0.0, 0.785]),
    actuator_steps=10,
    foot_step_height=0.04,
    squat_height=0.03,
    y_offset_zmp=0.06,
)

walking_configs = {
    "sustaina_op_pybullet": sustaina_op_pybullet_walking_config,
    "robotis_op3_pybullet": robotis_op3_pybullet_walking_config,
    "robotis_op3_mujoco": robotis_op3_mujoco_walking_config,
    "base_mujoco": base_mujoco_walking_config,
}
