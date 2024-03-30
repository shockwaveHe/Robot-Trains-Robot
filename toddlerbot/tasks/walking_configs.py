from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class WalkingConfig:
    """Data class to hold walking parameters."""

    plan_max_stride: list = field(default_factory=lambda: [0.05, 0.01, np.pi / 8])
    plan_t_step: float = 0.6
    control_dt: float = 0.01
    control_t_preview: float = 1.0
    control_t_filter: float = 0.5
    control_cost_Q_val: float = 1.0
    control_cost_R_val: float = 1e-6
    target_pose_init: list = field(default_factory=lambda: [0.2, 0.0, 0.785])
    actuator_steps: int = 10
    foot_step_height: float = 0.04
    squat_height: float = 0.03
    y_offset_zmp: float = 0.06
    filter_dynamics: bool = False
    rotate_torso: bool = False

    @staticmethod
    def create_config(**kwargs):
        """Helper method to create a new WalkingConfig with specified overrides."""
        config = WalkingConfig()
        for key, value in kwargs.items():
            setattr(config, key, value)
        return config


# Creating configurations using the helper method
walking_configs = {
    "sustaina_op_pybullet": WalkingConfig.create_config(
        plan_max_stride=[0.05, 0.03, 0.2],
        plan_t_step=0.4,
        target_pose_init=[0.4, 0.0, 0.5],
        foot_step_height=0.06,
    ),
    "robotis_op3_pybullet": WalkingConfig.create_config(
        plan_max_stride=[0.05, 0.01, 0.2],
        plan_t_step=0.4,
        target_pose_init=[0.4, 0.0, 0.5],
        foot_step_height=0.06,
    ),
    "robotis_op3_mujoco": WalkingConfig.create_config(),
    "base_mujoco": WalkingConfig.create_config(
        squat_height=0.01,
        # rotate_torso=True,
        # filter_dynamics=True
    ),
    "base_legs_mujoco": WalkingConfig.create_config(squat_height=0.01),
    "base_real": WalkingConfig.create_config(squat_height=0.01, actuator_steps=2),
}
