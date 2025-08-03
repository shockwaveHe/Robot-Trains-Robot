import os
from dataclasses import dataclass, field
from typing import List
import gin


def get_env_config(env: str, save_path: str = None) -> "MJXConfig":
    """Retrieves and parses the configuration for a specified environment.

    Args:
        env (str): The name of the environment for which to retrieve the configuration.

    Returns:
        MJXConfig: An instance of MJXConfig initialized with the parsed configuration.

    Raises:
        FileNotFoundError: If the configuration file for the specified environment does not exist.
    """
    gin_file_path = os.path.join(os.path.dirname(__file__), env + ".gin")

    gin.parse_config_file(gin_file_path)
    env_config = MJXConfig()
    if save_path is not None:
        with open(os.path.join(save_path, "env_config.gin"), "w") as f:
            f.writelines(gin.operative_config_str())
    return env_config


@gin.configurable
@dataclass
class MJXConfig:
    """Configuration class for the MJX environment."""

    @gin.configurable
    @dataclass
    class SimConfig:
        timestep: float = 0.004
        solver: int = 2  # Newton
        iterations: int = 1
        ls_iterations: int = 4

    @gin.configurable
    @dataclass
    class ObsConfig:
        frame_stack: int = 15
        c_frame_stack: int = 15
        num_single_obs: int = 83
        num_single_privileged_obs: int = 126

    @gin.configurable
    @dataclass
    class ObsScales:
        lin_vel: float = 2.0
        ang_vel: float = 1.0
        dof_pos: float = 1.0
        dof_vel: float = 0.05
        euler: float = 1.0
        ee_force: float = 1.0
        ee_torque: float = 1.0
        arm_ee_pos: float = 1.0
        # height_measurements: float = 5.0

    @gin.configurable
    @dataclass
    class ActionConfig:
        action_parts: List[str] = field(default_factory=lambda: ["leg"])
        action_scale: float = 0.35
        filter_type: str = "none"
        filter_order: int = 4
        filter_cutoff: float = 10.0
        contact_force_threshold: float = 1.0
        n_steps_delay: int = 1
        n_frames: int = 5
        cycle_time: float = 0.72
        waist_roll_max: float = 0.0

    @gin.configurable
    @dataclass
    class RewardsConfig:
        healthy_z_range: List[float] = field(default_factory=lambda: [0.2, 0.4])
        tracking_sigma: float = 100.0
        arm_force_z_sigma: float = 0.1
        min_feet_y_dist: float = 0.05
        max_feet_y_dist: float = 0.13
        torso_roll_range: List[float] = field(default_factory=lambda: [-0.1, 0.1])
        torso_pitch_range: List[float] = field(default_factory=lambda: [-0.2, 0.2])

    @gin.configurable
    @dataclass
    class RewardScales:
        torso_pos: float = 0.0  # 1.0
        torso_quat: float = 0.0  # 1.0
        torso_roll: float = 0.0
        torso_pitch: float = 0.0
        torso_yaw_vel: float = 0.0
        lin_vel_xy: float = 1.0
        lin_vel_z: float = 1.0
        ang_vel_xy: float = 1.0
        ang_vel_z: float = 1.0
        neck_motor_pos: float = 0.0  # 0.1
        arm_motor_pos: float = 0.0  # 0.1
        waist_motor_pos: float = 0.0  # 50.0
        leg_motor_pos: float = 5.0
        motor_torque: float = 1e-2
        energy: float = 1e-2
        neck_action_rate: float = 0.0  # 1e-2
        neck_action_acc: float = 0.0  # 1e-2
        arm_action_rate: float = 0.0  # 1e-2
        arm_action_acc: float = 0.0  # 1e-2
        waist_action_rate: float = 0.0  # 1e-2
        waist_action_acc: float = 0.0  # 1e-2
        leg_action_rate: float = 0.05
        leg_action_acc: float = 0.05
        feet_contact: float = 1.0
        collision: float = 0.0  # 1.0
        survival: float = 10.0
        feet_air_time: float = 0.0
        feet_distance: float = 0.0
        feet_slip: float = 0.0
        feet_clearance: float = 0.0
        stand_still: float = 0.0  # 1.0
        align_ground: float = 0.0  # 1.0

        def reset(self):
            for key in vars(self):
                setattr(self, key, 0.0)

    @gin.configurable
    @dataclass
    class CommandsConfig:
        resample_time: float = 3.0
        reset_time: float = 100.0  # No resetting by default
        mean_reversion: float = 0.5
        zero_chance: float = 0.2
        turn_chance: float = 0.3
        command_obs_indices: List[int] = field(default_factory=lambda: [])
        command_range: List[List[float]] = field(default_factory=lambda: [[]])
        deadzone: List[float] = field(default_factory=lambda: [])

    @gin.configurable
    @dataclass
    class DomainRandConfig:
        add_domain_rand: bool = True
        friction_range: List[float] = field(default_factory=lambda: [0.5, 2.0])
        damping_range: List[float] = field(default_factory=lambda: [0.5, 1.5])
        armature_range: List[float] = field(default_factory=lambda: [0.5, 1.5])
        frictionloss_range: List[float] = field(default_factory=lambda: [0.5, 1.5])
        gravity_range: List[float] = field(default_factory=lambda: [1.0, 1.0])
        body_mass_range: List[float] = field(default_factory=lambda: [-0.3, 0.3])
        ee_mass_range: List[float] = field(default_factory=lambda: [0.0, 0.1])
        other_mass_range: List[float] = field(default_factory=lambda: [0.0, 0.0])
        push_interval_s: int = 2  # seconds
        push_lin_vel: float = 0.5
        push_ang_vel: float = 1.0

    @gin.configurable
    @dataclass
    class NoiseConfig:
        add_noise: bool = True
        action_noise: float = 0.02
        obs_noise_scale: float = 0.05
        dof_pos: float = 1.0
        dof_vel: float = 2.0
        ang_vel: float = 5.0
        euler: float = 2.0
        tau_limit_scale: float = 0.1
        backlash_scale: float = 0.02
        backlash_activation: float = 0.1

    @gin.configurable
    @dataclass
    class HangConfig:
        use_hang: bool = False
        init_hang_force: float = 0.0
        final_hang_force: float = 0.0
        hang_force_decay_episodes: float = 200.0
        tendon_kp: float = 0.0

    def __init__(self):
        self.sim = self.SimConfig()
        self.obs = self.ObsConfig()
        self.obs_scales = self.ObsScales()
        self.action = self.ActionConfig()
        self.rewards = self.RewardsConfig()
        self.reward_scales = self.RewardScales()
        self.commands = self.CommandsConfig()
        self.domain_rand = self.DomainRandConfig()
        self.noise = self.NoiseConfig()
        self.hang = self.HangConfig()


if __name__ == "__main__":
    from copy import deepcopy

    mjx_config = MJXConfig()
    mjx_config_copy = deepcopy(mjx_config)
    mjx_config_copy.HangConfig.init_hang_force = 1.0
    print(mjx_config.HangConfig.init_hang_force)  # fake deepcopy
    mjx_config_copy.hang.init_hang_force = 1.0
    print(mjx_config.hang.init_hang_force)  # real deepcopy
