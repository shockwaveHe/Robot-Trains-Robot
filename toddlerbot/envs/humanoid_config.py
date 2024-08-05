#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from typing import List


# Define individual dataclasses for each section
@dataclass
class EnvConfig:
    num_envs: int = 1
    env_spacing: float = 3.0
    episode_length_s: int = 24
    frame_stack: int = 15
    c_frame_stack: int = 3
    num_single_obs: int = 47
    num_single_privileged_obs = 73
    send_timeouts: bool = True
    use_ref_actions: bool = False

    def __post_init__(self):
        self.num_observations = self.frame_stack * self.num_single_obs
        self.num_privileged_obs = self.c_frame_stack * self.num_single_privileged_obs


@dataclass
class InitStateConfig:
    pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # w, x, y, z
    quat: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    lin_vel: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    ang_vel: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class ControlConfig:
    action_scale: float = 0.25
    decimation: int = 10


@dataclass
class CommandRanges:
    lin_vel_x: List[float] = field(default_factory=lambda: [-0.3, 0.6])
    lin_vel_y: List[float] = field(default_factory=lambda: [-0.3, 0.3])
    ang_vel_yaw: List[float] = field(default_factory=lambda: [-0.3, 0.3])
    heading: List[float] = field(default_factory=lambda: [-3.14, 3.14])


@dataclass
class CommandsConfig:
    # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading
    # (in heading mode ang_vel_yaw is recomputed from heading error)
    curriculum: bool = False
    max_curriculum: float = 8.0  # time before command are changed[s]
    num_commands: int = 4
    resampling_time: float = 10.0
    heading_command: bool = True  # if true: compute ang vel command from heading error
    ranges: CommandRanges = CommandRanges()


@dataclass
class DomainRandConfig:
    randomize_friction: bool = True
    friction_range: List[float] = field(default_factory=lambda: [0.1, 2.0])
    randomize_base_mass: bool = True
    added_mass_range: List[float] = field(default_factory=lambda: [-0.1, 0.1])
    push_robots: bool = True
    push_interval_s: int = 4
    max_push_xy_vel: float = 0.2
    max_push_ang_vel: float = 0.4
    max_push_duration: float = 0.2


@dataclass
class RewardScales:
    termination: float = -0.0
    tracking_lin_vel: float = 1.2
    tracking_ang_vel: float = 1.1
    vel_mismatch_exp = 0.5
    low_speed = 0.2
    track_vel_hard = 0.5
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05
    orientation: float = 1.0
    torques: float = -1e-5
    default_dof_pos: float = 0.5  # TODO: change the name of the reward
    dof_pos: float = 1.6  # TODO: change the name of the reward
    dof_vel: float = -5e-4
    dof_acc: float = -1e-7
    base_height: float = 0.2
    base_acc: float = 0.2
    feet_clearance: float = 1.0
    feet_contact_number: float = 1.2
    feet_air_time: float = 1.0
    feet_slip: float = -0.05  # TODO: change the name of the reward
    feet_stumble: float = -0.0
    feet_distance: float = 0.2
    feet_contact_forces: float = -0.01
    collision: float = -1.0
    action_rate: float = -0.0
    action_smoothness: float = -0.002
    stand_still: float = -0.0


@dataclass
class RewardsConfig:
    base_height_target: float = 0.3
    min_dist: float = 0.06
    max_dist: float = 0.15
    target_joint_pos_scale: float = 0.17
    target_feet_height: float = 0.04
    cycle_time: float = 0.64
    only_positive_rewards: bool = True
    tracking_sigma: float = 5.0
    max_contact_force: float = 50.0
    scales: RewardScales = RewardScales()


@dataclass
class NormalizationObsScales:
    lin_vel: float = 2.0
    ang_vel: float = 1.0
    dof_pos: float = 1.0
    dof_vel: float = 0.05
    quat: float = 1.0
    height_measurements: float = 5.0


@dataclass
class NormalizationConfig:
    clip_observations: float = 18.0
    clip_actions: float = 18.0
    scales: NormalizationObsScales = NormalizationObsScales()


@dataclass
class NoiseScales:
    dof_pos: float = 0.05
    dof_vel: float = 0.5
    lin_vel: float = 0.05
    ang_vel: float = 0.1
    quat: float = 0.03
    gravity: float = 0.05
    height_measurements: float = 0.1


@dataclass
class NoiseConfig:
    add_noise: bool = True
    noise_level: float = 0.6
    scales: NoiseScales = NoiseScales()


# Top-level configuration dataclass
@dataclass
class HumanoidCfg:
    env: EnvConfig = EnvConfig()
    init_state: InitStateConfig = InitStateConfig()
    control: ControlConfig = ControlConfig()
    commands: CommandsConfig = CommandsConfig()
    domain_rand: DomainRandConfig = DomainRandConfig()
    rewards: RewardsConfig = RewardsConfig()
    normalization: NormalizationConfig = NormalizationConfig()
    noise: NoiseConfig = NoiseConfig()
