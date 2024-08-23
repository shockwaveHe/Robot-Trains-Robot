from dataclasses import dataclass, field
from typing import List


@dataclass
class MJConfig:
    timestep: float = 0.002
    solver: int = 2  # Newton
    iterations: int = 1
    ls_iterations: int = 4


@dataclass
class ObsScales:
    lin_vel: float = 2.0
    ang_vel: float = 1.0
    dof_pos: float = 1.0
    dof_vel: float = 0.05
    euler: float = 1.0
    # height_measurements: float = 5.0


@dataclass
class ObsConfig:
    frame_stack: int = 15
    c_frame_stack: int = 15  # 3
    num_single_obs: int = 101
    num_single_privileged_obs: int = 138
    scales: ObsScales = ObsScales()


@dataclass
class ActionConfig:
    action_scale: float = 0.25
    contact_force_threshold: float = 5.0
    cycle_time: float = 0.6
    n_frames: int = 6


@dataclass
class RewardScales:
    torso_pos: float = 0.0  # 1.0
    torso_quat: float = 1.0
    lin_vel_xy: float = 1.0
    lin_vel_z: float = 0.5
    ang_vel_xy: float = 0.5
    ang_vel_z: float = 0.5
    leg_joint_pos: float = 0.0  # 5.0
    leg_joint_vel: float = 0.0  # 1e-4
    arm_joint_pos: float = 0.0  # 0.1
    arm_joint_vel: float = 0.0  # 1e-3
    neck_joint_pos: float = 0.0  # 0.1
    neck_joint_vel: float = 0.0  # 1e-3
    waist_joint_pos: float = 0.0  # 50.0
    waist_joint_vel: float = 0.0  # 1e-3
    feet_air_time: float = 10.0
    feet_clearance: float = 0.0  # 1.0 # Doesn't help
    feet_contact: float = 0.5
    feet_distance: float = 0.5
    feet_slip: float = 0.0  # 0.1
    stand_still: float = 0.0  # 1.0
    joint_torque: float = 0.0  # 5e-2
    joint_acc: float = 0.0  # 5e-7
    leg_action_rate: float = 0.0  # 1e-2
    leg_action_acc: float = 0.0  # 1e-2
    arm_action_rate: float = 0.0  # 1e-2
    arm_action_acc: float = 0.0  # 1e-2
    neck_action_rate: float = 0.0  # 1e-2
    neck_action_acc: float = 0.0  # 1e-2
    waist_action_rate: float = 0.0  # 1e-2
    waist_action_acc: float = 0.0  # 1e-2
    collision: float = 0.0  # 1.0
    survival: float = 10.0


@dataclass
class RewardsConfig:
    healthy_z_range: List[float] = field(default_factory=lambda: [0.2, 0.4])
    min_feet_distance: float = 0.05
    max_feet_distance: float = 0.15
    target_feet_z_delta: float = 0.03
    scales: RewardScales = RewardScales()


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
    # curriculum: bool = False
    # max_curriculum: float = 8.0  # time before command are changed[s]
    num_commands: int = 4
    resample_time: float = 5.0
    # if true: compute ang vel command from heading error
    has_heading_command: bool = True
    ranges: CommandRanges = CommandRanges()


@dataclass
class DomainRandConfig:
    randomize_friction: bool = True
    friction_range: List[float] = field(default_factory=lambda: [0.1, 2.0])
    randomize_base_mass: bool = True
    added_mass_range: List[float] = field(default_factory=lambda: [-0.1, 0.1])
    push_robots: bool = False  # TODO: change it back to True
    push_interval_s: int = 4
    max_push_xy_vel: float = 0.2
    max_push_ang_vel: float = 0.4
    max_push_duration: float = 0.2
    dynamic_randomization: float = 0.02


@dataclass
class NoiseConfig:
    add_noise: bool = True
    noise_scale: float = 0.05
    dof_pos: float = 1.0
    dof_vel: float = 1.0
    ang_vel: float = 0.2
    euler: float = 1.0


# Top-level configuration dataclass
@dataclass
class MuJoCoConfig:
    mj: MJConfig = MJConfig()
    obs: ObsConfig = ObsConfig()
    action: ActionConfig = ActionConfig()
    rewards: RewardsConfig = RewardsConfig()
    commands: CommandsConfig = CommandsConfig()
    domain_rand: DomainRandConfig = DomainRandConfig()
    noise: NoiseConfig = NoiseConfig()
