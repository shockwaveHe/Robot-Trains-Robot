from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Define individual dataclasses for each section


@dataclass
class EnvConfig:
    num_envs: int = 1
    num_observations: int = 235
    num_privileged_obs: Optional[int] = None
    env_spacing: float = 3.0
    send_timeouts: bool = True
    episode_length_s: int = 20
    frame_stack: int = 15
    c_frame_stack: int = 3
    num_single_obs: int = 47
    num_single_privileged_obs = 73


@dataclass
class TerrainConfig:
    mesh_type: str = "trimesh"
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    border_size: int = 25
    curriculum: bool = True
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0
    measure_heights: bool = True
    measured_points_x: List[float] = field(
        default_factory=lambda: [
            -0.8,
            -0.7,
            -0.6,
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
        ]
    )
    measured_points_y: List[float] = field(
        default_factory=lambda: [
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ]
    )
    selected: bool = False
    terrain_kwargs: Optional[Dict] = None
    max_init_terrain_level: int = 5
    terrain_length: float = 8.0
    terrain_width: float = 8.0
    num_rows: int = 10
    num_cols: int = 20
    terrain_proportions: List[float] = field(
        default_factory=lambda: [0.1, 0.1, 0.35, 0.25, 0.2]
    )
    slope_threshold: float = 0.75


@dataclass
class CommandRanges:
    lin_vel_x: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    lin_vel_y: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    ang_vel_yaw: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    heading: List[float] = field(default_factory=lambda: [-3.14, 3.14])


@dataclass
class CommandsConfig:
    curriculum: bool = False
    max_curriculum: float = 1.0
    num_commands: int = 4
    resampling_time: float = 10.0
    heading_command: bool = True
    ranges: CommandRanges = CommandRanges()


@dataclass
class InitStateConfig:
    pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    rot: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    lin_vel: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    ang_vel: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class ControlConfig:
    stiffness: Dict[str, float] = field(
        default_factory=lambda: {"joint_a": 10.0, "joint_b": 15.0}
    )
    damping: Dict[str, float] = field(
        default_factory=lambda: {"joint_a": 1.0, "joint_b": 1.5}
    )
    action_scale: float = 0.5
    decimation: int = 4


@dataclass
class AssetConfig:
    file: str = ""
    name: str = "legged_robot"
    foot_name: str = "None"
    penalize_contacts_on: List[str] = field(default_factory=list)
    terminate_after_contacts_on: List[str] = field(default_factory=list)
    disable_gravity: bool = False
    collapse_fixed_joints: bool = True
    fix_base_link: bool = False
    default_dof_drive_mode: int = 3
    self_collisions: int = 0
    replace_cylinder_with_capsule: bool = True
    flip_visual_attachments: bool = True
    density: float = 0.001
    angular_damping: float = 0.0
    linear_damping: float = 0.0
    max_angular_velocity: float = 1000.0
    max_linear_velocity: float = 1000.0
    armature: float = 0.0
    thickness: float = 0.01


@dataclass
class DomainRandConfig:
    push_interval: int = 1500
    push_interval_s: int = 15
    randomize_friction: bool = True
    friction_range: List[float] = field(default_factory=lambda: [0.5, 1.25])
    randomize_base_mass: bool = False
    added_mass_range: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    push_robots: bool = True
    max_push_vel_xy: float = 1.0


@dataclass
class RewardScales:
    termination: float = -0.0
    tracking_lin_vel: float = 1.0
    tracking_ang_vel: float = 0.5
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05
    orientation: float = -0.0
    torques: float = -0.00001
    dof_vel: float = -0.0
    dof_acc: float = -2.5e-7
    base_height: float = -0.0
    feet_air_time: float = 1.0
    collision: float = -1.0
    feet_stumble: float = -0.0
    action_rate: float = -0.0
    stand_still: float = -0.0


@dataclass
class RewardsConfig:
    scales: RewardScales = RewardScales()
    only_positive_rewards: bool = True
    tracking_sigma: float = 0.25
    max_contact_force: float = 100.0


@dataclass
class NormalizationObsScales:
    lin_vel: float = 2.0
    ang_vel: float = 0.25
    dof_pos: float = 1.0
    dof_vel: float = 0.05
    quat: float = 1.0
    height_measurements: float = 5.0


@dataclass
class NormalizationConfig:
    obs_scales: NormalizationObsScales = NormalizationObsScales()
    clip_observations: float = 100.0
    clip_actions: float = 100.0


@dataclass
class NoiseScales:
    dof_pos: float = 0.01
    dof_vel: float = 1.5
    lin_vel: float = 0.1
    ang_vel: float = 0.2
    quat: float = 0.03
    gravity: float = 0.05
    height_measurements: float = 0.1


@dataclass
class NoiseConfig:
    add_noise: bool = True
    noise_level: float = 1.0
    noise_scales: NoiseScales = NoiseScales()


@dataclass
class ViewerConfig:
    ref_env: int = 0
    pos: List[int] = field(default_factory=lambda: [10, 0, 6])
    lookat: List[float] = field(default_factory=lambda: [11.0, 5.0, 3.0])


@dataclass
class PhysXConfig:
    num_threads: int = 10
    solver_type: int = 1
    num_position_iterations: int = 4
    num_velocity_iterations: int = 0
    contact_offset: float = 0.01
    rest_offset: float = 0.0
    bounce_threshold_velocity: float = 0.5
    max_depenetration_velocity: float = 1.0
    max_gpu_contact_pairs: int = 2**23
    default_buffer_size_multiplier: int = 5
    contact_collection: int = 2


@dataclass
class SimConfig:
    dt: float = 0.005
    substeps: int = 1
    gravity: List[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    up_axis: int = 1
    physx: PhysXConfig = PhysXConfig()


# Top-level configuration dataclass


@dataclass
class LeggedRobotCfg:
    env: EnvConfig = EnvConfig()
    terrain: TerrainConfig = TerrainConfig()
    commands: CommandsConfig = CommandsConfig()
    init_state: InitStateConfig = InitStateConfig()
    control: ControlConfig = ControlConfig()
    asset: AssetConfig = AssetConfig()
    domain_rand: DomainRandConfig = DomainRandConfig()
    rewards: RewardsConfig = RewardsConfig()
    normalization: NormalizationConfig = NormalizationConfig()
    noise: NoiseConfig = NoiseConfig()
    viewer: ViewerConfig = ViewerConfig()
    sim: SimConfig = SimConfig()
