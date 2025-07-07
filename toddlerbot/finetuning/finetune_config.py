import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import gin
import numpy as np


def get_finetune_config(env: str, save_path: str = None):
    gin_file_path = os.path.join(os.path.dirname(__file__), env + ".gin")
    if not os.path.exists(gin_file_path):
        raise FileNotFoundError(f"File {gin_file_path} not found.")

    gin.parse_config_file(gin_file_path)
    finetune_config = FinetuneConfig()
    if save_path is not None:
        with open(os.path.join(save_path, "finetune_config.gin"), "w") as f:
            f.writelines(gin.operative_config_str())
    return finetune_config


@gin.configurable
@dataclass
class FinetuneConfig:
    # obs shape
    num_single_obs: int = 83
    num_single_privileged_obs: int = 122
    # Hidden layer sizes
    policy_hidden_layer_sizes: Tuple[int, ...] = (512, 256, 128)
    value_hidden_layer_sizes: Tuple[int, ...] = (512, 256, 128)
    dynamics_hidden_layer_sizes: Tuple[int, ...] = (512, 256, 128)

    # Evaluation and episode configuration
    num_evals: int = 1000
    ope_rollout_length: int = 200
    eval_rollout_length: int = 1000
    rollout_batch_size: int = 32
    buffer_size: int = 105_000
    buffer_valid_size: int = 5000

    # Pink Noise
    beta: float = 0.0  # Added from argparse (Pink Noise)
    episode_length: int = 1000  # Added from argparse (Pink Noise)
    noise_std_type: str = "learned"

    # Update configuration
    num_updates_per_batch: int = 4
    value_update_steps: int = int(1e4)
    dynamics_update_steps: int = 256
    target_update_freq: int = 2  # Matches the value from the argparse config
    use_tanh: bool = False  # Use tanh transformation for the action distribution

    # Discounting and learning rates
    gamma: float = 0.99
    value_lr: float = 1e-4
    policy_lr: float = 1e-4
    Q_lr: float = 1e-4
    bc_lr: float = 1e-4  # Added from argparse (BehaviorCloning learning rate)
    bppo_lr: float = 2e-4  # Added from argparse (BPPO learning rate)
    dynamics_lr: float = 1e-4

    # Batch sizes
    policy_batch_size: int = 512  # Matches argparse config
    value_batch_size: int = 512  # Overridden to match argparse
    dynamics_batch_size: int = 512
    # Alpha, entropy, and clipping parameters
    kl_alpha: float = 0.1  # Matches argparse (`alpha_bppo`)
    entropy_weight: float = 1e-3
    clipping_epsilon: float = 0.1  # Conflict: argparse uses `clip_ratio=0.25`
    clip_ratio: float = 0.1  # Added from argparse
    is_clip_decay: bool = False  # Added from argparse

    # Decay and scaling parameters
    tau: float = 0.005
    omega: float = 0.5
    decay: float = 0.96  # Added from argparse
    is_bppo_lr_decay: bool = False  # Added from argparse
    decay_steps: int = 50_000_000

    # Exploration and initialization
    update_interval: int = 10_00
    enlarge_when_full: bool = True
    warmup_steps: int = 100_000
    is_linear_decay: bool = True  # Added from argparse

    # Behavior Cloning and BPPO
    bppo_steps: int = 1000  # Added from argparse
    offline_initial_steps: int = 0  # Added from argparse
    offline_total_steps: int = 0
    abppo_update_steps: int = 1  # Added from argparse
    num_policy: int = 4  # Added from argparse
    is_update_old_policy: bool = True  # Added from argparse

    # Evaluation and rendering
    render_interval: int = 50
    eval_step: int = 200  # Added from argparse

    # Miscellaneous
    is_iql: bool = False  # Added from argparse
    update_mode: str = "remote"  # remote or local
    kl_update: bool = False  # Added from argparse
    log_freq: int = 10
    valid_freq: int = 1e3
    frame_stack: int = 15
    use_double_q: bool = True  # Matches argparse (`is_double_q`)
    is_state_norm: bool = False  # Added from argparse
    is_eval_state_norm: bool = False  # Added from argparse
    temperature: Optional[float] = None  # Added from argparse
    kl_strategy: str = "max"  # Added from argparse
    kl_bc: str = "data"  # Added from argparse with description
    scale_strategy: Optional[str] = None  # Added from argparse with description
    is_clip_action: bool = True  # Added from argparse
    sim_vis_type: str = "none"  # Added from argparse

    # Tracking
    vicon_ip: str = "172.24.69.2"
    object_name: str = "arya"
    tracking_tf_matrix: np.ndarray = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    tracking_alpha: float = 0.2
    mocap_marker_offset: np.ndarray = np.array([0.092, 0, -0.031])

    # Healthy range
    healty_ee_force_z: np.ndarray = np.array([-10.0, 40.0])
    healty_ee_force_xy: np.ndarray = np.array([-3.0, 3.0])
    healty_torso_roll: np.ndarray = np.array([-0.5, 0.5])
    healty_torso_pitch: np.ndarray = np.array([-0.5, 0.5])
    pos_error_threshold: float = 0.05
    swing_buffer_size: int = 1000
    action_window_size: int = 30
    symmetric_action: bool = False
    swing_squat: bool = False

    use_residual: bool = False
    residual_action_scale: float = 0.1

    use_latent: bool = False
    optimize_z: bool = False
    optimize_critic: bool = False
    exp_type: str = "walk"

    @gin.configurable
    @dataclass
    class FinetuneRewardsConfig:
        healthy_z_range: List[float] = field(default_factory=lambda: [0.2, 0.4])
        tracking_sigma: float = 100.0
        arm_force_x_sigma: float = 0.1
        arm_force_y_sigma: float = 0.1
        arm_force_z_sigma: float = 0.1
        min_feet_y_dist: float = 0.06
        max_feet_y_dist: float = 0.13
        torso_roll_range: List[float] = field(default_factory=lambda: [-0.1, 0.1])
        torso_pitch_range: List[float] = field(default_factory=lambda: [-0.15, 0.15])

    @gin.configurable
    @dataclass
    class OnlineConfig:
        # base parameters for online PPO
        max_train_step: int = 1e6
        batch_size: int = 1024
        mini_batch_size: int = 128
        K_epochs: int = 20
        gamma: float = 0.99
        lamda: float = 0.95
        epsilon: float = 0.1  # PPO clip ratio
        entropy_coef: float = 0.01
        lr_a: float = 1e-4
        lr_c: float = 1e-4
        num_envs: int = 1

        use_adv_norm: bool = True
        use_grad_clip: bool = True
        use_lr_decay: bool = True
        is_clip_value: bool = True
        is_clip_decay: bool = False
        set_adam_eps: bool = True
        is_state_norm: bool = False
        is_eval_state_norm: bool = False
        is_double_q: bool = True

    @gin.configurable
    @dataclass
    class FinetuneRewardScales:
        torso_pos: float = 0.0
        torso_quat: float = 0.0
        torso_roll: float = 0.0
        torso_pitch: float = 0.0
        torso_yaw_vel: float = 0.0
        lin_vel_x: float = 0.0
        lin_vel_y: float = 0.0
        lin_vel_z: float = 0.0
        ang_vel_x: float = 0.0
        ang_vel_y: float = 0.0
        ang_vel_z: float = 0.0
        neck_motor_pos: float = 0.0
        arm_motor_pos: float = 0.0
        waist_motor_pos: float = 0.0
        leg_motor_pos: float = 0.0
        motor_torque: float = 0.0
        energy: float = 0.0
        action_rate: float = 0.0  # 1e-2
        action_acc: float = 0.0  # 1e-2
        neck_action_rate: float = 0.0  # 1e-2
        neck_action_acc: float = 0.0  # 1e-2
        waist_action_rate: float = 0.0  # 1e-2
        waist_action_acc: float = 0.0  # 1e-2
        leg_action_rate: float = 0.0
        leg_action_acc: float = 0.0
        feet_contact: float = 0.0
        collision: float = 0.0  # 1.0
        survival: float = 0.0
        feet_air_time: float = 0.0
        feet_distance: float = 0.0
        feet_slip: float = 0.0
        feet_clearance: float = 0.0
        stand_still: float = 0.0  # 1.0
        align_ground: float = 0.0  # 1.0
        arm_force_x: float = 0.0
        arm_force_y: float = 0.0
        arm_force_z: float = 0.0
        arm_position: float = 0.0
        fx_sine_amp: float = 0.0
        fx_sine_freq: float = 0.0
        fx: float = 0.0
        fz_sine_amp: float = 0.0
        fz_sine_freq: float = 0.0
        action_symmetry: float = 0.0

    def __post_init__(self):
        self.finetune_reward_scales = self.FinetuneRewardScales()
        self.finetune_rewards = self.FinetuneRewardsConfig()
        self.online = self.OnlineConfig()
