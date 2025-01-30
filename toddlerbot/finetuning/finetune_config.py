from dataclasses import dataclass
from typing import Optional, Tuple

import gin
import numpy as np

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
    ope_rollout_length: int = 20
    eval_rollout_length: int = 1000
    rollout_batch_size: int = 32
    
    # Update configuration
    num_updates_per_batch: int = 4
    value_update_steps: int = 128
    dynamics_update_steps: int = 128
    target_update_freq: int = 2  # Matches the value from the argparse config
    
    # Discounting and learning rates
    gamma: float = 0.99 
    value_lr: float = 1e-4
    policy_lr: float = 1e-4
    Q_lr: float = 1e-4
    bc_lr: float = 1e-4  # Added from argparse (BehaviorCloning learning rate)
    bppo_lr: float = 1e-4  # Added from argparse (BPPO learning rate)
    dynamics_lr: float = 1e-4 

    # Batch sizes
    policy_batch_size: int = 512  # Matches argparse config
    value_batch_size: int = 512  # Overridden to match argparse
    dynamics_batch_size: int = 512
    # Alpha, entropy, and clipping parameters
    kl_alpha: float = 0.1  # Matches argparse (`alpha_bppo`)
    entropy_weight: float = 5e-4
    clipping_epsilon: float = 0.1  # Conflict: argparse uses `clip_ratio=0.25`
    clip_ratio: float = 0.25  # Added from argparse

    # Decay and scaling parameters
    tau: float = 0.005
    omega: float = 0.9
    decay: float = 0.96  # Added from argparse
    is_clip_decay: bool = True  # Added from argparse
    is_bppo_lr_decay: bool = False  # Added from argparse
    decay_steps: int = 50_000_000
    
    # Exploration and initialization
    update_interval: int = 3_000
    warmup_steps: int = 100_000
    is_linear_decay: bool = True  # Added from argparse

    # Behavior Cloning and BPPO
    bppo_steps: int = int(4e2)  # Added from argparse
    abppo_update_steps: int = 1  # Added from argparse
    num_policy: int = 4  # Added from argparse
    is_update_old_policy: bool = True  # Added from argparse

    # Evaluation and rendering
    render_interval: int = 50
    eval_step: int = 100  # Added from argparse

    # Miscellaneous
    is_iql: bool = True # Added from argparse
    kl_update: bool = False  # Added from argparse
    log_freq: int = 100 
    frame_stack: int = 15
    use_double_q: bool = True  # Matches argparse (`is_double_q`)
    is_state_norm: bool = False  # Added from argparse
    is_eval_state_norm: bool = False  # Added from argparse
    temperature: Optional[float] = None  # Added from argparse
    kl_strategy: str = "max"  # Added from argparse
    kl_bc: str = "data"  # Added from argparse with description
    scale_strategy: Optional[str] = None  # Added from argparse with description
    is_clip_action: bool = False  # Added from argparse
    sim_vis_type: str = "none"  # Added from argparse

    # Healthy range
    healty_ee_force_z: np.ndarray = np.array([-10.0, 40.0])
    healty_ee_force_xy: np.ndarray = np.array([-3.0, 3.0])
    healty_torso_roll: np.ndarray = np.array([-0.5, 0.5])
    healty_torso_pitch: np.ndarray = np.array([-0.5, 0.5])
    pos_error_threshold: float = 0.05