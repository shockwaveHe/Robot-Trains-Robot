from dataclasses import dataclass
from typing import Tuple

import gin


@gin.configurable
@dataclass
class PPOConfig:
    """Data class for storing PPO hyperparameters."""

    policy_hidden_layer_sizes: Tuple[int, ...] = (512, 256, 128)
    value_hidden_layer_sizes: Tuple[int, ...] = (512, 256, 128)
    num_timesteps: int = 100_000_000
    num_evals: int = 1000
    episode_length: int = 1000
    unroll_length: int = 20
    num_updates_per_batch: int = 4
    discounting: float = 0.97
    learning_rate: float = 1e-4
    decay_steps: int = 50_000_000
    alpha: float = 0.1
    entropy_cost: float = 5e-4
    clipping_epsilon: float = 0.1
    num_train_envs: int = 1000
    num_eval_envs: int = 24
    num_envs: int = 1024
    render_interval: int = 50
    batch_size: int = 250
    num_minibatches: int = 4
    seed: int = 0
    use_base_policy: bool = False
