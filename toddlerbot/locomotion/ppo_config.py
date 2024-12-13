from dataclasses import dataclass
from typing import Tuple

import gin


@gin.configurable
@dataclass
class PPOConfig:
    policy_hidden_layer_sizes: Tuple[int, ...] = (256, 256, 256)
    value_hidden_layer_sizes: Tuple[int, ...] = (256, 256, 256)
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
    num_envs: int = 1024
    render_eval_interval: int = 50
    render_train_interval: int = 0
    pretrain_value_percent: float = 0.0
    batch_size: int = 256
    num_minibatches: int = 4
    seed: int = 0
