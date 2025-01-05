from dataclasses import dataclass
from typing import Tuple

import gin


@gin.configurable
@dataclass
class FinetuneConfig:
    policy_hidden_layer_sizes: Tuple[int, ...] = (256, 256, 256)
    value_hidden_layer_sizes: Tuple[int, ...] = (256, 256, 256)
    num_evals: int = 1000
    episode_length: int = 1000
    unroll_length: int = 20
    num_updates_per_batch: int = 4
    discounting: float = 0.99
    value_lr: float = 1e-4
    value_batch_size: int = 256
    policy_lr: float = 1e-4
    Q_lr: float = 1e-4
    tau: float = 0.005
    omega: float = 0.9
    target_update_freq: int = 2,
    use_double_q: bool = True
    init_steps: int = 10_000
    warmup_steps: int = 100_000
    decay_steps: int = 50_000_000
    alpha: float = 0.1
    entropy_cost: float = 5e-4
    clipping_epsilon: float = 0.1
    num_envs: int = 1024
    render_interval: int = 50
    batch_size: int = 512
    num_minibatches: int = 4
    seed: int = 0
