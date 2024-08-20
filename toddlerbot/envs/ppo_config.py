from dataclasses import dataclass
from typing import Tuple


@dataclass
class PPOConfig:
    policy_hidden_layer_sizes: Tuple[int, ...] = (128,) * 4
    value_hidden_layer_sizes: Tuple[int, ...] = (128,) * 4
    num_timesteps: int = 100_000_000
    num_evals: int = 1000
    episode_length: int = 1000
    unroll_length: int = 20
    num_minibatches: int = 4
    num_updates_per_batch: int = 4
    discounting: float = 0.97
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-5
    transition_steps: int = 10_000_000
    entropy_cost: float = 1e-4
    num_envs: int = 1024
    batch_size: int = 256
    seed: int = 0
