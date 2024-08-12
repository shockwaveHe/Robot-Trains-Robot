from dataclasses import dataclass


@dataclass
class PPOConfig:
    num_timesteps: int = 10_000_000
    num_evals: int = 10
    reward_scaling: float = 1.0
    episode_length: int = 1000
    action_repeat: int = 1
    unroll_length: int = 20
    num_minibatches: int = 16
    num_updates_per_batch: int = 4
    discounting: float = 0.97
    learning_rate: float = 3.0e-4
    entropy_cost: float = 1e-2
    num_envs: int = 1024
    batch_size: int = 64
    seed: int = 0
