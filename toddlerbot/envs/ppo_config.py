from dataclasses import dataclass


@dataclass
class PPOConfig:
    num_timesteps: int = 200_000_000
    num_evals: int = 1000
    reward_scaling: float = 1.0
    episode_length: int = 1000
    action_repeat: int = 1
    unroll_length: int = 20
    num_minibatches: int = 4
    num_updates_per_batch: int = 4
    discounting: float = 0.97
    learning_rate: float = 1e-4
    entropy_cost: float = 1e-4
    num_envs: int = 1024
    batch_size: int = 256
    seed: int = 0
