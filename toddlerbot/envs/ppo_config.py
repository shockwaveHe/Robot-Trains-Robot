#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from typing import List, Optional

# Define dataclasses for each configuration section


@dataclass
class PolicyConfig:
    init_noise_std: float = 1.0
    actor_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])


@dataclass
class AlgorithmConfig:
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"  # could be "adaptive" or "fixed"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


@dataclass
class RunnerConfig:
    policy_class_name: str = "ActorCritic"
    algorithm_class_name: str = "PPO"
    num_steps_per_env: int = 24
    max_iterations: int = 1500
    save_interval: int = 100
    experiment_name: str = "test"
    run_name: str = ""
    resume: bool = False
    load_run: str = ""
    checkpoint: int = -1
    resume_path: Optional[str] = None


@dataclass
class PPOCfg:
    runner_class_name: str = "OnPolicyRunner"
    policy: PolicyConfig = PolicyConfig()
    algorithm: AlgorithmConfig = AlgorithmConfig()
    runner: RunnerConfig = RunnerConfig()
