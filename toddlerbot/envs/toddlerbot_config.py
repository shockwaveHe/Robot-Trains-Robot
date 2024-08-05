#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from toddlerbot.envs.humanoid_config import (
    EnvConfig,
    HumanoidCfg,
    RewardScales,
    RewardsConfig,
)
from toddlerbot.envs.ppo_config import AlgorithmConfig, PPOCfg, RunnerConfig

toddlerbot_cfg = HumanoidCfg(
    env=EnvConfig(
        num_envs=1,
        num_single_obs=101,
        num_single_privileged_obs=143,
    ),
    rewards=RewardsConfig(scales=RewardScales()),
)


toddlerbot_ppo_cfg = PPOCfg(
    algorithm=AlgorithmConfig(
        entropy_coef=0.001,
        learning_rate=1e-5,
        num_learning_epochs=2,
        gamma=0.994,
        lam=0.9,
    ),
    runner=RunnerConfig(
        num_steps_per_env=60,
        max_iterations=3001,
        experiment_name="walk_toddlerbot",
        run_name="v0.2",
        resume=False,  # load and resume
        load_run="",  # -1 = last run
        checkpoint=-1,  # -1 = last saved model
        resume_path=None,  # updated from load_run and chkpt
    ),
)
