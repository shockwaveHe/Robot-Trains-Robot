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
        num_single_privileged_obs=146,
    ),
    rewards=RewardsConfig(
        scales=RewardScales(
            # tracking_lin_vel=0.0,
            # tracking_ang_vel=0.0,
            # low_speed=0.0,
            # orientation=0.0,
            default_dof_pos=0.0,
            dof_pos=0.0,
            # dof_vel=0.0,
            # dof_acc=0.0,
            # base_height=0.0,
            # base_acc=0.0,
            feet_air_time=0.0,
            feet_clearance=0.0,
            feet_contact_forces=0.0,
            feet_contact_number=0.0,
            feet_distance=0.0,
            feet_slip=0.0,
            # collision=0.0,
            # action_smoothness=0.0,
        ),
    ),
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
