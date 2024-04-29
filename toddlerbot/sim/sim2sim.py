# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import argparse
import math
import os
import time
from collections import deque

import numpy as np
from humanoid.envs import ToddlerbotLegsCfg
from tqdm import tqdm
from transforms3d.euler import quat2euler

from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import HumanoidRobot
from toddlerbot.utils.misc_utils import precise_sleep


class cmd:
    vx = 0.4
    vy = 0.0
    dyaw = 0.0


def run_mujoco(robot, policy, cfg, sim_duration=30.0):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    sim = MuJoCoSim(robot)
    sim.run_simulation(headless=True)

    target_q = np.zeros((cfg.env.num_actions))
    action = np.zeros((cfg.env.num_actions))

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs]))

    step_idx = 0
    time_start = time.time()
    progress_bar = tqdm(
        total=round(sim_duration / (cfg.control.decimation * cfg.sim.dt)),
        desc="Running simulation",
    )

    while time.time() - time_start < sim_duration:
        step_start = time.time()

        # Obtain an observation
        q, dq, quat, v, omega, gvec = sim.get_observation()
        q = q[-cfg.env.num_actions :]
        dq = dq[-cfg.env.num_actions :]

        obs = np.zeros([1, cfg.env.num_single_obs])
        eu_ang = np.array(quat2euler(quat))
        eu_ang[eu_ang > math.pi] -= 2 * math.pi

        obs[0, 0] = math.sin(
            2 * math.pi * step_idx * cfg.sim.dt / cfg.rewards.cycle_time
        )
        obs[0, 1] = math.cos(
            2 * math.pi * step_idx * cfg.sim.dt / cfg.rewards.cycle_time
        )
        obs[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
        obs[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel
        obs[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
        obs[0, 5:17] = q * cfg.normalization.obs_scales.dof_pos
        obs[0, 17:29] = dq * cfg.normalization.obs_scales.dof_vel
        obs[0, 29:41] = action
        obs[0, 41:44] = omega
        obs[0, 44:47] = eu_ang

        obs = np.clip(
            obs,
            -cfg.normalization.clip_observations,
            cfg.normalization.clip_observations,
        )

        hist_obs.append(obs)
        hist_obs.popleft()

        policy_input = np.zeros([1, cfg.env.num_observations])
        for i in range(cfg.env.frame_stack):
            policy_input[
                0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs
            ] = hist_obs[i][0, :]

        policy_output = policy(torch.tensor(policy_input, dtype=torch.float32))
        action[:] = policy_output[0].detach().numpy()
        action = np.clip(
            action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions
        )

        target_q = action * cfg.control.action_scale

        sim.set_joint_angles(target_q)

        step_idx += 1
        progress_bar.update(1)

        time_until_next_step = cfg.sim.dt * cfg.control.decimation - (
            time.time() - step_start
        )
        if time_until_next_step > 0:
            precise_sleep(time_until_next_step)

    sim.close()

    exp_name = f"sim2sim_{robot.name}"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    exp_folder_path = f"results/{time_str}_{exp_name}"

    os.makedirs(exp_folder_path, exist_ok=True)

    if hasattr(sim, "visualizer") and hasattr(sim.visualizer, "save_recording"):
        sim.visualizer.save_recording(exp_folder_path)
    else:
        print("Current visualizer does not support video writing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deployment script.")

    parser.add_argument(
        "--robot-name", type=str, required=True, help="Name of the robot."
    )
    parser.add_argument(
        "--load-model", type=str, required=True, help="Run to load from."
    )
    parser.add_argument(
        "--terrain", action="store_true", default=False, help="terrain or plane"
    )

    args = parser.parse_args()

    robot = HumanoidRobot(args.robot_name)

    import torch

    policy = torch.jit.load(args.load_model)

    run_mujoco(robot, policy, ToddlerbotLegsCfg())
