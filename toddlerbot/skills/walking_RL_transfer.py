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
import pickle
import time
from collections import deque

import numpy as np
from tqdm import tqdm
from transforms3d.euler import quat2euler

from toddlerbot.sim.robot import HumanoidRobot
from toddlerbot.utils.math_utils import resample_trajectory, round_floats
from toddlerbot.utils.misc_utils import (
    dump_profiling_data,
    log,
    precise_sleep,
    snake2camel,
)
from toddlerbot.visualization.vis_plot import plot_joint_tracking, plot_joint_velocity


class ToddlerbotLegsCfg:
    class env:
        frame_stack = 15
        num_single_obs = 47
        num_observations = int(frame_stack * num_single_obs)
        num_actions = 12

    class init_state:
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "left_hip_yaw": 0.0,
            "left_hip_roll": 0.0,
            "left_hip_pitch": 0.325,
            "left_knee": 0.65,
            "left_ank_pitch": 0.325,
            "left_ank_roll": 0.0,
            "right_hip_yaw": 0.0,
            "right_hip_roll": 0.0,
            "right_hip_pitch": 0.325,
            "right_knee": -0.65,
            "right_ank_pitch": -0.325,
            "right_ank_roll": 0.0,
        }

    class control:
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim:
        dt = 0.001  # 1000 Hz

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.0

        clip_observations = 18.0
        clip_actions = 18.0

    class rewards:
        cycle_time = 0.64


class cmd:
    vx = 0.4
    vy = 0.0
    dyaw = 0.0


# @profile()
def main(sim, robot, policy, cfg, duration=5.0, debug=False):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    header_name = f"sim2{sim.name}"
    device = next(policy.parameters()).device

    control_dt = cfg.sim.dt * cfg.control.decimation
    joint_ordering = list(cfg.init_state.default_joint_angles.keys())
    default_q = np.array(list(cfg.init_state.default_joint_angles.values()))

    if sim.name == "mujoco":
        sim.run_simulation(headless=True)

        zero_joint_angles, initial_joint_angles = robot.initialize_joint_angles()
        joint_angles_traj = []
        joint_angles_traj.append((0.0, zero_joint_angles))
        joint_angles_traj.append((0.5, initial_joint_angles))
        joint_angles_traj.append((1.5, cfg.init_state.default_joint_angles))
        joint_angles_traj.append((2.0, cfg.init_state.default_joint_angles))
        joint_angles_traj = resample_trajectory(
            joint_angles_traj,
            desired_interval=control_dt,
            interp_type="cubic",
        )
        step_idx = 0
        time_start = time.time()
        while time.time() - time_start < joint_angles_traj[-1][0]:
            step_start = time.time()

            _, joint_angles = joint_angles_traj[
                min(step_idx, len(joint_angles_traj) - 1)
            ]
            sim.set_joint_angles(joint_angles)

            step_idx += 1

            time_until_next_step = control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                precise_sleep(time_until_next_step)

    elif sim.name == "isaac":
        sim.reset_dof_state(default_q)

        sim.run_simulation(headless=True)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs]))

    step_idx = 0
    progress_bar = tqdm(total=round(duration / control_dt), desc=f"Running {sim.name}")

    time_seq_ref = []
    time_seq_dict = {}
    dof_pos_ref_dict = {}
    dof_pos_dict = {}
    dof_vel_dict = {}

    # with open("results/20240507_130013_walk_toddlerbot_legs_isaac/obs.pkl", "rb") as f:
    #     obs_dump = pickle.load(f)

    target_q = np.zeros((cfg.env.num_actions))
    action = np.zeros((cfg.env.num_actions))
    try:
        while step_idx < duration / control_dt:
            step_start = time.time()

            # Obtain an observation
            q_obs, dq, quat, omega = sim.get_observation(joint_ordering)
            q = q_obs - default_q
            eu_ang = np.array(quat2euler(quat))
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            if debug:
                log(
                    f"q: {round_floats(q, 3)}",
                    header=snake2camel(header_name),
                    level="debug",
                )
                log(
                    f"dq: {round_floats(dq, 3)}",
                    header=snake2camel(header_name),
                    level="debug",
                )
                log(f"quat: {quat}", header=snake2camel(header_name), level="debug")
                log(f"omega: {omega}", header=snake2camel(header_name), level="debug")

            obs = np.zeros([1, cfg.env.num_single_obs])
            # step_idx + 1?
            obs[0, 0] = math.sin(
                2 * math.pi * step_idx * control_dt / cfg.rewards.cycle_time
            )
            obs[0, 1] = math.cos(
                2 * math.pi * step_idx * control_dt / cfg.rewards.cycle_time
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

            # policy_input = obs_dump[step_idx]
            policy_input_tensor = torch.tensor(
                policy_input, dtype=torch.float32, device=device
            )
            policy_output = policy(policy_input_tensor)
            action[:] = policy_output[0].detach().cpu().numpy()
            action = np.clip(
                action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions
            )

            action_scaled = action * cfg.control.action_scale
            target_q = action_scaled + default_q

            time_seq_ref.append(step_idx * control_dt)
            joint_angles = {}
            for name, target_angle in zip(joint_ordering, target_q):
                if name not in dof_pos_ref_dict:
                    dof_pos_ref_dict[name] = []

                joint_angles[name] = target_angle
                dof_pos_ref_dict[name].append(target_angle)

            if debug:
                log(
                    f"Joint angles: {joint_angles}",
                    header=snake2camel(header_name),
                    level="debug",
                )

            sim.set_joint_angles(joint_angles)

            joint_state_dict = sim.get_joint_state()
            for name, joint_state in joint_state_dict.items():
                if name not in time_seq_dict:
                    time_seq_dict[name] = []
                    dof_pos_dict[name] = []
                    dof_vel_dict[name] = []

                # Assume the state fetching is instantaneous
                time_seq_dict[name].append(step_idx * control_dt)
                dof_pos_dict[name].append(joint_state.pos)
                dof_vel_dict[name].append(joint_state.vel)

            step_idx += 1
            progress_bar.update(1)

            time_until_next_step = control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                precise_sleep(time_until_next_step)

    except KeyboardInterrupt:
        log("KeyboardInterrupt recieved. Closing...", header=snake2camel(header_name))

    finally:
        sim.close()

        exp_name = f"sim2{sim.name}_{robot.name}"
        time_str = time.strftime("%Y%m%d_%H%M%S")
        exp_folder_path = f"results/{time_str}_{exp_name}"

        os.makedirs(exp_folder_path, exist_ok=True)

        prof_path = os.path.join(exp_folder_path, "profile_output.lprof")
        dump_profiling_data(prof_path)

        if hasattr(sim, "visualizer") and hasattr(sim.visualizer, "save_recording"):
            sim.visualizer.save_recording(exp_folder_path)
        else:
            log(
                "Current visualizer does not support video writing.",
                header=snake2camel(header_name),
                level="warning",
            )

        log("Visualizing...", header="Walking")
        plot_joint_tracking(
            time_seq_dict,
            time_seq_ref,
            dof_pos_dict,
            dof_pos_ref_dict,
            save_path=exp_folder_path,
            file_suffix="",
            motor_params=robot.config.motor_params,
            colors_dict={
                "dynamixel": "cyan",
                "sunny_sky": "oldlace",
                "mighty_zap": "whitesmoke",
            },
        )

        plot_joint_velocity(
            time_seq_dict,
            dof_vel_dict,
            save_path=exp_folder_path,
            file_suffix="",
            motor_params=robot.config.motor_params,
            colors_dict={
                "dynamixel": "cyan",
                "sunny_sky": "oldlace",
                "mighty_zap": "whitesmoke",
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deployment script.")

    parser.add_argument(
        "--robot-name", type=str, required=True, help="Name of the robot."
    )
    parser.add_argument(
        "--sim",
        type=str,
        default="pybullet",
        help="The simulator to use.",
    )
    parser.add_argument(
        "--load-model", type=str, required=True, help="Run to load from."
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        default=False,
        help="Use CPU or CUDA to run the policy.",
    )
    parser.add_argument("--terrain", action="store_true", help="terrain or plane")

    args = parser.parse_args()

    robot = HumanoidRobot(args.robot_name)

    if args.sim == "pybullet":
        from toddlerbot.sim.pybullet_sim import PyBulletSim

        sim = PyBulletSim(robot)
    elif args.sim == "mujoco":
        from toddlerbot.sim.mujoco_sim import MuJoCoSim

        sim = MuJoCoSim(robot)

    elif args.sim == "isaac":
        from toddlerbot.sim.isaac_sim import IsaacSim

        custom_parameters = [
            {"name": "--robot-name", "type": str, "default": args.robot_name},
            {"name": "--sim", "type": str, "default": "pybullet"},
            {"name": "--load-model", "type": str, "default": args.load_model},
            {"name": "--use-cpu", "type": bool, "default": args.use_cpu},
        ]
        sim = IsaacSim(robot, custom_parameters=custom_parameters)

    elif args.sim == "real":
        from toddlerbot.sim.real_world import RealWorld

        sim = RealWorld(robot)
    else:
        raise ValueError("Unknown simulator")

    import torch

    policy = torch.jit.load(args.load_model)
    if not args.use_cpu and torch.cuda.is_available():
        policy = policy.cuda()

    main(sim, robot, policy, ToddlerbotLegsCfg())
