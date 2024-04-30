import argparse
import json
import os
import pickle
import time
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np

from toddlerbot.sim.real_world import RealWorld
from toddlerbot.sim.robot import HumanoidRobot
from toddlerbot.tasks.walking import Walking
from toddlerbot.tasks.walking_configs import walking_configs
from toddlerbot.utils.math_utils import round_floats
from toddlerbot.utils.misc_utils import dump_profiling_data, log, precise_sleep, profile
from toddlerbot.visualization.vis_plot import (
    plot_footsteps,
    plot_joint_tracking,
    plot_line_graph,
)


# @profile()
def main():
    parser = argparse.ArgumentParser(description="Run the walking simulation.")
    parser.add_argument(
        "--robot-name",
        type=str,
        default="sustaina_op",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    parser.add_argument(
        "--sim",
        type=str,
        default="pybullet",
        help="The simulator to use.",
    )
    parser.add_argument(
        "--sleep-time",
        type=float,
        default=0.0,
        help="Time to sleep between steps.",
    )
    args = parser.parse_args()

    exp_name = f"walk_{args.robot_name}_{args.sim}"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    exp_folder_path = f"results/{time_str}_{exp_name}"

    config = walking_configs[f"{args.robot_name}_{args.sim}"]

    robot = HumanoidRobot(args.robot_name)

    walking = Walking(robot, config)

    if args.sim == "pybullet":
        from toddlerbot.sim.pybullet_sim import PyBulletSim

        sim = PyBulletSim(robot)
    elif args.sim == "mujoco":
        from toddlerbot.sim.mujoco_sim import MuJoCoSim

        sim = MuJoCoSim(robot)
    elif args.sim == "real":
        sim = RealWorld(robot)
    else:
        raise ValueError("Unknown simulator")

    torso_pos_init, torso_mat_init = sim.get_torso_pose()
    curr_pose = np.concatenate(
        [torso_pos_init[:2], [np.arctan2(torso_mat_init[1, 0], torso_mat_init[0, 0])]]
    )
    target_pose = np.array(config.target_pose_init)

    path, foot_steps, zmp_ref_traj, zmp_traj, com_ref_traj, joint_angles_traj = (
        walking.plan(curr_pose, target_pose)
    )

    com_traj = []
    # zmp_approx_traj = []

    time_seq_ref = []
    time_seq_dict = {}
    joint_angle_ref_dict = {}
    joint_angle_dict = {}
    actuate_horizon = 0

    if sim.name == "mujoco":
        vis_data = {
            "foot_steps": foot_steps,
            "com_ref_traj": com_ref_traj,
            "torso": None,
        }
        sim.run_simulation(headless=True, vis_data=vis_data)

    time_start = time.time()
    duration_max = 60
    try:
        step_idx = 0
        while time.time() - time_start < duration_max:
            step_start = time.time()

            time_ref, joint_angles_ref = joint_angles_traj[
                min(step_idx, len(joint_angles_traj) - 1)
            ]
            time_seq_ref.append(time_ref)
            for name, angle in joint_angles_ref.items():
                if name not in joint_angle_ref_dict:
                    joint_angle_ref_dict[name] = []
                joint_angle_ref_dict[name].append(angle)

            _, joint_angles = joint_angles_traj[
                min(step_idx + actuate_horizon, len(joint_angles_traj) - 1)
            ]

            sim.set_joint_angles(joint_angles)

            joint_state_dict = sim.get_joint_state()
            for name, joint_state in joint_state_dict.items():
                if name not in time_seq_dict:
                    time_seq_dict[name] = []
                    joint_angle_dict[name] = []

                # Assume the state fetching is instantaneous
                time_seq_dict[name].append(step_idx * config.control_dt)
                joint_angle_dict[name].append(joint_state.pos)

            torso_pos, torso_mat = sim.get_torso_pose()
            torso_mat_delta = torso_mat @ torso_mat_init.T
            torso_theta = np.arctan2(torso_mat_delta[1, 0], torso_mat_delta[0, 0])
            # print(f"torso_pos: {torso_pos}, torso_theta: {torso_theta}")

            com_pos = [
                torso_pos[0] + np.cos(torso_theta) * walking.x_offset_com_to_foot,
                torso_pos[1] + np.sin(torso_theta) * walking.x_offset_com_to_foot,
            ]
            com_traj.append(com_pos)
            # zmp_pos = sim.get_zmp(com_pos)
            # zmp_approx_traj.append(zmp_pos)

            if step_idx >= len(joint_angles_traj):
                tracking_error = np.array(target_pose) - np.array(
                    [*torso_pos[:2], torso_theta]
                )
                log(
                    f"Tracking error: {round_floats(tracking_error, 6)}",
                    header="Walking",
                )

            step_idx += 1

            time_until_next_step = 1 / config.speed_factor * config.control_dt - (
                time.time() - step_start
            )
            if time_until_next_step > 0:
                precise_sleep(time_until_next_step)

    except KeyboardInterrupt:
        log("KeyboardInterrupt recieved. Closing...", header="Walking")

    finally:
        sim.close()

        log("Saving config and data...", header="Walking")
        os.makedirs(exp_folder_path, exist_ok=True)

        with open(os.path.join(exp_folder_path, "config.json"), "w") as f:
            json.dump(asdict(config), f, indent=4)

        prof_path = os.path.join(exp_folder_path, "profile_output.lprof")
        dump_profiling_data(prof_path)

        if hasattr(sim, "visualizer") and hasattr(sim.visualizer, "save_recording"):
            sim.visualizer.save_recording(exp_folder_path)
        else:
            log(
                "Current visualizer does not support video writing.",
                header="Walking",
                level="warning",
            )

        robot_state_traj_data = {
            "zmp_ref_traj": zmp_ref_traj,
            "zmp_traj": zmp_traj,
            # "zmp_approx_traj": zmp_approx_traj,
            "com_ref_traj": com_ref_traj,
            "com_traj": com_traj,
        }

        file_suffix = ""
        if len(file_suffix) > 0:
            robot_state_traj_file_name = f"robot_state_traj_data_{file_suffix}.pkl"
        else:
            robot_state_traj_file_name = "robot_state_traj_data.pkl"

        with open(os.path.join(exp_folder_path, robot_state_traj_file_name), "wb") as f:
            pickle.dump(robot_state_traj_data, f)

        log("Visualizing...", header="Walking")
        plot_joint_tracking(
            time_seq_dict,
            time_seq_ref,
            joint_angle_dict,
            joint_angle_ref_dict,
            save_path=exp_folder_path,
            file_suffix=file_suffix,
            motor_params=robot.config.motor_params,
            colors_dict={
                "dynamixel": "cyan",
                "sunny_sky": "oldlace",
                "mighty_zap": "whitesmoke",
            },
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_aspect("equal")

        plot_footsteps(
            path,
            foot_steps,
            robot.foot_size[:2],
            robot.offsets["y_offset_com_to_foot"],
            title="Footsteps Planning",
            save_path=exp_folder_path,
            file_suffix=file_suffix,
            ax=ax,
        )()

        plot_line_graph(
            [[record[1] for record in x] for x in robot_state_traj_data.values()],
            x=[[record[0] for record in x] for x in robot_state_traj_data.values()],
            title="Footsteps Planning",
            x_label="X",
            y_label="Y",
            save_config=True,
            save_path=exp_folder_path,
            file_suffix=file_suffix,
            legend_labels=list(robot_state_traj_data.keys()),
            ax=ax,
        )()


if __name__ == "__main__":
    main()
