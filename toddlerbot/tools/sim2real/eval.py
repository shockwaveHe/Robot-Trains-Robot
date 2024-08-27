import argparse
import json
import os
import pickle
import time
from typing import Dict, List

import numpy as np
import numpy.typing as npt

from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.visualization.vis_plot import (
    plot_ang_vel_gap,
    plot_euler_gap,
    plot_joint_angle_tracking,
    plot_sim2real_gap,
)


def load_datasets(
    robot: Robot, sim_data_path: str, real_data_path: str, idx: int = 1000
):
    # Use glob to find all pickle files matching the pattern
    sim_data: Dict[str, Dict[str, npt.NDArray[np.float32]]] = {}
    real_data: Dict[str, Dict[str, npt.NDArray[np.float32]]] = {}
    for data_dict, data_path in zip(
        [sim_data, real_data], [sim_data_path, real_data_path]
    ):
        pickle_file_path = os.path.join(data_path, "log_data.pkl")
        if not os.path.exists(pickle_file_path):
            raise ValueError("No data files found")

        with open(pickle_file_path, "rb") as f:
            log_data_dict = pickle.load(f)

        obs_list: List[Obs] = log_data_dict["obs_list"]
        motor_angles_list: List[Dict[str, float]] = log_data_dict["motor_angles_list"]

        data_dict["imu"] = {
            "euler": np.array([obs.euler for obs in obs_list[:idx]]),
            "ang_vel": np.array([obs.ang_vel for obs in obs_list[:idx]]),
        }

        for joint_name in robot.joint_ordering:
            data_dict[joint_name] = {}
            data_dict[joint_name]["obs_time"] = np.array(
                [obs.time for obs in obs_list[:idx]]
            )
            data_dict[joint_name]["obs_pos"] = np.array(
                [obs.q for obs in obs_list[:idx]]
            )
            data_dict[joint_name]["obs_vel"] = np.array(
                [obs.dq for obs in obs_list[:idx]]
            )
            data_dict[joint_name]["action"] = np.array(
                [
                    list(motor_angles.values())
                    for motor_angles in motor_angles_list[:idx]
                ]
            )

    return sim_data, real_data


def evaluate(
    robot: Robot,
    sim_data: Dict[str, Dict[str, npt.NDArray[np.float32]]],
    real_data: Dict[str, Dict[str, npt.NDArray[np.float32]]],
    exp_folder_path: str,
):
    time_seq_sim_dict: Dict[str, List[float]] = {}
    time_seq_real_dict: Dict[str, List[float]] = {}
    joint_pos_sim_dict: Dict[str, List[float]] = {}
    joint_pos_real_dict: Dict[str, List[float]] = {}
    joint_vel_sim_dict: Dict[str, List[float]] = {}
    joint_vel_real_dict: Dict[str, List[float]] = {}
    action_sim_dict: Dict[str, List[float]] = {}
    action_real_dict: Dict[str, List[float]] = {}

    rmse_pos_dict: Dict[str, float] = {}
    rmse_vel_dict: Dict[str, float] = {}
    rmse_action_dict: Dict[str, float] = {}

    for joint_name in sim_data:
        if joint_name == "imu":
            continue

        joint_idx = robot.joint_ordering.index(joint_name)
        obs_pos_sim = sim_data[joint_name]["obs_pos"][:, joint_idx]
        obs_pos_real = real_data[joint_name]["obs_pos"][:, joint_idx]
        obs_vel_sim = sim_data[joint_name]["obs_vel"][:, joint_idx]
        obs_vel_real = real_data[joint_name]["obs_vel"][:, joint_idx]
        action_sim = sim_data[joint_name]["action"][:, joint_idx]
        action_real = real_data[joint_name]["action"][:, joint_idx]

        time_seq_sim_dict[joint_name] = sim_data[joint_name]["obs_time"].tolist()
        time_seq_real_dict[joint_name] = real_data[joint_name]["obs_time"].tolist()

        joint_pos_sim_dict[joint_name] = obs_pos_sim.tolist()
        joint_pos_real_dict[joint_name] = obs_pos_real.tolist()
        joint_vel_sim_dict[joint_name] = obs_vel_sim.tolist()
        joint_vel_real_dict[joint_name] = obs_vel_real.tolist()
        action_sim_dict[joint_name] = action_sim.tolist()
        action_real_dict[joint_name] = action_real.tolist()

        rmse_pos_dict[joint_name] = np.sqrt(np.mean((obs_pos_real - obs_pos_sim) ** 2))
        rmse_vel_dict[joint_name] = np.sqrt(np.mean((obs_vel_real - obs_vel_sim) ** 2))
        rmse_action_dict[joint_name] = np.sqrt(np.mean((action_real - action_sim) ** 2))

    plot_euler_gap(
        time_seq_sim_dict[list(sim_data.keys())[-1]],
        time_seq_real_dict[list(sim_data.keys())[-1]],
        sim_data["imu"]["euler"],
        real_data["imu"]["euler"],
        save_path=exp_folder_path,
    )

    plot_ang_vel_gap(
        time_seq_sim_dict[list(sim_data.keys())[-1]],
        time_seq_real_dict[list(sim_data.keys())[-1]],
        sim_data["imu"]["ang_vel"],
        real_data["imu"]["ang_vel"],
        save_path=exp_folder_path,
    )

    for rmse_dict, label in zip(
        [rmse_pos_dict, rmse_vel_dict, rmse_action_dict],
        ["joint_pos", "joint_vel", "action"],
    ):
        plot_sim2real_gap(
            rmse_dict,
            label,
            save_path=exp_folder_path,
            file_name="sim2real_gap_" + label,
        )

    plot_joint_angle_tracking(
        time_seq_sim_dict,
        time_seq_real_dict,
        joint_pos_sim_dict,
        joint_pos_real_dict,
        robot.joint_limits,
        save_path=exp_folder_path,
        file_name="sim2real_joint_pos",
        set_ylim=False,
        line_suffix=["_sim", "_real"],
    )

    plot_joint_angle_tracking(
        time_seq_sim_dict,
        time_seq_real_dict,
        joint_vel_sim_dict,
        joint_vel_real_dict,
        robot.joint_limits,
        save_path=exp_folder_path,
        file_name="sim2real_joint_vel",
        set_ylim=False,
        line_suffix=["_sim", "_real"],
    )

    plot_joint_angle_tracking(
        time_seq_sim_dict,
        time_seq_real_dict,
        action_sim_dict,
        action_real_dict,
        robot.joint_limits,
        save_path=exp_folder_path,
        file_name="sim2real_action",
        set_ylim=False,
        line_suffix=["_sim", "_real"],
    )


def main():
    parser = argparse.ArgumentParser(description="Run the SysID optimization.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    parser.add_argument(
        "--sim-data",
        type=str,
        default="",
        required=True,
        help="The name of the run.",
    )
    parser.add_argument(
        "--real-data",
        type=str,
        default="",
        required=True,
        help="The name of the run.",
    )
    args = parser.parse_args()

    sim_data_path = os.path.join("results", args.sim_data)
    if not os.path.exists(sim_data_path):
        raise ValueError("Invalid sim experiment folder path")

    real_data_path = os.path.join("results", args.real_data)
    if not os.path.exists(real_data_path):
        raise ValueError("Invalid real experiment folder path")

    robot = Robot(args.robot)

    policy_name = args.sim_data.split("_")[1]

    exp_name = f"{robot.name}_{policy_name}_sim2real_eval"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    exp_folder_path = f"results/{exp_name}_{time_str}"

    os.makedirs(exp_folder_path, exist_ok=True)

    with open(os.path.join(exp_folder_path, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    sim_data, real_data = load_datasets(robot, sim_data_path, real_data_path)

    ##### Evaluate the optimized parameters in the simulation ######
    evaluate(robot, sim_data, real_data, exp_folder_path)


if __name__ == "__main__":
    main()
