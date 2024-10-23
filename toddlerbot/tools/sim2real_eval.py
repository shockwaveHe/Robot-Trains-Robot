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
    plot_joint_tracking,
    plot_joint_tracking_frequency,
    # plot_sim2real_gap_bar,
    plot_sim2real_gap_line,
)


def load_datasets(robot: Robot, sim_data_path: str, real_data_path: str):
    # Use glob to find all pickle files matching the pattern
    sim_pickle_file_path = os.path.join(sim_data_path, "log_data.pkl")
    if not os.path.exists(sim_pickle_file_path):
        raise ValueError("No data files found")

    with open(sim_pickle_file_path, "rb") as f:
        sim_log_data_dict = pickle.load(f)

    sim_obs_list: List[Obs] = sim_log_data_dict["obs_list"]
    sim_motor_angles_list: List[Dict[str, float]] = sim_log_data_dict[
        "motor_angles_list"
    ]

    real_pickle_file_path = os.path.join(real_data_path, "log_data.pkl")
    if not os.path.exists(real_pickle_file_path):
        raise ValueError("No data files found")

    with open(real_pickle_file_path, "rb") as f:
        real_log_data_dict = pickle.load(f)

    real_obs_list: List[Obs] = real_log_data_dict["obs_list"]
    real_motor_angles_list: List[Dict[str, float]] = real_log_data_dict[
        "motor_angles_list"
    ]
    idx = min(len(sim_obs_list), len(real_obs_list))

    sim_data: Dict[str, Dict[str, npt.NDArray[np.float32]]] = {}
    real_data: Dict[str, Dict[str, npt.NDArray[np.float32]]] = {}

    for data_dict, obs_list, motor_angles_list in zip(
        [sim_data, real_data],
        [sim_obs_list, real_obs_list],
        [sim_motor_angles_list, real_motor_angles_list],
    ):
        data_dict["imu"] = {
            # "lin_vel": np.array([obs.lin_vel for obs in obs_list[:idx]]),
            "ang_vel": np.array([obs.ang_vel for obs in obs_list[:idx]]),
            "euler": np.array([obs.torso_euler for obs in obs_list[:idx]]),
        }
        for motor_name in robot.motor_ordering:
            data_dict[motor_name] = {}
            data_dict[motor_name]["time"] = np.array(
                [obs.time for obs in obs_list[:idx]]
            )
            data_dict[motor_name]["pos"] = np.array(
                [obs.motor_pos for obs in obs_list[:idx]]
            )
            data_dict[motor_name]["vel"] = np.array(
                [obs.motor_vel for obs in obs_list[:idx]]
            )
            data_dict[motor_name]["action"] = np.array(
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
    motor_pos_sim_dict: Dict[str, List[float]] = {}
    motor_pos_real_dict: Dict[str, List[float]] = {}
    motor_vel_sim_dict: Dict[str, List[float]] = {}
    motor_vel_real_dict: Dict[str, List[float]] = {}
    action_sim_dict: Dict[str, List[float]] = {}
    action_real_dict: Dict[str, List[float]] = {}

    rmse_pos_dict: Dict[str, float] = {}
    rmse_vel_dict: Dict[str, float] = {}
    rmse_action_dict: Dict[str, float] = {}

    for motor_name in sim_data:
        if motor_name == "imu":
            continue

        motor_idx = robot.motor_ordering.index(motor_name)

        motor_pos_sim = sim_data[motor_name]["pos"][:, motor_idx]
        motor_pos_real = real_data[motor_name]["pos"][:, motor_idx]
        motor_vel_sim = sim_data[motor_name]["vel"][:, motor_idx]
        motor_vel_real = real_data[motor_name]["vel"][:, motor_idx]
        action_sim = sim_data[motor_name]["action"][:, motor_idx]
        action_real = real_data[motor_name]["action"][:, motor_idx]

        time_seq_sim_dict[motor_name] = sim_data[motor_name]["time"].tolist()
        time_seq_real_dict[motor_name] = real_data[motor_name]["time"].tolist()

        motor_pos_sim_dict[motor_name] = motor_pos_sim.tolist()
        motor_pos_real_dict[motor_name] = motor_pos_real.tolist()
        motor_vel_sim_dict[motor_name] = motor_vel_sim.tolist()
        motor_vel_real_dict[motor_name] = motor_vel_real.tolist()

        action_sim_dict[motor_name] = action_sim.tolist()
        action_real_dict[motor_name] = action_real.tolist()

        rmse_pos_dict[motor_name] = np.sqrt(
            np.mean((motor_pos_real - motor_pos_sim) ** 2)
        )
        rmse_vel_dict[motor_name] = np.sqrt(
            np.mean((motor_vel_real - motor_vel_sim) ** 2)
        )
        rmse_action_dict[motor_name] = np.sqrt(np.mean((action_real - action_sim) ** 2))

    # plot_sim2real_gap_line(
    #     time_seq_sim_dict[list(sim_data.keys())[-1]],
    #     time_seq_real_dict[list(sim_data.keys())[-1]],
    #     sim_data["imu"]["lin_vel"],
    #     real_data["imu"]["lin_vel"],
    #     save_path=exp_folder_path,
    #     title="Linear Velocity",
    #     y_label="Linear Velocities (m/s)",
    #     axis_names=["x", "y", "z"],
    #     file_name="lin_vel_gap",
    # )

    plot_sim2real_gap_line(
        time_seq_sim_dict[list(sim_data.keys())[-1]],
        time_seq_real_dict[list(sim_data.keys())[-1]],
        sim_data["imu"]["ang_vel"],
        real_data["imu"]["ang_vel"],
        save_path=exp_folder_path,
        title="Angular Velocity",
        y_label="Angular Velocities (rad/s)",
        file_name="ang_vel_gap",
    )

    plot_sim2real_gap_line(
        time_seq_sim_dict[list(sim_data.keys())[-1]],
        time_seq_real_dict[list(sim_data.keys())[-1]],
        sim_data["imu"]["euler"],
        real_data["imu"]["euler"],
        save_path=exp_folder_path,
    )

    # for rmse_dict, label in zip(
    #     [rmse_pos_dict, rmse_vel_dict, rmse_action_dict],
    #     ["motor_pos", "motor_vel", "action"],
    # ):
    #     plot_sim2real_gap_bar(
    #         rmse_dict,
    #         label,
    #         save_path=exp_folder_path,
    #         file_name=f"{label}_gap",
    #     )

    plot_joint_tracking(
        time_seq_sim_dict,
        time_seq_real_dict,
        motor_pos_sim_dict,
        motor_pos_real_dict,
        robot.joint_limits,
        save_path=exp_folder_path,
        file_name="sim2real_motor_pos",
        line_suffix=["_sim", "_real"],
    )

    plot_joint_tracking_frequency(
        time_seq_sim_dict,
        time_seq_real_dict,
        motor_pos_sim_dict,
        motor_pos_real_dict,
        save_path=exp_folder_path,
        file_name="sim2real_motor_freq",
        line_suffix=["_sim", "_real"],
    )

    plot_joint_tracking(
        time_seq_sim_dict,
        time_seq_real_dict,
        motor_vel_sim_dict,
        motor_vel_real_dict,
        robot.joint_limits,
        save_path=exp_folder_path,
        file_name="sim2real_motor_vel",
        set_ylim=False,
        line_suffix=["_sim", "_real"],
    )

    plot_joint_tracking(
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

    plot_joint_tracking(
        time_seq_sim_dict,
        time_seq_sim_dict,
        motor_pos_sim_dict,
        action_sim_dict,
        robot.joint_limits,
        save_path=exp_folder_path,
        file_name="sim_motor_pos_tracking",
    )

    plot_joint_tracking(
        time_seq_real_dict,
        time_seq_real_dict,
        motor_pos_real_dict,
        action_real_dict,
        robot.joint_limits,
        save_path=exp_folder_path,
        file_name="real_motor_pos_tracking",
    )


def main():
    parser = argparse.ArgumentParser(description="Run the SysID optimization.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="walk",
        help="The name of the task.",
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

    exp_name = f"{robot.name}_{args.policy}_sim2real_eval"
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
