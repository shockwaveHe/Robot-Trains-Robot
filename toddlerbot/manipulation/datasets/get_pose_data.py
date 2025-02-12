import argparse
import os

import joblib
import numpy as np
from tqdm import tqdm

from toddlerbot.sim.robot import Robot
from toddlerbot.visualization.vis_plot import plot_teleop_dataset

# This script processes raw data to create a arm pose dataset to perturb the RL training.


def process_raw_dataset(
    robot: Robot, task: str, time_str: str, dt: float = 0.1, time_offset: float = 0.2
):
    """Processes raw dataset files for a specified robot and task, extracting and saving relevant data.

    Args:
        robot (Robot): The robot instance containing joint limits and motor ordering.
        task (str): The task type, either "pick" or another task, determining data extraction logic.
        time_str (str): A string representing the time, used to locate the dataset directory.
        dt (float, optional): The time step for data processing. Defaults to 0.1.
        time_offset (float, optional): The time offset applied to the data. Defaults to 0.2.

    Processes files in the specified dataset directory, extracting time, action, and motor position data.
    Concatenates and saves the processed data into a new dataset file, and generates plots for visualization.
    """
    # motor_limits = np.array([robot.joint_limits[name] for name in robot.motor_ordering])
    # find all files in the path named "toddlerbot_x.lz4"
    dataset_path = os.path.join(
        "results", f"{args.robot}_teleop_follower_pd_real_world_{time_str}"
    )

    files = os.listdir(dataset_path)
    files = [f for f in files if f.startswith("toddlerbot")]
    files.sort()

    time_list = []
    action_list = []
    agent_pos_list = []
    for idx in tqdm(range(len(files)), desc="Loading raw data"):
        raw_data = joblib.load(os.path.join(dataset_path, files[idx]))

        time_list.append(raw_data["time"][-1] - raw_data["start_time"])

        if task == "pick":
            action = np.concatenate(
                [
                    raw_data["action"][-1:, 23:30],
                    raw_data["action"][-1:, 31:],
                ],
                axis=1,
            )
            agent_pos = np.concatenate(
                [
                    raw_data["motor_pos"][-1:, 23:30],
                    raw_data["motor_pos"][-1:, 31:],
                ],
                axis=1,
            )
        else:
            action = np.concatenate(
                [
                    raw_data["action"][-1:, 16:30],
                    raw_data["action"][-1:, 30:]
                    if robot.has_gripper
                    else np.zeros((1, 2)),
                ],
                axis=1,
            )
            agent_pos = np.concatenate(
                [
                    raw_data["motor_pos"][-1:, 16:30],
                    raw_data["motor_pos"][-1:, 30:]
                    if robot.has_gripper
                    else np.zeros((1, 2)),
                ],
                axis=1,
            )

        action_list.append(action)
        agent_pos_list.append(agent_pos)

    # Concatenate resampled results
    final_time = np.array(time_list)
    final_agent_pos = np.concatenate(agent_pos_list, axis=0)
    final_action = np.concatenate(action_list, axis=0)

    final_dataset = {
        "time": final_time,
        "motor_pos": final_agent_pos,
        "action": final_action,
    }

    plot_teleop_dataset(
        final_agent_pos,
        np.array([final_agent_pos.shape[0]]),
        save_path="motion",
        file_name="arm_pose",
        file_suffix="dataset",
    )

    # save the dataset
    joblib.dump(final_dataset, os.path.join("motion", "arm_pose_dataset.lz4"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw data to create dataset.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in descriptions.",
        choices=["toddlerbot", "toddlerbot_gripper"],
    )
    parser.add_argument(
        "--task",
        type=str,
        default="hug",
        help="The task.",
    )
    parser.add_argument(
        "--time-str",
        type=str,
        default="20241210_231952",
        help="The time str of the dataset.",
    )
    args = parser.parse_args()

    robot = Robot(args.robot)

    process_raw_dataset(robot, args.task, args.time_str)
