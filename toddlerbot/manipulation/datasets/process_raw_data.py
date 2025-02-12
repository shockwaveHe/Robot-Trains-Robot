import argparse
import os
from typing import List

import cv2
import joblib
import numpy as np
from tqdm import tqdm

from toddlerbot.manipulation.utils.dataset_utils import create_video_grid
from toddlerbot.sim.robot import Robot
from toddlerbot.visualization.vis_plot import plot_teleop_dataset

# This script processes raw data to create a dataset to train the diffusion policy.


def process_raw_dataset(
    robot: Robot, task: str, time_str: str, dt: float = 0.1, time_offset: float = 0.2
):
    """Processes raw dataset files for a specified robot and task, resampling and organizing data for further analysis.

    Args:
        robot (Robot): The robot object containing configuration and state information.
        task (str): The task type, which determines how actions and positions are processed.
        time_str (str): A string representing the time, used for file path generation.
        dt (float, optional): The time interval for resampling the data. Defaults to 0.1.
        time_offset (float, optional): The time offset applied during resampling. Defaults to 0.2.
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
    images_list = []
    action_list = []
    agent_pos_list = []
    for idx in tqdm(range(len(files)), desc="Loading raw data"):
        raw_data = joblib.load(os.path.join(dataset_path, files[idx]))

        time_list.append(raw_data["time"] - raw_data["start_time"])

        resized_images = np.array(
            [cv2.resize(img, (128, 96))[:, 16:112] for img in raw_data["image"]],
            dtype=np.float32,
        )
        images_list.append(resized_images)

        if task == "pick":
            action = np.concatenate(
                [
                    raw_data["action"][:, 23:30],
                    raw_data["action"][:, 31:],
                ],
                axis=1,
            )
            agent_pos = np.concatenate(
                [
                    raw_data["motor_pos"][:, 23:30],
                    raw_data["motor_pos"][:, 31:],
                ],
                axis=1,
            )
        else:
            action = np.concatenate(
                [
                    raw_data["action"][:, 16:30],
                    raw_data["action"][:, 30:]
                    if robot.has_gripper
                    else np.zeros((raw_data["action"].shape[0], 2)),
                ],
                axis=1,
            )
            agent_pos = np.concatenate(
                [
                    raw_data["motor_pos"][:, 16:30],
                    raw_data["motor_pos"][:, 30:]
                    if robot.has_gripper
                    else np.zeros((raw_data["motor_pos"].shape[0], 2)),
                ],
                axis=1,
            )
        agent_pos_list.append(agent_pos)

        action_list.append(action)
        agent_pos_list.append(agent_pos)

    # Resample each episode
    # shift_index = int(time_offset / dt)
    resampled_time: List[np.ndarray] = []
    resampled_images = []
    resampled_action = []
    resampled_pos = []
    for ep_time, ep_img, ep_act, ep_pos in zip(
        time_list, images_list, action_list, agent_pos_list
    ):
        # Uniform time vector
        uniform_t = np.arange(ep_time[0], ep_time[-1], dt)

        # Nearest neighbor for images:
        idx = np.searchsorted(ep_time, uniform_t, side="left")
        idx = np.clip(idx, 0, len(ep_time) - 1)
        selected_imgs = ep_img[idx]

        # Linear interpolation for agent_pos:
        interp_pos = np.zeros((len(uniform_t), ep_pos.shape[1]), dtype=np.float32)
        for dim in range(ep_pos.shape[1]):
            interp_pos[:, dim] = np.interp(uniform_t, ep_time, ep_pos[:, dim])

        interp_action = np.zeros((len(uniform_t), ep_act.shape[1]), dtype=np.float32)
        for dim in range(ep_act.shape[1]):
            interp_action[:, dim] = np.interp(uniform_t, ep_time, ep_act[:, dim])

        # shifted_state = np.concatenate(
        #     [
        #         interp_pos[shift_index:],
        #         np.repeat(interp_pos[-1][None, :], shift_index, axis=0),
        #     ]
        # )

        if len(resampled_time) == 0:
            resampled_time.append(uniform_t)
        else:
            resampled_time.append(uniform_t + resampled_time[-1][-1])

        resampled_images.append(selected_imgs)
        resampled_action.append(interp_action)
        resampled_pos.append(interp_pos)

    # Concatenate resampled results
    final_time = np.concatenate(resampled_time)
    final_images = np.concatenate(resampled_images, axis=0)
    final_agent_pos = np.concatenate(resampled_pos, axis=0)
    final_action = np.concatenate(resampled_action, axis=0)
    final_episode_ends = np.cumsum([x.shape[0] for x in resampled_pos])

    final_dataset = {
        "time": final_time,
        "images": final_images,
        "agent_pos": final_agent_pos,
        "action": final_action,
        "episode_ends": final_episode_ends,
    }

    output_path = os.path.join("datasets", f"{task}_dataset_{time_str}")
    os.makedirs(output_path, exist_ok=True)
    print(output_path)

    create_video_grid(
        final_images.transpose(0, 3, 1, 2),
        final_episode_ends,
        output_path,
        "image_data.mp4",
    )
    plot_teleop_dataset(
        final_agent_pos,
        final_episode_ends,
        save_path=output_path,
        file_name="motor_pos_data",
        file_suffix=time_str,
    )
    plot_teleop_dataset(
        final_action,
        final_episode_ends,
        save_path=output_path,
        file_name="action_data",
        file_suffix=time_str,
    )

    # save the dataset
    joblib.dump(final_dataset, os.path.join(output_path, "dataset.lz4"))


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
