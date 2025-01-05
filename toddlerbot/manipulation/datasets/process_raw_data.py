import argparse
import os
from typing import List

import cv2
import joblib
import numpy as np
from tqdm import tqdm

from toddlerbot.manipulation.utils.dataset_utils import create_video_grid
from toddlerbot.sim.robot import Robot


def process_raw_dataset(
    robot: Robot, task: str, time_str: str, dt: float = 0.1, time_offset: float = 0.2
):
    motor_limits = np.array([robot.joint_limits[name] for name in robot.motor_ordering])
    # find all files in the path named "toddlerbot_x.lz4"
    dataset_path = os.path.join(
        "results", f"{args.robot}_teleop_follower_pd_real_world_{time_str}"
    )

    files = os.listdir(dataset_path)
    files = [f for f in files if f.startswith("toddlerbot")]
    files.sort()

    time_list = []
    images_list = []
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
            agent_pos = np.concatenate(
                [
                    raw_data["motor_pos"][:, 23:30],
                    raw_data["fsr_data"][:, -1:] / 100 * motor_limits[-1:, 1],
                ],
                axis=1,
            )
        else:
            agent_pos = np.concatenate(
                [
                    raw_data["motor_pos"][:, 16:30],
                    raw_data["fsr_data"] / 100 * motor_limits[-2:, 1],
                ],
                axis=1,
            )
        agent_pos_list.append(agent_pos)

    # Resample each episode
    shift_index = int(time_offset / dt)
    resampled_time: List[np.ndarray] = []
    resampled_images = []
    resampled_pos = []
    resampled_action = []
    for ep_time, ep_img, ep_pos in zip(time_list, images_list, agent_pos_list):
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

        shifted_state = np.concatenate(
            [
                interp_pos[shift_index:],
                np.repeat(interp_pos[-1][None, :], shift_index, axis=0),
            ]
        )

        if len(resampled_time) == 0:
            resampled_time.append(uniform_t)
        else:
            resampled_time.append(uniform_t + resampled_time[-1][-1])

        resampled_images.append(selected_imgs)
        resampled_pos.append(interp_pos)
        resampled_action.append(shifted_state)

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

    create_video_grid(
        final_images.transpose(0, 3, 1, 2),
        final_episode_ends,
        "datasets",
        f"teleop_dataset_{time_str}.mp4",
    )

    # save the dataset
    output_path = os.path.join("datasets", f"teleop_dataset_{time_str}.lz4")
    print(output_path)

    joblib.dump(final_dataset, output_path)


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
