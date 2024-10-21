"""
This script processes the raw data collected in teleop to create a dataset for dp training.
Raw dataset entry:
state_array: [time(1), motor_angles(14), fsrL(1), fsrR(1), camera_frame_idx(1)]

It creates an output dataset of the form:
image: (N, 96, 96,3) - RGB
episode_ends: (N,)
state: (N, ns)
action: (N, na)
"""

import argparse
import os

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# new format is: raw_data.keys():
# dict_keys(['time', 'arm_motor_pos', 'fsr_data', 'image', 'start_time'])
# time (83,)
# arm_motor_pos (83, 30)
# fsr_data (83, 2)
# image (83, 480, 640, 3)
# start_time (1,)


def load_single_seq(dataset_path: str):
    raw_data = joblib.load(dataset_path)

    # convert the format of data class to dict
    raw_data_converted = dict()
    raw_data_converted["state_array"] = np.hstack(
        [
            raw_data["time"].reshape(-1, 1),
            raw_data["arm_motor_pos"][:, 16:30],
            raw_data["fsr_data"],
            np.arange(len(raw_data["time"])).reshape(-1, 1),
        ]
    )
    raw_data_converted["images"] = raw_data["image"]
    # raw_data_converted["episode_ends"] = raw_data["episode_ends"]
    raw_data_converted["episode_ends"] = [raw_data_converted["state_array"].shape[0]]
    raw_data_converted["start_time"] = raw_data["start_time"]

    print(raw_data.keys())
    print(raw_data_converted.keys())
    return raw_data_converted


def load_raw_dataset(dataset_path: str, debug=True):
    # find all files in the path named "toddlerbot_x.lz4"
    files = os.listdir(dataset_path)
    files = [f for f in files if f.startswith("toddlerbot")]
    files.sort()

    raw_data = []
    print("Loading raw data...")
    for idx in tqdm(range(len(files))):
        raw_data.append(
            load_single_seq(os.path.join(dataset_path, f"toddlerbot_{idx}.lz4"))
        )

    # Patch in all the sequences
    combined_dataset = dict()
    combined_dataset["state_array"] = np.vstack([x["state_array"] for x in raw_data])
    combined_dataset["images"] = np.vstack([x["images"] for x in raw_data])
    combined_dataset["episode_ends"] = np.cumsum(
        [x["state_array"].shape[0] for x in raw_data]
    )
    combined_dataset["start_time"] = raw_data[0]["start_time"]

    if debug:
        print("Saving a gif of the dataset...")
        from PIL import Image

        nframes = min(5000, combined_dataset["images"].shape[0])
        height, width = combined_dataset["images"].shape[1:3]
        new_size = (width // 2, height // 2)

        frames = combined_dataset["images"][:nframes][::20]
        blank_frames = np.zeros((20, height, width, 3), dtype=np.uint8)
        frames = np.concatenate([frames, blank_frames], axis=0)

        frames = [
            Image.fromarray(frame.astype("uint8")).resize(
                new_size, Image.Resampling.LANCZOS
            )
            for frame in frames
        ]
        frames[0].save(
            "/home/weizhuo2/test.gif",
            save_all=True,
            append_images=frames[1:],
            fps=30,
            loop=0,
            optimize=True,
            quality=30,
        )
    return combined_dataset


def get_idle_idx(seq, threshold=0.2):
    """
    Get the index of the last idle state in the sequence.
    """
    idle_idx = 0
    maxdiff = 0
    for idx, state in enumerate(seq):
        diff = np.abs(state - seq[0])
        maxdiff = max(maxdiff, np.max(diff))
        if np.any(diff > threshold):
            idle_idx = idx
            break
    return idle_idx


"""
Output dataset entry:
images: (N, 96, 96, 3)
episode_ends: (N,)
agent_pos: (N, ns)
action: (N, na)
"""


def main(dataset_path: str, output_path: str):
    raw_data = load_raw_dataset(dataset_path)

    # convert from 30hz to 10hz
    low_freq_state_array = np.array([])
    low_freq_epi_ends = []
    last_epi_end_idx = 0
    for epi_end_idx in raw_data["episode_ends"]:
        epi_state_array = raw_data["state_array"][last_epi_end_idx:epi_end_idx]
        epi_state_array = epi_state_array[::3]
        low_freq_state_array = (
            np.vstack([low_freq_state_array, epi_state_array])
            if low_freq_state_array.size
            else epi_state_array
        )
        low_freq_epi_ends.append(low_freq_state_array.shape[0])
        last_epi_end_idx = epi_end_idx

    raw_data["state_array"] = low_freq_state_array
    raw_data["episode_ends"] = low_freq_epi_ends
    raw_data["images"] = raw_data["images"][low_freq_state_array[:, -1].astype(int)]

    output_dataset = {}
    # convert images to 171x96 resolution
    # resized_images = [cv2.resize(image, (171, 96)) for image in raw_data["images"]]
    # output_dataset["images"] = np.array(resized_images, dtype=np.float32)[
    #     :, :96, 38:134
    # ]
    resized_images = [cv2.resize(image, (128, 96)) for image in raw_data["images"]]
    output_dataset["images"] = np.array(resized_images, dtype=np.float32)[
        :, :96, 16:112
    ]

    # import matplotlib.pyplot as plt

    # plt.imshow(resized_images[0])
    # plt.show()

    # assign state and action
    output_dataset["agent_pos"] = raw_data["state_array"][:, 1:17].astype(np.float32)
    action_list = []
    last_idx = 0
    offset = 2
    for idx in raw_data["episode_ends"]:
        shifted_state = raw_data["state_array"][last_idx + offset : idx, 1:17]
        # Create n copies of the last row
        repeated_last_rows = np.tile(shifted_state[-1], (offset, 1))
        shifted_state = np.vstack([shifted_state, repeated_last_rows])
        action_list.append(shifted_state)
        last_idx = idx

    output_dataset["episode_ends"] = raw_data["episode_ends"]
    output_dataset["action"] = action_list
    output_dataset["action"] = np.vstack(output_dataset["action"]).astype(np.float32)

    # remove the idle time in each sequence
    last_idx = 0
    temp_dict = {"agent_pos": [], "action": [], "images": [], "episode_ends": []}
    for idx in output_dataset["episode_ends"]:
        seq_agent_pos = output_dataset["agent_pos"][last_idx:idx]
        seq_action = output_dataset["action"][last_idx:idx]
        seq_image = output_dataset["images"][last_idx:idx]
        idle_idx = get_idle_idx(seq_agent_pos)

        # reassign them to the output dataset
        seq_agent_pos = seq_agent_pos[idle_idx:]
        seq_action = seq_action[idle_idx:]
        seq_image = seq_image[idle_idx:]

        temp_dict["agent_pos"].append(seq_agent_pos)
        temp_dict["action"].append(seq_action)
        temp_dict["images"].append(seq_image)
        if len(temp_dict["episode_ends"]) == 0:
            temp_dict["episode_ends"].append(seq_agent_pos.shape[0])
        else:
            temp_dict["episode_ends"].append(
                temp_dict["episode_ends"][-1] + seq_agent_pos.shape[0]
            )

        last_idx = idx

    temp_dict["images"] = np.vstack(temp_dict["images"])
    temp_dict["agent_pos"] = np.vstack(temp_dict["agent_pos"])
    temp_dict["action"] = np.vstack(temp_dict["action"])

    output_dataset = temp_dict

    joblib.dump(output_dataset, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw data to create dataset.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot_OP3",
        help="The name of the robot. Need to match the name in descriptions.",
        choices=["toddlerbot_OP3", "toddlerbot_arms"],
    )
    parser.add_argument(
        "--time-str",
        type=str,
        default="",
        help="The time str of the dataset.",
    )
    args = parser.parse_args()

    dataset_path = os.path.join(
        "results",
        f"{args.robot}_teleop_follower_pd_real_world_{args.time_str}",
        # "toddlerbot_0.lz4",
    )
    # save the dataset
    output_path = os.path.join("datasets", f"teleop_dataset_{args.time_str}.lz4")
    print(output_path)

    main(dataset_path, output_path)
