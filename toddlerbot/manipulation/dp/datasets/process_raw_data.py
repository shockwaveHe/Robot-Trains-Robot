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

import cv2
import joblib
import numpy as np

# new format is: raw_data.keys():
# dict_keys(['time', 'arm_motor_pos', 'fsr_data', 'image', 'start_time'])
# time (83,)
# arm_motor_pos (83, 30)
# fsr_data (83, 2)
# image (83, 480, 640, 3)
# start_time (1,)


def load_raw_dataset():
    result_dir = "/Users/weizhuo2/Documents/gits/toddleroid/results/"
    # dataset_path = "toddlerbot_arms_teleop_fixed_mujoco_20240909_204445/dataset.lz4"
    dataset_path = (
        "toddlerbot_OP3_teleop_follower_pd_real_world_20241014_223229/dataset.lz4"
    )
    raw_data = joblib.load(result_dir + dataset_path)

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
    raw_data_converted["episode_ends"] = raw_data["episode_ends"]  # to be added
    raw_data_converted["start_time"] = raw_data["start_time"]

    print(raw_data.keys())
    print(raw_data_converted.keys())
    return raw_data_converted


def main():
    output_dataset = dict()
    raw_data = load_raw_dataset()

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

    # convert images to 171x96 resolution
    # resized_images = [cv2.resize(image, (171, 96)) for image in raw_data["images"]]
    # output_dataset["images"] = np.array(resized_images, dtype=np.float32)[
    #     :, :96, 38:134
    # ]
    resized_images = [cv2.resize(image, (128, 96)) for image in raw_data["images"]]
    output_dataset["images"] = np.array(resized_images, dtype=np.float32)[
        :, :96, 16:112
    ]

    # assign state and action
    output_dataset["agent_pos"] = raw_data["state_array"][:, 1:17].astype(np.float32)
    output_dataset["action"] = []
    last_idx = 0
    offset = 2
    for idx in raw_data["episode_ends"]:
        shifted_state = raw_data["state_array"][last_idx + offset : idx, 1:17]
        repeated_last_rows = np.tile(
            shifted_state[-1], (offset, 1)
        )  # Create n copies of the last row
        shifted_state = np.vstack([shifted_state, repeated_last_rows])
        output_dataset["action"].append(shifted_state)
        last_idx = idx
    output_dataset["action"] = np.vstack(output_dataset["action"]).astype(np.float32)
    # output_dataset['action'] = raw_data['state_array'][:, 1:17]
    output_dataset["episode_ends"] = raw_data["episode_ends"]

    # save the dataset
    output_path = (
        "/Users/weizhuo2/Documents/gits/toddleroid/datasets/teleop_dataset_neo.lz4"
    )
    joblib.dump(output_dataset, output_path)


if __name__ == "__main__":
    main()
