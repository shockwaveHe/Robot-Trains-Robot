from typing import List

import joblib
import numpy as np
import numpy.typing as npt
import torch

from toddlerbot.manipulation.utils.dataset_utils import (
    create_sample_indices,
    create_video_grid,
    get_data_stats,
    normalize_data,
    sample_sequence,
)
from toddlerbot.visualization.vis_plot import plot_teleop_dataset


# episode_ends idx is the index of the next start. In other words, you can use
# train_data[:episode_ends[idx]], and train_data[episode_ends[idx]:episode_ends[idx+1]]
class TeleopImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path_list: List[str],
        exp_folder_path: str,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
    ):
        # read from zarr dataset
        train_image_list = []
        train_agent_pos_list = []
        train_action_list = []
        episode_ends_list: List[npt.NDArray[np.float32]] = []
        for dataset_path in dataset_path_list:
            dataset_root = joblib.load(dataset_path)
            train_image_list.append(np.moveaxis(dataset_root["images"], -1, 1))
            train_agent_pos_list.append(dataset_root["agent_pos"])
            train_action_list.append(dataset_root["action"])
            if len(episode_ends_list) > 0:
                episode_ends_list.append(
                    episode_ends_list[-1][-1] + dataset_root["episode_ends"]
                )
            else:
                episode_ends_list.append(dataset_root["episode_ends"])

        # concatenate all the data
        train_image_data = np.concatenate(train_image_list, axis=0)
        train_agent_pos = np.concatenate(train_agent_pos_list, axis=0)
        train_action = np.concatenate(train_action_list, axis=0)
        episode_ends = np.concatenate(episode_ends_list, axis=0)

        create_video_grid(
            train_image_data, episode_ends, exp_folder_path, "image_data.mp4"
        )
        plot_teleop_dataset(
            train_agent_pos,
            episode_ends,
            save_path=exp_folder_path,
            file_name="motor_pos_data",
        )
        plot_teleop_dataset(
            train_action,
            episode_ends,
            save_path=exp_folder_path,
            file_name="action_data",
        )

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in zip(["agent_pos", "action"], [train_agent_pos, train_action]):
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        # images are already normalized
        normalized_train_data["image"] = train_image_data

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            self.indices[idx]
        )

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        # discard unused observations
        nsample["image"] = nsample["image"][: self.obs_horizon, :]
        nsample["agent_pos"] = nsample["agent_pos"][: self.obs_horizon, :]
        return nsample
