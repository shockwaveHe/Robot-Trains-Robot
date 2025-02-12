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


class TeleopImageDataset(torch.utils.data.Dataset):
    """Dataset class for teleoperation data with images."""

    def __init__(
        self,
        dataset_path_list: List[str],
        exp_folder_path: str,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
    ):
        """Initializes the data processing pipeline for a teleoperation dataset.

        This constructor loads multiple datasets, processes and normalizes the data, and prepares it for model training or evaluation. It handles image, agent position, and action data, creating video visualizations and plots for analysis. The method also computes sample indices for state-action sequences, considering prediction, observation, and action horizons.

        Args:
            dataset_path_list (List[str]): List of file paths to the datasets to be loaded.
            exp_folder_path (str): Path to the folder where experiment outputs, such as videos and plots, will be saved.
            pred_horizon (int): The prediction horizon length for the model.
            obs_horizon (int): The observation horizon length for the model.
            action_horizon (int): The action horizon length for the model.
        """
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
        """Returns the number of elements in the collection.

        This method provides the length of the collection by returning the count of indices stored.

        Returns:
            int: The number of elements in the collection.
        """
        return len(self.indices)

    def __getitem__(self, idx):
        """Retrieves a normalized data sample for a given index.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            dict: A dictionary containing the normalized data sample with keys 'image' and 'agent_pos', each truncated to the observation horizon.
        """
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
