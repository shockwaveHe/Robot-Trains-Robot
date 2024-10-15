import numpy as np
import torch
import joblib
from diffusion_policy_minimal.utils.dataset_utils import create_sample_indices, get_data_stats, normalize_data, sample_sequence

# episode_ends idx is the index of the next start. In other words, you can use 
# train_data[:episode_ends[idx]], and train_data[episode_ends[idx]:episode_ends[idx+1]]
class TeleopImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: str,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
    ):
        # read from zarr dataset
        dataset_root = joblib.load(dataset_path)
        # dataset_root['state_array'] = dataset_root['state_array'].astype(np.float32)
        # dataset_root['images'] = dataset_root['images'].astype(np.float32)

        # float32, [0,1], (N,96,96,3)
        train_image_data = dataset_root["images"]
        train_image_data = np.moveaxis(train_image_data, -1, 1)
        # (N,3,96,96)

        # (N, D)
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            "agent_pos": dataset_root["agent_pos"],
            "action": dataset_root["action"],
        }
        episode_ends = dataset_root["episode_ends"]
        # episode_ends = np.array([len(dataset_root["state_array"])])

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
        for key, data in train_data.items():
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

