import os

import gdown
import numpy as np
import torch
from envs.pusht_env import PushTImageEnv

from datasets.pusht_dataset import PushTImageDataset

if __name__ == "__main__":
    ### **Env Demo**
    # Standard Gym Env (0.21.0 API)
    # 0. create env object
    env = PushTImageEnv()

    # 1. seed env for initial state.
    # Seed 0-200 are used for the demonstration dataset.
    env.seed(1000)

    # 2. must reset before use
    obs, info = env.reset()

    # 3. 2D positional action space [0,512]
    action = env.action_space.sample()

    # 4. Standard gym step method
    obs, reward, terminated, truncated, info = env.step(action)

    # prints and explains each dimension of the observation and action vectors
    with np.printoptions(precision=4, suppress=True, threshold=5):
        print("obs['image'].shape:", obs["image"].shape, "float32, [0,1]")
        print("obs['agent_pos'].shape:", obs["agent_pos"].shape, "float32, [0,512]")
        print("action.shape: ", action.shape, "float32, [0,512]")

    # ### **Dataset Demo**

    # download demonstration data from Google Drive
    dataset_path = "pusht_cchi_v7_replay.zarr.zip"
    if not os.path.isfile(dataset_path):
        id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
        gdown.download(id=id, output=dataset_path, quiet=False)

    # parameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    # |o|o|                             observations: 2
    # | |a|a|a|a|a|a|a|a|               actions executed: 8
    # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    # create dataset from file
    dataset = PushTImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
    )
    # save training data statistics (min, max) for each dim
    stats = dataset.stats

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True,
    )

    # visualize data in batch
    batch = next(iter(dataloader))
    print("batch['image'].shape:", batch["image"].shape)
    print("batch['agent_pos'].shape:", batch["agent_pos"].shape)
    print("batch['action'].shape", batch["action"].shape)
