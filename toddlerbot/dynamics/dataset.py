import io
import os
import platform
import random

import lz4.frame
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import BatchSampler, DataLoader


class StepEnvBatchSampler(BatchSampler):
    def __init__(
        self, num_steps, num_envs, step_batch_size, env_batch_size, shuffle=True
    ):
        # Compute the number of valid starting steps per environment.
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.step_batch_size = step_batch_size
        self.env_batch_size = env_batch_size
        self.shuffle = shuffle

        # Precompute batches as lists of (step_idx, env_idx) tuples.
        self.batches = self._create_batches()

    def _create_batches(self):
        batches = []
        for env_start in range(0, self.num_envs, self.env_batch_size):
            env_end = min(env_start + self.env_batch_size, self.num_envs)
            for step_start in range(
                0,
                self.num_steps // self.step_batch_size * self.step_batch_size,
                self.step_batch_size,
            ):
                batch = []
                for env_idx in range(env_start, env_end):
                    for step_idx in range(
                        step_start,
                        min(step_start + self.step_batch_size, self.num_steps),
                    ):
                        index = step_idx + env_idx * self.num_steps
                        batch.append(index)

                if batch:
                    batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        return batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        num_env_batches = self.num_envs // self.env_batch_size
        num_step_batches = self.num_steps // self.step_batch_size
        return num_env_batches * num_step_batches


class TransitionDataset(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.split = "full"
        self.step_batch_size = getattr(cfg, "step_batch_size")
        self.env_batch_size = getattr(cfg, "env_batch_size")
        self.num_test_envs = getattr(cfg, "num_test_envs")
        self.num_test_steps = getattr(cfg, "num_test_steps")
        self.num_workers = (
            getattr(cfg, "num_workers") if platform.system() != "Darwin" else 0
        )
        self.history = getattr(cfg, "history")
        self.horizon = getattr(cfg, "horizon")
        self.data_root = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "results",
            getattr(self.cfg, "run_name"),
            "transition_data",
        )
        print(f"data root: {self.data_root}")

        # # Read the compressed file
        # with open(os.path.join(self.data_root, "transition_data.pt.lz4"), "rb") as f:
        #     compressed_data = f.read()

        # # Decompress the data using LZ4
        # serialized_data = lz4.frame.decompress(compressed_data)

        # # Wrap the decompressed bytes in a BytesIO buffer
        # buffer = io.BytesIO(serialized_data)

        # # Load the object with torch.load
        # transition_data = torch.load(buffer)
        # self.obs = transition_data["obs"].to(torch.float32)
        # self.action = transition_data["action"].to(torch.float32)

        self.obs, self.action, self.rewards, self.dones = self._load_data()

        self.num_steps = self.action.shape[0] - self.horizon - self.history + 1
        self.num_envs = self.action.shape[1]

        print("obs shape:", self.obs.shape)
        print("action shape:", self.action.shape)

    def _load_data(self):
        obs_list, action_list, reward_list, done_list = [], [], [], []

        files = sorted(
            [
                os.path.join(self.data_root, f)
                for f in os.listdir(self.data_root)
                if f.endswith(".pt.lz4")
            ]
        )

        for file in files:
            with open(file, "rb") as f:
                compressed = f.read()
                serialized = lz4.frame.decompress(compressed)
                buffer = io.BytesIO(serialized)
                data_batch = torch.load(buffer)

                for entry in data_batch:
                    obs_list.append(entry["obs"])
                    action_list.append(entry["actions"])
                    reward_list.append(entry["rewards"])
                    done_list.append(entry["dones"])

        obs = torch.stack(obs_list).to(torch.float32)
        actions = torch.stack(action_list).to(torch.float32)
        rewards = torch.stack(reward_list).to(torch.float32)
        dones = torch.stack(done_list).to(torch.float32)

        return obs, actions, rewards, dones

    def __len__(self):
        if self.split == "train":
            return self.num_envs * len(self.train_data)
        elif self.split == "val":
            return self.num_envs * len(self.val_data)
        elif self.split == "test":
            return self.num_test_envs
        return self.num_steps * self.num_envs

    @property
    def setup(self):
        # Split environments
        step_indices = list(range(self.num_steps))
        random.shuffle(step_indices)
        train_size = int(0.9 * self.num_steps)
        self.train_data = step_indices[:train_size]
        self.val_data = step_indices[train_size:]
        print(f"Train data: {len(self.train_data)}, Val data: {len(self.val_data)}")

    def __getitem__(self, idx):
        if self.split == "train":
            step_idx = self.train_data[idx // self.num_envs]
            step_end = step_idx + self.horizon + self.history
            env_idx = idx % self.num_envs
        elif self.split == "val":
            step_idx = self.val_data[idx // self.num_envs]
            step_end = step_idx + self.horizon + self.history
            env_idx = idx % self.num_envs
        else:
            step_idx = idx // self.num_test_envs
            step_end = step_idx + self.num_test_steps
            env_idx = idx % self.num_test_envs

        obs = self.obs[step_idx:step_end, env_idx]
        action = self.action[step_idx:step_end, env_idx]
        return obs, action

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        self.split = "train"
        sampler = StepEnvBatchSampler(
            num_steps=len(self.train_data),
            num_envs=self.num_envs,
            step_batch_size=self.step_batch_size,
            env_batch_size=self.env_batch_size,
            shuffle=True,
        )
        return DataLoader(
            self,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        self.split = "val"
        sampler = StepEnvBatchSampler(
            num_steps=len(self.val_data),
            num_envs=self.num_envs,
            step_batch_size=self.step_batch_size,
            env_batch_size=self.env_batch_size,
            shuffle=True,
        )
        return DataLoader(
            self,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        self.split = "test"
        return DataLoader(
            self,
            batch_size=self.num_test_envs,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
