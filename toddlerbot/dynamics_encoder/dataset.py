import platform

import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split

from toddlerbot.autoencoder.dataset import ParamsDataset
from toddlerbot.dynamics.dataset import TransitionDataset


class CombinedDataset(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Extract dynamics-related fields
        self.dynamics_cfg = OmegaConf.create(
            {
                "num_envs": cfg.num_envs,
                "num_test_envs": cfg.num_test_envs,
                "num_test_steps": cfg.num_test_steps,
                "step_batch_size": 1,
                "env_batch_size": cfg.batch_size,
                "num_workers": cfg.num_workers if platform.system() != "Darwin" else 0,
                "horizon": cfg.horizon,
                "history": cfg.history,
                "time_str": cfg.transition_time_str,
            }
        )

        # Extract autoencoder-related fields
        self.autoencoder_cfg = OmegaConf.create(
            {
                "batch_size": cfg.batch_size,
                "num_workers": cfg.num_workers if platform.system() != "Darwin" else 0,
                "time_str": cfg.params_time_str,
            }
        )

        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers if platform.system() != "Darwin" else 0

    def __len__(self):
        return len(self.params_dataset)

    def __getitem__(self, idx):
        return self.params_dataset[idx], *self.transition_dataset[idx]

    @property
    def setup(self):
        # First dataset
        self.params_dataset = ParamsDataset(self.autoencoder_cfg)
        # Second dataset (transitions)
        self.transition_dataset = TransitionDataset(self.dynamics_cfg)
        self.transition_dataset.split = "test"
        assert len(self.params_dataset) == len(self.transition_dataset)
        # Combined dataset
        train_size = int(0.9 * len(self))
        val_size = int(0.09 * len(self))
        test_size = len(self) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
