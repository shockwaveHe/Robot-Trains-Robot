import json
import os
import pickle
import platform
import pytorch_lightning as pl
import torch
from abc import ABC, abstractmethod
from brax.io.torch import jax_to_torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split, Subset


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

class BaseDataset(pl.LightningDataModule, ABC):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg["batch_size"]
        self.num_workers = cfg["num_workers"] if platform.system() != "Darwin" else 0

        self.train_size = int(cfg["num_train_envs"])
        self.val_size = int(cfg["num_eval_envs"])
        self.get_params()

        self.train_dataset = Subset(self, range(self.train_size))
        self.val_dataset = Subset(self, range(self.train_size, self.train_size + self.val_size))
        self.test_dataset = Subset(self, range(self.train_size + self.val_size, len(self.params)))

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return self.params[idx]

    @abstractmethod
    def get_params(self):
        pass

    @property
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            # pin_memory=True,
            # persistent_workers=self.num_workers > 0,
        )

    @property
    def eval_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_size, # sample all the val data
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            # pin_memory=True,
            # persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            # pin_memory=True,
            # persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self):
        return DataLoader(
            self,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            # pin_memory=True,
            # persistent_workers=self.num_workers > 0,
        )
    def normalize_params(self):
        self.params_mean = self.params.mean(dim=0)
        self.params_std = self.params.std(dim=0)
        self.params = (self.params - self.params_mean) / self.params_std
        print("Normalized params")


class HyperparameterDataset(BaseDataset):
    def get_params(self):
        with open(os.path.join("results", self.cfg["time_str"], "dr_params.pkl"), "rb") as f:
            dr_params = pickle.load(f)

        sys_dict = dr_params[0]
        self.params = self.parse_parameters(sys_dict)
        print("params shape:", self.params.shape)
        self.test_size = len(self.params) - self.train_size - self.val_size
        print("train_size:", self.train_size, "val_size:", self.val_size, "test_size:", self.test_size)	

    def parse_parameters(self, sys_dict):
        parameters = []
        for key, value in sys_dict.items():
            value = jax_to_torch(value)
            if key == 'geom_friction':
                assert torch.all(torch.all(value == value[:, :1, :], axis=(0, 2))).item()
                parameters.append(value[:, 0, :])
            elif value.ndim == 3:
                parameters.append(value.flatten(start_dim=1))
            elif value.ndim == 2:
                parameters.append(value)
            else:
                raise ValueError(f"Unsupported parameter shape: {value.shape}")
        parameters = torch.cat(parameters, dim=1)
        parameters = parameters[:, torch.where(parameters.std(axis=0) > 0)[0]]
        return parameters