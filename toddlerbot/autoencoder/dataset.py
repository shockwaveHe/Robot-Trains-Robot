import json
import os
import platform

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class ParamsDataset(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg["batch_size"]
        self.num_workers = cfg["num_workers"] if platform.system() != "Darwin" else 0

        self.data_root = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "results",
            f"dynamics_model_{cfg['time_str']}",
        )
        print(f"data root: {self.data_root}")

        # Check the root path
        # assert os.path.exists(self.data_root), f'{self.data_root} not exists'
        with open(os.path.join(self.data_root, "summary.json"), "r") as f:
            summary = json.load(f)

        ckpt = torch.load(
            os.path.join(self.data_root, summary["best_ckpt_path"]),
            map_location=get_device(),
        )
        params_all = []
        for i in range(summary["num_envs"]):
            params_list = []
            for j in range(len(summary["hidden_layers"]) + 1):
                layer_name = f"model.ensemble_mlp.{i}.layers.{2 * j}"
                params_list.append(ckpt["state_dict"][layer_name + ".weight"].flatten())
                params_list.append(ckpt["state_dict"][layer_name + ".bias"])

            params = torch.cat(params_list, dim=0)
            params_all.append(params)

        self.params = torch.stack(params_all)
        print("params shape:", self.params.shape)

        train_size = int(0.9 * len(self.params))
        val_size = int(0.09 * len(self.params))
        test_size = len(self.params) - train_size - val_size

        print("train_size:", train_size)
        print("val_size:", val_size)
        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self, [train_size, val_size, test_size]
        )

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return self.params[idx]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            # pin_memory=True,
            # persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
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
