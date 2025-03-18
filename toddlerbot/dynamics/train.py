import json
import os
import shutil

import hydra
import pytorch_lightning as pl
import torch
from dataset import TransitionDataset
from dynamics import DynamicsModel
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import wandb


def set_seed(seed):
    pl.seed_everything(seed)


def set_device(device):
    torch.backends.cudnn.enabled = True
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    torch.set_float32_matmul_precision("medium")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@hydra.main(config_name="config")
def main(cfg):
    set_seed(cfg.seed)
    set_device(cfg.device)

    output_dir = os.getcwd()
    print(f"OUTPUT_DIR: {output_dir}")

    datamodule = TransitionDataset(cfg.data)
    system = DynamicsModel(cfg)
    trainer: Trainer = hydra.utils.instantiate(cfg.train.trainer)

    if cfg.run_mode == "train":
        run_name = os.path.basename(output_dir)
        wandb.init(project="DynamicsEncoder", name=run_name)
        wandb_logger = WandbLogger()
        trainer.logger = wandb_logger

        # Train the model
        trainer.fit(system, datamodule=datamodule)

        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        print("Best checkpoint path:", best_ckpt_path)

        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(
                {
                    "run_name": cfg.data.run_name,
                    "best_ckpt_path": best_ckpt_path,
                    "observation_size": cfg.model.observation_size,
                    "action_size": cfg.model.action_size,
                    "hidden_layers": list(cfg.model.hidden_layers),
                    "num_envs": cfg.data.num_envs,
                    "horizon": cfg.data.horizon,
                    "history": cfg.data.history,
                },
                f,
            )

    else:
        if "encoder" in output_dir:
            best_ckpt_path = os.path.join(output_dir, "ckpt_decoded.pt")
        else:
            # Load best checkpoint from summary.json if available
            summary_path = os.path.join(output_dir, "summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    best_ckpt_path = json.load(f).get("best_ckpt_path", "")
            else:
                raise FileNotFoundError("No checkpoint found for evaluation.")

        print(f"Evaluating model from checkpoint: {best_ckpt_path}")
        system.load_state_dict(
            torch.load(best_ckpt_path, map_location=get_device())["state_dict"]
        )

    trainer.test(system, datamodule=datamodule)

    shutil.copy2(
        os.path.join(datamodule.data_root, "..", "dr_params.pkl"),
        os.path.join(output_dir, "dr_params.pkl"),
    )

    wandb.finish()


if __name__ == "__main__":
    main()
