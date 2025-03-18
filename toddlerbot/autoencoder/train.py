import json
import os
import shutil

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from dataset import ParamsDataset
from encoder import Encoder
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from sklearn.decomposition import PCA

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


def load_and_visualize_embeddings(output_dir):
    plt.switch_backend("Agg")

    # Load embeddings
    embeddings = np.load(os.path.join(output_dir, "embeddings.npy"))
    print(f"Loaded embeddings with shape: {embeddings.shape}")

    # Split into layers
    feature_dim = embeddings.shape[1] // 2
    layer0 = embeddings[:, :feature_dim]
    layer1 = embeddings[:, feature_dim:]

    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    layer0_pca = pca.fit_transform(layer0)
    print(f"Layer 0 PCA shape: {layer0_pca.shape}")  # Should be (1024, 2)
    layer1_pca = pca.fit_transform(layer1)
    print(f"Layer 1 PCA shape: {layer1_pca.shape}")  # Should be (1024, 2)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Layer 0 (first layer)
    ax1.scatter(
        layer0_pca[:, 0], layer0_pca[:, 1], alpha=0.5, c="blue", label="Layer 0"
    )
    ax1.set_title("PCA of First Layer Embeddings")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.legend()
    ax1.grid(True)

    # Plot Layer 1 (second layer)
    ax2.scatter(
        layer1_pca[:, 0], layer1_pca[:, 1], alpha=0.5, c="orange", label="Layer 1"
    )
    ax2.set_title("PCA of Second Layer Embeddings")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.legend()
    ax2.grid(True)

    # Adjust layout and show
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "embeddings_pca.png"))


@hydra.main(config_name="config")
def main(cfg):
    set_seed(cfg.seed)
    set_device(cfg.device)

    output_dir = os.getcwd()
    print(f"OUTPUT_DIR: {output_dir}")

    datamodule = ParamsDataset(cfg.data)
    system = Encoder(cfg)
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
                    "time_str": cfg.data.time_str,
                    "best_ckpt_path": best_ckpt_path,
                    "is_vae": cfg.model.is_vae,
                },
                f,
            )

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

    pred_list = trainer.predict(system, dataloaders=datamodule.predict_dataloader())
    embeddings = torch.cat([pred[0] for pred in pred_list], dim=0).flatten(start_dim=1)
    outputs = torch.cat([pred[1] for pred in pred_list], dim=0)
    print(f"prediction loss: {torch.stack([pred[2] for pred in pred_list])}")

    with open(os.path.join(datamodule.data_root, "summary.json"), "r") as f:
        summary = json.load(f)

    ckpt = torch.load(
        os.path.join(datamodule.data_root, summary["best_ckpt_path"]),
        map_location=get_device(),
    )
    ckpt_decoded = {"state_dict": {}}
    for i in range(summary["num_envs"]):
        index = 0
        for j in range(len(summary["hidden_layers"]) + 1):
            layer_name = f"model.ensemble_mlp.{i}.layers.{2 * j}"
            weight = ckpt["state_dict"][layer_name + ".weight"]
            weight_size = weight.shape[0] * weight.shape[1]
            ckpt_decoded["state_dict"][layer_name + ".weight"] = outputs[
                i, index : index + weight_size
            ].reshape(weight.shape[0], weight.shape[1])
            index += weight_size

            bias = ckpt["state_dict"][layer_name + ".bias"]
            ckpt_decoded["state_dict"][layer_name + ".bias"] = outputs[
                i, index : index + bias.shape[0]
            ]
            index += bias.shape[0]

    # Save to a .pt file
    embedding_path = os.path.join(output_dir, "embeddings.npy")
    np.save(embedding_path, embeddings.cpu().numpy())
    ckpt_path = os.path.join(output_dir, "ckpt_decoded.pt")
    torch.save(ckpt_decoded, ckpt_path)

    load_and_visualize_embeddings(output_dir)

    shutil.copy2(
        os.path.join(datamodule.data_root, "dr_params.pkl"),
        os.path.join(output_dir, "dr_params.pkl"),
    )

    wandb.finish()


if __name__ == "__main__":
    main()
