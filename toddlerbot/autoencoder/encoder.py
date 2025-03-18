import json
import os

import hydra
import matplotlib.pyplot as plt
import torch
import torch.optim.lr_scheduler
from base_system import BaseSystem
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import wandb
from toddlerbot.autoencoder.network import EncoderDecoder


class Encoder(BaseSystem):
    def __init__(self, cfg):
        super(Encoder, self).__init__(cfg)
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.train_cfg = cfg.train
        self.model_cfg = cfg.model

        self.data_root = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "results",
            f"dynamics_model_{getattr(cfg.data, 'time_str')}",
        )
        # Check the root path
        # assert os.path.exists(self.data_root), f'{self.data_root} not exists'
        with open(os.path.join(self.data_root, "summary.json"), "r") as f:
            summary = json.load(f)

        input_dim = summary["history"] * (
            summary["observation_size"] + summary["action_size"]
        )
        first_layer_size = (input_dim + 1) * summary["hidden_layers"][0]
        param_size = 0
        for i in range(len(summary["hidden_layers"]) + 1):
            if i == 0:
                param_size += (input_dim + 1) * summary["hidden_layers"][i]
            elif i == len(summary["hidden_layers"]):
                param_size += (summary["hidden_layers"][i - 1] + 1) * summary[
                    "observation_size"
                ]
            else:
                param_size += (summary["hidden_layers"][i - 1] + 1) * summary[
                    "hidden_layers"
                ][i]

        self.input_splits = [first_layer_size, param_size - first_layer_size]
        self.model = self.build_model()
        self.loss_func = self.build_loss_func()
        self.test_results = []

    def training_step(self, batch, **kwargs):
        optimizer = self.optimizers()
        param = batch
        loss = self.forward(param, **kwargs)
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        if hasattr(self, "lr_scheduler"):
            self.lr_scheduler.step()
        wandb.log({"train/loss": loss})
        return {"loss": loss}

    def build_model(self, **kwargs):
        self.model = EncoderDecoder(
            self.input_splits,
            self.model_cfg.n_embd,
            self.model_cfg.encoder_depth,
            self.model_cfg.decoder_depth,
            self.model_cfg.input_noise_factor,
            self.model_cfg.latent_noise_factor,
            self.model_cfg.is_vae,
        )
        if self.train_cfg.finetune:
            self.load_encoder(self.train_cfg.pretrain_model)

        return self.model

    def build_loss_func(self):
        # Instantiate the reconstruction loss function from the config
        recon_loss_func = hydra.utils.instantiate(self.train_cfg.loss_func)
        # Get the KL weight from the config if in VAE mode (default to 1.0 if not specified)
        kl_weight = (
            self.train_cfg.get("kl_weight", 1.0) if self.model_cfg.is_vae else None
        )

        def loss_func(recon, mean, logvar, target):
            recon_loss = recon_loss_func(recon, target)
            # If in VAE mode and mean/logvar are provided, add KL divergence
            if self.model_cfg.is_vae and mean is not None and logvar is not None:
                # Compute KL divergence loss
                kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
                total_loss = recon_loss + kl_weight * kl_loss
            else:
                # In AE mode, use only the reconstruction loss
                total_loss = recon_loss

            return total_loss

        return loss_func

    def configure_optimizers(self, **kwargs):
        parameters = self.model.parameters()
        self.optimizer = hydra.utils.instantiate(self.train_cfg.optimizer, parameters)

        self.lr_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        return self.optimizer

    def validation_step(self, batch, batch_idx, **kwargs):
        z, mean, logvar = self.encode(batch)
        x_recon = self.decode(z)
        val_loss = self.loss_func(x_recon, mean, logvar, batch)
        wandb.log({"val/loss": val_loss.detach()})
        self.log(
            "val_loss",
            val_loss.cpu().detach().mean().item(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"val_loss": val_loss}

    def forward(self, batch, **kwargs):
        x_recon, mean, logvar = self.model(batch)
        loss = self.loss_func(x_recon, mean, logvar, batch, **kwargs)
        self.log(
            "loss",
            loss.cpu().detach().mean().item(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def encode(self, x, **kwargs):
        return self.model.encode(x)

    def decode(self, z, **kwargs):
        return self.model.decode(z)

    def load_encoder(self, encoder_path, evaluate=False):
        print("Loading encoders from {}".format(encoder_path))
        encoder_ckpt = torch.load(encoder_path, map_location="cpu")
        weights_dict = {}
        weights = encoder_ckpt["state_dict"]
        for k, v in weights.items():
            new_k = k.replace("model.", "") if "model." in k else k
            weights_dict[new_k] = v
        self.model.load_state_dict(weights_dict)

    def test_step(self, batch, batch_idx):
        """Processes a full sequence at once for testing"""
        z, mean, logvar = self.encode(batch)
        x_recon = self.decode(z)
        test_loss = self.loss_func(x_recon, mean, logvar, batch)
        # Return data for `test_epoch_end`
        self.test_results.append({"loss": test_loss.cpu()})

    def predict_step(self, batch, batch_idx):
        z, mean, logvar = self.encode(batch)
        x_recon = self.decode(z)
        pred_loss = self.loss_func(x_recon, mean, logvar, batch)
        return z.cpu(), x_recon.cpu(), pred_loss.cpu()

    def on_test_epoch_end(self):
        """Aggregates losses and plots PCA comparisons at different rollout stages."""
        # Aggregate loss over time
        plt.switch_backend("Agg")

        loss = torch.stack([res["loss"] for res in self.test_results]).numpy()

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # 1st plot: Loss over time
        ax.plot(loss, label="Loss", color="r")
        ax.set_xlabel("Env Index")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Change Over Envs")
        ax.legend()
        ax.grid()

        plt.tight_layout()

        output_dir = os.getcwd()
        plt.savefig(os.path.join(output_dir, "env_loss.png"))
