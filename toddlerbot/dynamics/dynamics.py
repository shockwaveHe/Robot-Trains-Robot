import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from networks import DynamicsNetwork
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

import wandb
from toddlerbot.autoencoder.base_system import BaseSystem


class DynamicsModel(BaseSystem):
    def __init__(self, cfg) -> None:
        super(DynamicsModel, self).__init__(cfg)
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.data_cfg = cfg.data
        self.train_cfg = cfg.train
        self.model_cfg = cfg.model
        self.hist = cfg.data.history
        self.model = self.build_model()
        self.loss_func = self.build_loss_func()
        self.test_results = []

    def training_step(self, batch, batch_idx, **kwargs):
        optimizer = self.optimizers()
        loss = self.forward(batch, batch_idx, **kwargs)
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        if hasattr(self, "lr_scheduler"):
            self.lr_scheduler.step()

        wandb.log({"train/loss": loss})

        return {"loss": loss}

    def build_model(self, **kwargs):
        self.model = DynamicsNetwork(
            self.model_cfg.observation_size,
            self.model_cfg.action_size,
            self.model_cfg.hidden_layers,
            self.hist,
            self.data_cfg.num_envs,
            self.data_cfg.step_batch_size,
            self.data_cfg.env_batch_size,
        )

        return self.model

    def build_loss_func(self):
        if "loss_func" in self.train_cfg:
            loss_func = hydra.utils.instantiate(self.train_cfg.loss_func)
            return loss_func

    def configure_optimizers(self, **kwargs):
        self.optimizer = hydra.utils.instantiate(
            self.train_cfg.optimizer, self.model.parameters()
        )

        self.lr_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        return self.optimizer

    def validation_step(self, batch, batch_idx, **kwargs):
        obs, act = batch
        val_loss = 0
        for i in range(self.data_cfg.horizon):
            obs_ensemble = obs[:, i : i + self.hist]
            act_ensemble = act[:, i : i + self.hist]
            obs_hat = self.model(obs_ensemble, act_ensemble, batch_idx)
            obs_next = obs[:, i + self.hist]
            val_loss += self.loss_func(obs_hat, obs_next)

        wandb.log({"val/loss": val_loss.detach()})
        self.log(
            "val_loss",
            val_loss.cpu().detach().mean().item(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"val_loss": val_loss}

    def forward(self, batch, batch_idx, **kwargs):
        obs, act = batch
        total_loss = 0
        for i in range(self.data_cfg.horizon):
            obs_ensemble = obs[:, i : i + self.hist]
            act_ensemble = act[:, i : i + self.hist]
            obs_hat = self.model(obs_ensemble, act_ensemble, batch_idx)
            obs_next = obs[:, i + self.hist]
            total_loss += self.loss_func(obs_hat, obs_next)

        self.log(
            "loss",
            total_loss.cpu().detach().mean().item(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return total_loss

    def test_step(self, batch, batch_idx):
        """Processes a full sequence at once for testing"""
        obs, act = batch

        # Initialize rollout storage
        obs_pred = torch.zeros_like(obs)
        obs_pred[:, : self.hist] = obs[:, : self.hist]

        loss_list = []
        # Rollout loop
        with torch.no_grad():
            for i in tqdm(range(obs.shape[1] - self.hist), desc="Rollout"):
                obs_ensemble = obs_pred[:, i : i + self.hist]
                act_ensemble = act[:, i : i + self.hist]
                obs_pred[:, i + self.hist] = self.model(
                    obs_ensemble, act_ensemble, batch_idx
                )
                loss_list.append(
                    torch.square(
                        obs_pred[:, i + self.hist] - obs[:, i + self.hist]
                    ).mean(dim=-1)
                )

        # Return data for `test_epoch_end`
        self.test_results.append(
            {
                "loss_list": torch.stack(loss_list).cpu(),
                "obs": obs.cpu(),
                "obs_pred": obs_pred.cpu(),
            }
        )

    def on_test_epoch_start(self):
        self.model.num_envs = self.data_cfg.num_test_envs
        self.model.env_batch_size = self.data_cfg.num_test_envs
        self.model.step_batch_size = 1

    def on_test_epoch_end(self):
        """Aggregates losses and plots PCA comparisons at different rollout stages."""
        # Aggregate loss over time
        plt.switch_backend("Agg")

        loss_list = torch.cat([res["loss_list"] for res in self.test_results]).numpy()
        # Extract full observation rollouts
        obs_true_full = torch.cat(
            [res["obs"] for res in self.test_results], dim=0
        ).numpy()
        obs_pred_full = torch.cat(
            [res["obs_pred"] for res in self.test_results], dim=0
        ).numpy()

        num_envs, num_steps, feature_dim = obs_true_full.shape
        # Compute PCA projections at different time checkpoints
        fraction_indices = [
            self.hist,
            int(num_steps / 4),
            int(num_steps / 2),
            int(num_steps * 3 / 4),
            num_steps - 1,
        ]
        pca_results = []

        for i in fraction_indices:
            final_obs_true = obs_true_full[:, i]
            final_obs_pred = obs_pred_full[:, i]

            # PCA transformation
            pca = PCA(n_components=2)
            combined_obs = np.concatenate(
                [final_obs_true, final_obs_pred], axis=0
            ).reshape(2 * num_envs, -1)
            reduced_obs = pca.fit_transform(combined_obs).reshape(2, num_envs, -1)

            # Split PCA results
            reduced_obs_true = reduced_obs[0]
            reduced_obs_pred = reduced_obs[1]
            pca_results.append((reduced_obs_true, reduced_obs_pred))

        # Plot the results in a 2x3 grid
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        # 1st plot: Loss over time
        # Compute mean and standard deviation over the experiments (axis=1)
        mean_loss = np.mean(loss_list, axis=1)
        std_loss = np.std(loss_list, axis=1)
        median_loss = np.median(loss_list, axis=1)
        # Create x-axis values (time steps)
        time_steps = np.arange(len(mean_loss))
        # Plot the mean loss
        axes[0, 0].plot(time_steps, mean_loss, label="Loss over time", color="r")
        # Fill the area between (mean - std) and (mean + std)
        axes[0, 0].fill_between(
            time_steps,
            mean_loss - std_loss,
            mean_loss + std_loss,
            color="r",
            alpha=0.3,
            label="±1 std",
        )
        # Plot the median loss
        axes[0, 0].plot(time_steps, median_loss, label="Median loss", color="b")

        axes[0, 0].set_xlabel("Time Step")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Loss Change Over Time")
        axes[0, 0].set_ylim(0, 0.2)
        axes[0, 0].legend()
        axes[0, 0].grid()

        # PCA plots at different timesteps
        for i, (reduced_obs_true, reduced_obs_pred) in enumerate(pca_results):
            ax = axes[(i + 1) // 3, (i + 1) % 3]
            ax.scatter(
                reduced_obs_true[:, 0], reduced_obs_true[:, 1], label="True", alpha=0.6
            )
            ax.scatter(
                reduced_obs_pred[:, 0],
                reduced_obs_pred[:, 1],
                label="Predicted",
                alpha=0.6,
            )
            # Combine true and predicted points for percentile calculation
            all_points = np.vstack([reduced_obs_true, reduced_obs_pred])

            # Calculate 5th and 95th percentiles for x and y
            x_min, x_max = np.percentile(all_points[:, 0], [5, 95])
            y_min, y_max = np.percentile(all_points[:, 1], [5, 95])

            # Set the axes limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            ax.set_title(f"PCA Comparison at {fraction_indices[i]} of Prediction")
            ax.legend()
            ax.grid()

        plt.tight_layout()

        output_dir = os.getcwd()
        plt.savefig(os.path.join(output_dir, "pca_comparison.png"))
