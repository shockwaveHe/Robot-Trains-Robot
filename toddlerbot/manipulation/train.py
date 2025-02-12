import argparse
import os
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from toddlerbot.manipulation.datasets.teleop_dataset import TeleopImageDataset
from toddlerbot.manipulation.models.diffusion_model import ConditionalUnet1D
from toddlerbot.manipulation.utils.model_utils import get_resnet, replace_bn_with_gn
from toddlerbot.visualization.vis_plot import plot_line_graph


def train(
    dataset_path_list: List[str],
    exp_folder_path: str,
    weights: str,
    pred_horizon: int,
    obs_horizon: int,
    action_horizon: int,
    action_dim: int = 16,
    vision_feature_dim: int = 512,
    num_diffusion_iters: int = 100,
    num_epochs: int = 1000,
    early_stopping_patience: int = 100,  # Stop if no improvement for X epochs
    train_split_ratio: float = 0.8,
):
    """Trains a neural network model using a dataset of teleoperation images and actions.

    Args:
        dataset_path_list (List[str]): List of paths to the datasets.
        exp_folder_path (str): Path to the folder where experiment outputs will be saved.
        weights (str): Pre-trained weights for the vision encoder.
        pred_horizon (int): Prediction horizon for the model.
        obs_horizon (int): Observation horizon for the model.
        action_horizon (int): Action horizon for the model.
        action_dim (int, optional): Dimensionality of the action space. Defaults to 16.
        vision_feature_dim (int, optional): Dimensionality of the vision feature space. Defaults to 512.
        num_diffusion_iters (int, optional): Number of diffusion iterations. Defaults to 100.
        num_epochs (int, optional): Number of training epochs. Defaults to 1000.
        early_stopping_patience (int, optional): Number of epochs to wait for improvement before stopping early. Defaults to 100.
        train_split_ratio (float, optional): Ratio of the dataset to use for training. Defaults to 0.8.
    """
    plt.switch_backend("Agg")

    # ### **Network Demo**
    dataset = TeleopImageDataset(
        dataset_path_list, exp_folder_path, pred_horizon, obs_horizon, action_horizon
    )

    # Split into train/val sets
    train_size = int(len(dataset) * train_split_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )

    # construct ResNet18 encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    vision_encoder = get_resnet(
        "resnet18",
        weights=None if len(weights) == 0 else models.ResNet18_Weights.IMAGENET1K_V1,
    )
    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
    # performance will tank if you forget to do this!
    vision_encoder = replace_bn_with_gn(vision_encoder)

    # ResNet18 has output dim of 512
    # agent_pos is 2 dimensional
    # observation feature has 514 dims in total per step
    lowdim_obs_dim = action_dim
    obs_dim = vision_feature_dim + lowdim_obs_dim
    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
        down_dims=[128, 256, 384],
    )

    # the final arch has 2 parts
    nets = nn.ModuleDict(
        {"vision_encoder": vision_encoder, "noise_pred_net": noise_pred_net}
    )

    print(
        "ve # weights: ",
        np.sum([param.nelement() for param in vision_encoder.parameters()]),
    )
    print(
        "unet # weights: ",
        np.sum([param.nelement() for param in noise_pred_net.parameters()]),
    )

    # for this demo, we use DDPMScheduler with 100 diffusion iterations
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule="squaredcos_cap_v2",
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type="epsilon",
    )

    # device transfer
    device = torch.device("cuda")
    _ = nets.to(device)

    # ### **Training**
    ema = EMAModel(parameters=nets.parameters(), power=0.75)
    optimizer = torch.optim.AdamW(params=nets.parameters(), lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_dataloader) * num_epochs,
    )

    params = {
        "weights": weights,
        "pred_horizon": pred_horizon,
        "obs_horizon": obs_horizon,
        "action_horizon": action_horizon,
        "vision_feature_dim": vision_feature_dim,
        "lowdim_obs_dim": lowdim_obs_dim,
        "action_dim": action_dim,
    }
    print(params)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses_per_epoch = []
    val_losses_per_epoch = []

    tglobal = tqdm(range(num_epochs), desc="Epoch")
    try:
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(train_dataloader, desc="Batch", leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nimage = nbatch["image"][:, :obs_horizon].to(device)
                    nagent_pos = nbatch["agent_pos"][:, :obs_horizon].to(device)
                    naction = nbatch["action"].to(device)
                    B = nagent_pos.shape[0]

                    # encoder vision features
                    image_features = nets["vision_encoder"](nimage.flatten(end_dim=1))
                    image_features = image_features.reshape(*nimage.shape[:2], -1)
                    # (B,obs_horizon,D)

                    # concatenate vision feature and low-dim obs
                    obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)
                    # (B, obs_horizon * obs_dim)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (B,),
                        device=device,
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, global_cond=obs_cond
                    )

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(nets.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

            mean_train_loss = np.mean(epoch_loss)
            tglobal.set_postfix(train_loss=mean_train_loss)
            train_losses_per_epoch.append(mean_train_loss)

            # Validation loop
            nets.eval()
            val_losses = []
            with torch.no_grad():
                for nbatch in val_dataloader:
                    nimage = nbatch["image"][:, :obs_horizon].to(device)
                    nagent_pos = nbatch["agent_pos"][:, :obs_horizon].to(device)
                    naction = nbatch["action"].to(device)
                    B = nagent_pos.shape[0]

                    image_features = nets["vision_encoder"](nimage.flatten(end_dim=1))
                    image_features = image_features.reshape(*nimage.shape[:2], -1)

                    obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)

                    noise = torch.randn(naction.shape, device=device)

                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (B,),
                        device=device,
                    ).long()

                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                    noise_pred = nets["noise_pred_net"](
                        noisy_actions, timesteps, global_cond=obs_cond
                    )

                    val_loss = nn.functional.mse_loss(noise_pred, noise)
                    val_losses.append(val_loss.item())

            mean_val_loss = np.mean(val_losses)
            tglobal.set_postfix(train_loss=mean_train_loss, val_loss=mean_val_loss)
            val_losses_per_epoch.append(mean_val_loss)

            # Early stopping check
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss.item()
                patience_counter = 0
                # Update EMA nets if this is the best so far
                best_ema_nets = nn.ModuleDict(
                    {"vision_encoder": vision_encoder, "noise_pred_net": noise_pred_net}
                )
                ema.copy_to(best_ema_nets.parameters())
                # Save best model checkpoint immediately
                ckpt_path = os.path.join(exp_folder_path, "best_ckpt.pth")
                torch.save(
                    {
                        "state_dict": best_ema_nets.state_dict(),
                        "stats": dataset.stats,
                        "params": params,
                    },
                    ckpt_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break

    except KeyboardInterrupt:
        pass

    # If we stopped early, best checkpoint is already saved.
    # If we completed all epochs without triggering early stopping, save final EMA weights.
    if patience_counter < early_stopping_patience:
        ema_nets = nets
        ema.copy_to(ema_nets.parameters())

        ckpt_path = os.path.join(exp_folder_path, "last_ckpt.pth")
        # Save final model checkpoint
        torch.save(
            {
                "state_dict": ema_nets.state_dict(),
                "stats": dataset.stats,
                "params": params,
            },
            ckpt_path,
        )

    # Plot loss
    plot_line_graph(
        [train_losses_per_epoch, val_losses_per_epoch],
        legend_labels=["Train Loss", "Val Loss"],
        title="Training and Validation Loss Over Epochs",
        x_label="Epoch",
        y_label="Loss",
        save_config=True,
        save_path=exp_folder_path,
        file_name="loss_plot",
    )()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw data to create dataset.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in descriptions.",
        choices=["toddlerbot", "toddlerbot_gripper"],
    )
    parser.add_argument(
        "--task",
        type=str,
        default="hug",
        help="The manipulation task.",
    )
    parser.add_argument(
        "--time-str",
        type=str,
        default="",
        help="The time str of the dataset.",
    )
    parser.add_argument(
        "--pred-horizon",
        type=int,
        default=16,
        help="The horizon of the prediction.",
    )
    parser.add_argument(
        "--obs-horizon",
        type=int,
        default=7,
        help="The horizon of the observation.",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=8,
        help="The horizon of the action.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="The pretrained weights.",
    )

    args = parser.parse_args()

    time_str_list = args.time_str.split(" ")
    dataset_path_list = []
    for time_str in time_str_list:
        if len(time_str) > 0:
            dataset_path_list.append(
                os.path.join(
                    "datasets", f"{args.task}_dataset_{time_str}", "dataset.lz4"
                )
            )

    exp_name = f"{args.robot}_{args.task}_dp"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    exp_folder_path = f"results/{exp_name}_{time_str}"
    os.makedirs(exp_folder_path, exist_ok=True)
    print(f"Experiment folder: {exp_folder_path}")

    train(
        dataset_path_list,
        exp_folder_path,
        args.weights,
        args.pred_horizon,
        args.obs_horizon,
        args.action_horizon,
        action_dim=8 if args.task == "pick" else 16,
    )
