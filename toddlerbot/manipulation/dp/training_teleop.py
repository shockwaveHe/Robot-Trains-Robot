import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils.model_utils import get_resnet, replace_bn_with_gn

from toddlerbot.manipulation.dp.datasets.teleop_dataset import TeleopImageDataset
from toddlerbot.manipulation.dp.models.diffusion_model import ConditionalUnet1D


def main(dataset_path: str, ckpt_path: str):
    # ### **Network Demo**

    # construct ResNet18 encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    vision_encoder = get_resnet("resnet18")

    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
    # performance will tank if you forget to do this!
    vision_encoder = replace_bn_with_gn(vision_encoder)

    vision_feature_dim = 512  # ResNet18 has output dim of 512
    lowdim_obs_dim = 16  # agent_pos is 2 dimensional
    obs_dim = (
        vision_feature_dim + lowdim_obs_dim
    )  # observation feature has 514 dims in total per step
    action_dim = 16

    # parameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    # |o|o|                             observations: 2
    # | |a|a|a|a|a|a|a|a|               actions executed: 8
    # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    # create dataset from file
    dataset = TeleopImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
    )

    # dataset_path = "pusht_cchi_v7_replay.zarr.zip"
    # lowdim_obs_dim = 2       # agent_pos is 2 dimensional
    # obs_dim = vision_feature_dim + lowdim_obs_dim # observation feature has 514 dims in total per step
    # action_dim = 2
    # dataset = PushTImageDataset(
    #     dataset_path=dataset_path,
    #     pred_horizon=pred_horizon,
    #     obs_horizon=obs_horizon,
    #     action_horizon=action_horizon,
    # )

    # stats = dataset.stats  # save training data statistics (min, max) for each dim

    # create dataloader
    dataloader: DataLoader = DataLoader(
        dataset,
        batch_size=32,  # was 64
        num_workers=2,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True,
    )

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim, global_cond_dim=obs_dim * obs_horizon
    )

    # the final arch has 2 parts
    nets = nn.ModuleDict(
        {"vision_encoder": vision_encoder, "noise_pred_net": noise_pred_net}
    )

    # for this demo, we use DDPMScheduler with 100 diffusion iterations
    num_diffusion_iters = 100
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
    num_epochs = 100
    ema = EMAModel(parameters=nets.parameters(), power=0.75)
    optimizer = torch.optim.AdamW(params=nets.parameters(), lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs,
    )

    with tqdm(range(num_epochs), desc="Epoch") as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc="Batch", leave=False) as tepoch:
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
            tglobal.set_postfix(loss=np.mean(epoch_loss))

    # Weights of the EMA model
    # is used for inference
    ema_nets = nets
    ema.copy_to(ema_nets.parameters())
    # Save model checkpoint to disk
    torch.save(ema_nets.state_dict(), ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw data to create dataset.")
    parser.add_argument(
        "--time-str",
        type=str,
        default="",
        help="The time str of the dataset.",
    )
    args = parser.parse_args()

    dataset_path = os.path.join("datasets", f"teleop_dataset_{args.time_str}.lz4")
    ckpt_path = os.path.join(
        "toddlerbot", "policies", "checkpoints", "teleop_model.pth"
    )
    main(dataset_path, ckpt_path)
