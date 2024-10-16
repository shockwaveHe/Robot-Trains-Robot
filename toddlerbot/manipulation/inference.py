import collections
import os

import gdown
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from envs.pusht_env import PushTImageEnv
from models.diffusion_model import ConditionalUnet1D
from skvideo.io import vwrite
from tqdm.auto import tqdm
from utils.dataset_utils import normalize_data, unnormalize_data
from utils.model_utils import get_resnet, replace_bn_with_gn

from datasets.pusht_dataset import PushTImageDataset

if __name__ == "__main__":
    # construct ResNet18 encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    vision_encoder = get_resnet("resnet18")

    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
    # performance will tank if you forget to do this!
    vision_encoder = replace_bn_with_gn(vision_encoder)

    # ResNet18 has output dim of 512
    vision_feature_dim = 512
    # agent_pos is 2 dimensional
    lowdim_obs_dim = 2
    # observation feature has 514 dims in total per step
    obs_dim = vision_feature_dim + lowdim_obs_dim
    action_dim = 2

    # parameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    # |o|o|                             observations: 2
    # | |a|a|a|a|a|a|a|a|               actions executed: 8
    # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    # create dataset from file
    dataset_path = "pusht_cchi_v7_replay.zarr.zip"
    dataset = PushTImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
    )
    # save training data statistics (min, max) for each dim
    stats = dataset.stats

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
    device = torch.device("mps")
    _ = nets.to(device)

    # ### **Inference**

    # ### **Loading Pretrained Checkpoint**
    # Set `load_pretrained = True` to load pretrained weights.

    load_pretrained = True
    if load_pretrained:
        ckpt_path = "pusht_vision_100ep.ckpt"
        if not os.path.isfile(ckpt_path):
            id = "1XKpfNSlwYMGaF5CncoFaLKCDTWoLAHf1&confirm=t"
            gdown.download(id=id, output=ckpt_path, quiet=False)

        state_dict = torch.load(ckpt_path, map_location="mps")
        ema_nets = nets
        ema_nets.load_state_dict(state_dict)
        print("Pretrained weights loaded.")
    else:
        print("Skipped pretrained weight loading.")

    # limit enviornment interaction to 200 steps before termination
    max_steps = 200
    env = PushTImageEnv()
    # use a seed >200 to avoid initial states seen in the training dataset
    env.seed(100000)

    # get first observation
    obs, info = env.reset()

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    imgs = [env.render(mode="rgb_array")]
    rewards = list()
    done = False
    step_idx = 0

    with tqdm(total=max_steps, desc="Eval PushTImageEnv") as pbar:
        while not done:
            B = 1
            # stack the last obs_horizon number of observations
            images = np.stack([x["image"] for x in obs_deque])
            agent_poses = np.stack([x["agent_pos"] for x in obs_deque])

            # normalize observation
            nagent_poses = normalize_data(agent_poses, stats=stats["agent_pos"])
            # images are already normalized to [0,1]
            nimages = images

            # device transfer
            nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
            # (2,3,96,96)
            nagent_poses = torch.from_numpy(nagent_poses).to(
                device, dtype=torch.float32
            )
            # (2,2)

            # infer action
            with torch.no_grad():
                # get image features
                image_features = ema_nets["vision_encoder"](nimages)
                # (2,512)

                # concat with low-dim observations
                obs_features = torch.cat([image_features, nagent_poses], dim=-1)

                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn((B, pred_horizon, action_dim), device=device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = ema_nets["noise_pred_net"](
                        sample=naction, timestep=k, global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred, timestep=k, sample=naction
                    ).prev_sample

            # unnormalize action
            naction = naction.detach().to("cpu").numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=stats["action"])

            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end, :]
            # (action_horizon, action_dim)

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, _, info = env.step(action[i])
                # save observations
                obs_deque.append(obs)
                # and reward/vis
                rewards.append(reward)
                imgs.append(env.render(mode="rgb_array"))

                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                if done:
                    break

    # print out the maximum target coverage
    print("Score: ", max(rewards))

    # visualize
    vwrite("vis.mp4", imgs)
