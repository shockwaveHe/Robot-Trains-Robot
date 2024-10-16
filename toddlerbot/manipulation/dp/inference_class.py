import collections

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm

from toddlerbot.manipulation.dp.models.diffusion_model import ConditionalUnet1D
from toddlerbot.manipulation.dp.utils.dataset_utils import (
    normalize_data,
    unnormalize_data,
)
from toddlerbot.manipulation.dp.utils.model_utils import get_resnet, replace_bn_with_gn


class DPModel:
    def __init__(
        self,
        ckpt_path,
        pred_horizon,
        obs_horizon,
        action_horizon,
        lowdim_obs_dim,
        action_dim,
        stats=None,
    ):
        # |o|o|                             observations: 2
        # | |a|a|a|a|a|a|a|a|               actions executed: 8
        # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

        # net definitions
        self.vision_feature_dim = 512
        self.lowdim_obs_dim = lowdim_obs_dim
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.obs_dim = self.vision_feature_dim + self.lowdim_obs_dim

        # initialize scheduler
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule="squaredcos_cap_v2",
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type="epsilon",
        )

        # stats is get from dataset to denormalize data
        if stats is not None:
            self.stats = stats

        # initialize the network
        self.ckpt_path = ckpt_path
        self.load_model()

    def load_model(self):
        # Construct the network
        vision_encoder = get_resnet("resnet18")
        vision_encoder = replace_bn_with_gn(vision_encoder)

        noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim, global_cond_dim=self.obs_dim * self.obs_horizon
        )

        self.ema_nets = nn.ModuleDict(
            {"vision_encoder": vision_encoder, "noise_pred_net": noise_pred_net}
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("mps")
        self.ema_nets = self.ema_nets.to(self.device)

        state_dict = torch.load(self.ckpt_path, map_location=self.device)
        self.ema_nets.load_state_dict(state_dict)
        print("Pretrained weights loaded.")

        self.ema_nets.eval()

    def prepare_inputs(self, obs_deque):
        # stack the last obs_horizon number of observations
        images = np.stack([x["image"] for x in obs_deque])
        agent_poses = np.stack([x["agent_pos"] for x in obs_deque])

        # normalize observation
        nagent_poses = normalize_data(agent_poses, stats=self.stats["agent_pos"])
        # images are already normalized to [0,1]
        nimages = images

        # device transfer
        nimages = torch.from_numpy(nimages).to(
            self.device, dtype=torch.float32
        )  # (2,3,96,96)
        nagent_poses = torch.from_numpy(nagent_poses).to(
            self.device, dtype=torch.float32
        )  # (2,2)

        return nimages, nagent_poses

    def prediction_to_action(self, naction):
        # denormalize action
        naction = naction.detach().to("cpu").numpy()  # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=self.stats["action"])

        # only take action_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        action = action_pred[start:end, :]  # (action_horizon, action_dim)
        return action

    def get_action_from_obs(self, obs_deque):
        # prepare inputs
        nimages, nagent_poses = self.prepare_inputs(obs_deque)

        # generate denoised sample
        with torch.no_grad():
            # get image features
            image_features = self.ema_nets["vision_encoder"](nimages)  # (2,512)

            # concat with low-dim observations
            obs_features = torch.cat([image_features, nagent_poses], dim=-1)

            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

            # initialize n(oisy) action from Guassian noise
            B = 1
            naction = torch.randn(
                (B, self.pred_horizon, self.action_dim), device=self.device
            )

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.ema_nets["noise_pred_net"](
                    sample=naction, timestep=k, global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
                ).prev_sample

        # unpack to our format
        action = self.prediction_to_action(naction)

        return action


if __name__ == "__main__":
    from skvideo.io import vwrite

    from datasets.pusht_dataset import PushTImageDataset
    from toddlerbot.manipulation.dp.envs.pusht_env import PushTImageEnv

    pred_horizon, obs_horizon, action_horizon = 16, 2, 8
    lowdim_obs_dim, action_dim = 2, 2

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

    model = DPModel(
        "pusht_vision_100ep.ckpt",
        pred_horizon,
        obs_horizon,
        action_horizon,
        lowdim_obs_dim,
        action_dim,
        stats,
    )

    # ### **Inference**

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
            action = model.get_action_from_obs(obs_deque)

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
