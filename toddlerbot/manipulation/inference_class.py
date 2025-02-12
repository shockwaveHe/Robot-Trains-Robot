import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from toddlerbot.manipulation.models.diffusion_model import ConditionalUnet1D
from toddlerbot.manipulation.utils.dataset_utils import (
    normalize_data,
    unnormalize_data,
)
from toddlerbot.manipulation.utils.model_utils import get_resnet, replace_bn_with_gn


class DPModel:
    def __init__(self, ckpt_path, stats=None):
        """Initializes the model by setting up the device, loading parameters from a checkpoint, and configuring the noise schedulers.

        Args:
            ckpt_path (str): Path to the checkpoint file containing model parameters.
            stats (optional): Additional statistics or configurations for model initialization. Defaults to None.
        """

        # |o|o|                             observations: 2
        # | |a|a|a|a|a|a|a|a|               actions executed: 8
        # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"

        # Check if CUDA is available
        self.device = torch.device(device_str)
        # Load parameters and move to the correct device
        params = torch.load(ckpt_path, map_location=self.device)["params"]

        # net definitions
        self.weights = (
            None
            if len(params["weights"]) == 0
            else models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.vision_feature_dim = params["vision_feature_dim"]
        self.lowdim_obs_dim = params["lowdim_obs_dim"]
        self.action_dim = params["action_dim"]
        self.pred_horizon = params["pred_horizon"]
        self.obs_horizon = params["obs_horizon"]
        self.action_horizon = params["action_horizon"]
        self.obs_dim = self.vision_feature_dim + self.lowdim_obs_dim

        self.down_dims = None
        self.down_dims = [128, 256, 384]

        # initialize scheduler
        self.num_diffusion_iters = 100  # n steps trained on
        self.noise_scheduler_ddpm = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule="squaredcos_cap_v2",
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type="epsilon",
        )

        self.noise_scheduler_ddim = DDIMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule="squaredcos_cap_v2",
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type="epsilon",
            # set_alpha_to_one=False,
            timestep_spacing="linspace",
        )

        # initialize the network
        self.load_model(ckpt_path, stats=stats)

    def load_model(self, ckpt_path, stats=None):
        """Loads a pre-trained model from a checkpoint file and initializes the network components.

        Args:
            ckpt_path (str): Path to the checkpoint file containing the model's state dictionary.
            stats (dict, optional): Pre-computed statistics for the model. If not provided, statistics will be loaded from the checkpoint.

        Loads the model's state dictionary and statistics from the checkpoint, or uses provided statistics. Sets the model to evaluation mode.
        """
        # Construct the network
        vision_encoder = get_resnet("resnet18", weights=self.weights)
        vision_encoder = replace_bn_with_gn(vision_encoder)

        if self.down_dims is None:
            noise_pred_net = ConditionalUnet1D(
                input_dim=self.action_dim,
                global_cond_dim=self.obs_dim * self.obs_horizon,
            )
        else:
            noise_pred_net = ConditionalUnet1D(
                input_dim=self.action_dim,
                global_cond_dim=self.obs_dim * self.obs_horizon,
                down_dims=self.down_dims,
            )

        self.ema_nets = nn.ModuleDict(
            {"vision_encoder": vision_encoder, "noise_pred_net": noise_pred_net}
        )

        self.ema_nets = self.ema_nets.to(self.device)

        if stats is None:
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.ema_nets.load_state_dict(state_dict["state_dict"])
            self.stats = state_dict["stats"]
        else:
            self.stats = stats
            self.ema_nets.load_state_dict(
                torch.load(ckpt_path, map_location=self.device)
            )
        print("Pretrained weights loaded.")

        self.ema_nets.eval()

    def prepare_inputs(self, obs_deque):
        """Prepares and normalizes input data for model processing.

        This function stacks and normalizes the last set of observations from a deque,
        transferring them to the specified device for further processing.

        Args:
            obs_deque (collections.deque): A deque containing the most recent observations,
                where each observation is a dictionary with keys "image" and "agent_pos".

        Returns:
            tuple: A tuple containing:
                - nimages (torch.Tensor): A tensor of stacked and normalized images,
                  transferred to the specified device.
                - nagent_poses (torch.Tensor): A tensor of normalized agent positions,
                  transferred to the specified device.
        """
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
        """Converts a normalized action prediction to a denormalized action sequence.

        This method takes a normalized action prediction tensor, detaches it from the computation graph, and converts it to a NumPy array. It then denormalizes the action data using predefined statistics and extracts a sequence of actions based on the specified action horizon.

        Args:
            naction (torch.Tensor): A tensor containing the normalized action predictions with shape (B, pred_horizon, action_dim).

        Returns:
            numpy.ndarray: A denormalized action sequence with shape (action_horizon, action_dim).
        """
        # denormalize action
        naction = naction.detach().to("cpu").numpy()  # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=self.stats["action"])

        # only take action_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        action = action_pred[start:end, :]  # (action_horizon, action_dim)
        return action

    def inference_ddim(self, obs_cond, nsteps=10, naction=None):
        """Performs DDIM (Denoising Diffusion Implicit Models) inference to generate actions based on observed conditions.

        Args:
            obs_cond (torch.Tensor): The observed conditions used as input to condition the action generation.
            nsteps (int, optional): The number of diffusion steps to perform. Defaults to 10.
            naction (torch.Tensor, optional): Initial noisy actions. If None, actions are initialized from Gaussian noise.

        Returns:
            torch.Tensor: The denoised actions after performing the specified number of diffusion steps.
        """
        # initialize n(oisy) action from Guassian noise
        B = 1
        if naction is None:
            naction = torch.randn(
                (B, self.pred_horizon, self.action_dim), device=self.device
            )

        # init scheduler
        self.noise_scheduler_ddim.set_timesteps(nsteps)

        for k in self.noise_scheduler_ddim.timesteps:
            # predict noise
            noise_pred = self.ema_nets["noise_pred_net"](
                sample=naction, timestep=k, global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler_ddim.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample

        return naction

    def inference_ddpm(self, obs_cond, nsteps, naction=None):
        """Performs inference using the Denoising Diffusion Probabilistic Model (DDPM) to generate actions.

        This function initializes a noisy action from Gaussian noise and iteratively refines it using a noise prediction network and a noise scheduler. The process involves predicting noise at each timestep and removing it to obtain a cleaner action sample.

        Args:
            obs_cond (Tensor): The observation condition tensor used as global conditioning input for the noise prediction network.
            nsteps (int): The number of diffusion steps to perform during the inference process.
            naction (Tensor, optional): Initial noisy action tensor. If not provided, it is initialized from Gaussian noise.

        Returns:
            Tensor: The refined action tensor after performing the inverse diffusion process.
        """
        # initialize n(oisy) action from Guassian noise
        B = 1
        if naction is None:
            naction = torch.randn(
                (B, self.pred_horizon, self.action_dim), device=self.device
            )

        # init scheduler
        self.noise_scheduler_ddpm.set_timesteps(nsteps)

        for k in self.noise_scheduler_ddpm.timesteps:
            # predict noise
            noise_pred = self.ema_nets["noise_pred_net"](
                sample=naction, timestep=k, global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler_ddpm.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample

        return naction

    def get_action_from_obs(self, obs_deque):
        """Generate an action based on a sequence of observations.

        This function processes a deque of observations to prepare inputs, extracts features using a vision encoder, and performs inference to generate an action. The action is derived from denoised samples obtained through a diffusion model.

        Args:
            obs_deque (collections.deque): A deque containing the sequence of observations.

        Returns:
            torch.Tensor: The generated action in the required format.
        """
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

            # inference
            # naction = self.inference_ddim(obs_cond, nsteps=2)
            # naction = self.inference_ddpm(obs_cond, nsteps=2, naction=naction)

            naction = self.inference_ddpm(obs_cond, nsteps=3)

        # unpack to our format
        action = self.prediction_to_action(naction)

        return action
