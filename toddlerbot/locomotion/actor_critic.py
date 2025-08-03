# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from toddlerbot.utils.math_utils import soft_clamp


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer with statistics tracking."""
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.film = nn.Linear(latent_dim, 2 * hidden_dim)
        self.register_buffer('gamma_mean', torch.zeros(1))
        self.register_buffer('beta_mean', torch.zeros(1))
        self.register_buffer('count', torch.zeros(1))
        
    def forward(self, hidden, z):
        # Generate modulation parameters
        film_params = self.film(z)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        
        # Update running statistics (for monitoring only)
        if self.training:
            with torch.no_grad():
                batch_mean_gamma = gamma.mean()
                batch_mean_beta = beta.mean()
                total = self.count + gamma.numel()
                self.gamma_mean = (self.gamma_mean * self.count + batch_mean_gamma * gamma.numel()) / total
                self.beta_mean = (self.beta_mean * self.count + batch_mean_beta * gamma.numel()) / total
                self.count = total
        
        # Apply feature-wise transformation
        return gamma * hidden + beta
    
    def reset_stats(self):
        """Reset tracking statistics"""
        self.gamma_mean.zero_()
        self.beta_mean.zero_()
        self.count.zero_()


def forward_log_det_jacobian(x):
    # 2 * (log(2) - x - softplus(-2x))
    return 2.0 * (torch.log(torch.tensor(2.0)) - x - F.softplus(-2.0 * x))


class TanhNormal(Normal):
    def __init__(self, loc, scale):
        self.normal = Normal(loc=loc, scale=scale)
        self.tanh_transform = TanhTransform(cache_size=1)
        self.dist = TransformedDistribution(self.normal, self.tanh_transform)

    def sample(self):
        return self.dist.rsample()

    def log_prob(self, value):
        # Account for tanh transform
        return self.dist.log_prob(value=value)

    @property
    def mean(self):
        return torch.tanh(self.normal.mean)

    @property
    def stddev(self):
        return self.normal.scale

    def entropy(self):
        pre_tanh_sample = self.dist.transforms[-1].inv(self.dist.rsample())
        log_det_jac = forward_log_det_jacobian(pre_tanh_sample)
        return self.normal.entropy() + log_det_jac


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        autoencoder_cfg=None,
        **kwargs,
    ):
        super().__init__()
        activation = torch.nn.SiLU()

        # Store latent dimension
        self.use_tan_normal = True
        if autoencoder_cfg is not None:
            latent_dim = autoencoder_cfg["model"]["n_embd"]
            self.latent_dim = latent_dim
            self.film_layers = nn.ModuleList()
            self.film_layers.append(FiLMLayer(latent_dim, actor_hidden_dims[0]))
            self.use_tan_normal = False       
        # Modified Actor with FiLM layers
        self.actor_layers = []
        
        # Input layer
        self.actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        self.actor_layers.append(activation)
        
        
        # Subsequent layers
        for i in range(1, len(actor_hidden_dims)):
            self.actor_layers.append(nn.Linear(actor_hidden_dims[i-1], actor_hidden_dims[i]))
            self.actor_layers.append(activation)
            if autoencoder_cfg is not None:
                self.film_layers.append(FiLMLayer(latent_dim, actor_hidden_dims[i]))
        
        # Output layer
        self.actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions * 2))
        self.actor = nn.Sequential(*self.actor_layers)
        
        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(
                    critic_hidden_dims[layer_index],
                    critic_hidden_dims[layer_index + 1],
                ))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Initialize zero effect FiLM parameters
        if autoencoder_cfg is not None:
            for film_layer in self.film_layers:
                nn.init.constant_(film_layer.film.weight, 0.0)
                nn.init.constant_(film_layer.film.bias[:film_layer.film.out_features//2], 1.0)
                nn.init.constant_(film_layer.film.bias[film_layer.film.out_features//2:], 0.0)

        self._log_std_bound = (-10.0, 2.0)
        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

        self.use_base_actor = False
        self.autoencoder_cfg = autoencoder_cfg
        if autoencoder_cfg is not None:
            self.training_mode = autoencoder_cfg["train"]["train_mode"]
        else:
            self.training_mode = None

    def make_base_actor(self, num_actor_obs, num_actions, actor_hidden_dims=[256, 256, 256]):
        actor_layers = []
        activation = torch.nn.SiLU()
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[layer_index], num_actions * 2)
                )
            else:
                actor_layers.append(
                    nn.Linear(
                        actor_hidden_dims[layer_index],
                        actor_hidden_dims[layer_index + 1],
                    )
                )
                actor_layers.append(activation)
        self.base_actor = nn.Sequential(*actor_layers)
        self.base_actor.to(self.actor[0].weight.device)
        self.base_actor.requires_grad_(False)
        self.base_actor.eval()
        self.base_observation_dim = num_actor_obs
        self.use_base_actor = True

    def get_film_stats(self):
        """Returns dictionary of FiLM layer statistics"""
        stats = {}
        for i, layer in enumerate(self.film_layers):
            stats[f'film_{i}_gamma'] = layer.gamma_mean.item()
            stats[f'film_{i}_beta'] = layer.beta_mean.item()
        return stats
    
    def reset_film_stats(self):
        """Reset all FiLM layer statistics"""
        for layer in self.film_layers:
            layer.reset_stats()

    def actor_forward(self, observations, z):
        h = observations
        layer_idx = 0
        film_idx = 0
        
        # Process layers with FiLM
        while layer_idx < len(self.actor) - 1:
            # Linear layer
            h = self.actor[layer_idx](h)
            layer_idx += 1
            
            # Activation
            if isinstance(self.actor[layer_idx], nn.SiLU):
                h = self.actor[layer_idx](h)
                layer_idx += 1
            
            # Apply FiLM if available
            if film_idx < len(self.film_layers):
                h = self.film_layers[film_idx](h, z)
                film_idx += 1

        return self.actor[-1](h)
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, z=None):
        # compute mean
        if z is not None:
            mean, log_std = self.actor_forward(observations, z).chunk(2, dim=-1)
        else:
            mean, log_std = self.actor(observations).chunk(2, dim=-1)
        # mean = self.actor(observations)
        # compute standard deviation
        # if self.noise_std_type == "scalar":
        #     std = self.std.expand_as(mean)
        # elif self.noise_std_type == "log":
        log_std = soft_clamp(log_std, self._log_std_bound[0], self._log_std_bound[1])
        std = torch.exp(log_std)  # .expand_as(mean)
        if torch.isnan(mean).any() or torch.isnan(std).any():
            print("NaN detected in mean/std before distribution creation:")
            print(f"Mean: {mean}")
            print(f"log_std: {log_std}")
            print(f"std: {std}")
            raise RuntimeError("NaN detected in mean/std")

        # else:
        #     raise ValueError(
        #         f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
        #     )
        # create distribution
        if self.use_tan_normal:
            self.distribution = TanhNormal(mean, std)
        else:
            self.distribution = Normal(mean, std)

    def act(self, observations, z=None, **kwargs):
        self.update_distribution(observations, z)
        if z is not None:
            action_pi = self.distribution.sample()
            # action_pi = action_pi.clamp(-1.0, 1.0)
            if self.use_base_actor:
                action_base, _ = self.base_actor(observations[:, :self.base_observation_dim]).chunk(2, dim=-1)
                action_real = action_pi + action_base
            else:
                action_real = action_pi
            return action_pi, action_real
        else:
            return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, z=None):
        if z is not None:
            mean, _ = self.actor_forward(observations, z).chunk(2, dim=-1)
        else:
            mean, _ = self.actor(observations).chunk(2, dim=-1)
        if self.use_base_actor:
            action_base, _ = self.base_actor(observations[:, :self.base_observation_dim]).chunk(2, dim=-1)
            mean = mean + action_base
        return mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


class ActorCriticRMA(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        stack_frames,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        autoencoder_cfg=None,
        **kwargs,
    ):
        super().__init__()
        activation = torch.nn.SiLU()

        # Store latent dimension
        if autoencoder_cfg is not None:
            latent_dim = autoencoder_cfg["model"]["n_embd"]
            self.latent_dim = latent_dim

        # Modified Actor with FiLM layers
        self.actor_layers = []
        print(f"Actor observation size: {num_actor_obs} with latent dim {self.latent_dim}")
        # Input layer
        self.actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        self.actor_layers.append(activation)
        
        # Subsequent layers
        for i in range(1, len(actor_hidden_dims)):
            self.actor_layers.append(nn.Linear(actor_hidden_dims[i-1], actor_hidden_dims[i]))
            self.actor_layers.append(activation)
 
        # Output layer
        self.actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions * 2))
        self.actor = nn.Sequential(*self.actor_layers)
        
        # Critic remains unchanged
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(
                    critic_hidden_dims[layer_index],
                    critic_hidden_dims[layer_index + 1],
                ))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        self.stack_frames = stack_frames
        self.adaptation_module = AdaptationModule((num_actor_obs - self.latent_dim) // self.stack_frames, num_actions, self.stack_frames, self.latent_dim)

        self._log_std_bound = (-10.0, 2.0)
        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

        self.use_base_actor = False
        self.base_latent = None
        self.autoencoder_cfg = autoencoder_cfg

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean, log_std = self.actor(observations).chunk(2, dim=-1)

        log_std = soft_clamp(log_std, self._log_std_bound[0], self._log_std_bound[1])
        std = torch.exp(log_std)  # .expand_as(mean)
        if torch.isnan(mean).any() or torch.isnan(std).any():
            print("NaN detected in mean/std before distribution creation:")
            print(f"Mean: {mean}")
            print(f"log_std: {log_std}")
            print(f"std: {std}")
            raise RuntimeError("NaN detected in mean/std")

        self.distribution = Normal(mean, std)

    def act(self, observations, actions=None, **kwargs):
        self.update_distribution(observations)
        action_pi = self.distribution.sample()
        # action_pi = action_pi.clamp(-1.0, 1.0)
        action_real = action_pi
        return action_pi, action_real

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, actions):
        mean, _ = self.actor(observations).chunk(2, dim=-1)
        return mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    def get_latent(self, observations, actions):
        # Concatenate observations and actions
        observations = observations.reshape(-1, self.stack_frames, observations.shape[-1] // self.stack_frames)
        state_actions = torch.cat((observations, actions), dim=-1)
        latent = self.adaptation_module(state_actions)
        if self.base_latent is not None:
            latent = 0.1 * latent + self.base_latent[None, :]
        return latent


class AdaptationModule(nn.Module):
    """Adaptation module used by RMA"""
    def __init__(self, state_dim, action_dim, stack_frame, extrinsic_dim):
        super(AdaptationModule, self).__init__()
        latent_dim = 256
        print(f"Adaptation module extrinsic dim {extrinsic_dim} with latent dim {latent_dim}")
        self.channel_transform = nn.Sequential(
            nn.Linear(state_dim + action_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.kernel_sizes = [5, 3, 3]
        self.strides = [1, 1, 1]
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, (self.kernel_sizes[0],), stride=(self.strides[0],)),
            nn.ReLU(inplace=True),
            nn.Conv1d(latent_dim, latent_dim, (self.kernel_sizes[1],), stride=(self.strides[1],)),
            nn.ReLU(inplace=True),
            nn.Conv1d(latent_dim, latent_dim, (self.kernel_sizes[2],), stride=(self.strides[2],)),
            nn.ReLU(inplace=True),
        )
        self.conv_out_size = stack_frame
        for i in range(len(self.kernel_sizes)):
            self.conv_out_size = (self.conv_out_size - self.kernel_sizes[i]) // self.strides[i] + 1
        self.low_dim_proj = nn.Linear(latent_dim * self.conv_out_size, extrinsic_dim)
        self.low_dim_proj.weight.data.fill_(0.0)
        self.low_dim_proj.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.channel_transform(x)  # (N, 50, 32)
        x = x.permute((0, 2, 1))  # (N, 32, 50)
        x = self.temporal_aggregation(x)  # (N, 32, 3)
        x = self.low_dim_proj(x.flatten(1))
        return x


if __name__ == "__main__":
    # Example usage:
    # Assuming xt_k_t-1 is of shape (batch_size, k, state_dim) and at_k_t-1 is of shape (batch_size, k, action_dim)
    state_dim = 83  # Example state dimension
    action_dim = 12  # Example action dimension
    extrinsic_dim = 1024  # Example extrinsic vector dimension


    # Dummy input (batch_size, k, state_dim) and (batch_size, k, action_dim)
    batch_size = 1000
    k = 15  # Number of previous steps considered (e.g., 50 corresponds to 0.5 seconds)

    # Create the adaptation module
    adaptation_module = AdaptationModule(state_dim, action_dim, k, extrinsic_dim)

    states = torch.randn(batch_size, k, state_dim)
    actions = torch.randn(batch_size, k, action_dim)

    # Forward pass through the adaptation module
    state_actions = torch.cat((states, actions), dim=-1)  # Concatenate along the last dimension
    print(state_actions.shape)  # Should print (batch_size, k, state_dim + action_dim)
    z_hat = adaptation_module(state_actions)

    print(z_hat.shape)  # Should print (batch_size, k, extrinsic_dim)