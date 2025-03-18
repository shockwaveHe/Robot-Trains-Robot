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
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = torch.nn.SiLU()  # TODO: Remove the hardcode

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
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
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(
                    nn.Linear(
                        critic_hidden_dims[layer_index],
                        critic_hidden_dims[layer_index + 1],
                    )
                )
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        # self.noise_std_type = noise_std_type
        self._log_std_bound = (-10.0, 2.0)
        # if self.noise_std_type == "scalar":
        #     self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        # elif self.noise_std_type == "log":
        #     self.log_std = nn.Parameter(
        #         torch.log(init_noise_std * torch.ones(num_actions))
        #     )
        # else:
        #     raise ValueError(
        #         f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
        #     )

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

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
        # compute mean
        mean, log_std = self.actor(observations).chunk(2, dim=-1)
        # mean = self.actor(observations)
        # compute standard deviation
        # if self.noise_std_type == "scalar":
        #     std = self.std.expand_as(mean)
        # elif self.noise_std_type == "log":
        log_std = soft_clamp(log_std, self._log_std_bound[0], self._log_std_bound[1])
        std = torch.exp(log_std)  # .expand_as(mean)
        # else:
        #     raise ValueError(
        #         f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
        #     )
        # create distribution
        self.distribution = TanhNormal(mean, std)
        # self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        actions_clip = torch.clamp(actions, -1 + 1e-6, 1 - 1e-6)
        return self.distribution.log_prob(actions_clip).sum(dim=-1)

    def act_inference(self, observations):
        mean, _ = self.actor(observations).chunk(2, dim=-1)
        return mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
