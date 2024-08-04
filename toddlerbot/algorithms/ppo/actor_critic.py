#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: List[int] = [256, 256, 256],
        critic_hidden_dims: List[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        **kwargs: Dict[str, Any],
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation_layer = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers: List[nn.Module] = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation_layer)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[layer_index], num_actions)
                )
            else:
                actor_layers.append(
                    nn.Linear(
                        actor_hidden_dims[layer_index],
                        actor_hidden_dims[layer_index + 1],
                    )
                )
                actor_layers.append(activation_layer)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers: List[nn.Module] = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation_layer)
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
                critic_layers.append(activation_layer)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False  # type: ignore

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential: List[nn.Module], scales: List[int]):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])  # type: ignore
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    def reset(self, dones: Any = None):
        pass

    def forward(self):
        pass

    @property
    def action_mean(self):  # type: ignore
        return self.distribution.mean  # type: ignore

    @property
    def action_std(self):  # type: ignore
        return self.distribution.stddev  # type: ignore

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)  # type: ignore

    def update_distribution(self, observations: torch.Tensor):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations: torch.Tensor, **kwargs: Dict[str, Any]):
        self.update_distribution(observations)
        return self.distribution.sample()  # type: ignore

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)  # type: ignore

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(
        self, critic_observations: torch.Tensor, **kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        value = self.critic(critic_observations)
        return value


def get_activation(act_name: str):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError("invalid activation function!")
