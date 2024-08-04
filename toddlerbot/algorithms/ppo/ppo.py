#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage


class PPO:
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic: ActorCritic,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        device: str = "cpu",
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage: RolloutStorage = None  # type: ignore
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: List[int],
        critic_obs_shape: List[int],
        action_shape: List[int],
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()  # type: ignore

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()  # type: ignore
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()  # type: ignore
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(  # type: ignore
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()  # type: ignore
        self.transition.action_sigma = self.actor_critic.action_std.detach()  # type: ignore
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs  # type: ignore
        self.transition.critic_observations = critic_obs  # type: ignore
        return self.transition.actions

    def process_env_step(
        self, rewards: torch.Tensor, dones: Any, infos: Dict[str, Any]
    ):
        self.transition.rewards = rewards.clone()  # type: ignore
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(  # type: ignore
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs: torch.Tensor):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            self.actor_critic.act(
                obs_batch,
                masks=masks_batch,  # type: ignore
                hidden_states=hid_states_batch[0],  # type: ignore
            )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch,
                masks=masks_batch,  # type: ignore
                hidden_states=hid_states_batch[1],  # type: ignore
            )
            mu_batch = self.actor_critic.action_mean  # type: ignore
            sigma_batch = self.actor_critic.action_std  # type: ignore
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(  # type: ignore
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)  # type: ignore
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)  # type: ignore
                        )
                        / (2.0 * torch.square(sigma_batch))  # type: ignore
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)  # type: ignore

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:  # type: ignore
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()  # type: ignore

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()  # type: ignore
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()  # type: ignore
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)  # type: ignore
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss
