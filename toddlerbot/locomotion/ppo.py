# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import os
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.utils import string_to_callable
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.collections import LineCollection
from toddlerbot.autoencoder.dataset import HyperparameterDataset
from toddlerbot.autoencoder.network import EncoderDecoder
from toddlerbot.locomotion.actor_critic import ActorCritic
from toddlerbot.locomotion.rollout_storage import RolloutStorage


class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""
    """Adapted from the implementation in rsl_rl."""

    actor_critic: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        actor_critic,
        optimize_z: bool=False,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        autoencoder_loss_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        autoencoder_cfg: dict | None = None,
    ):
        self.device = device
        self.optimize_z = optimize_z
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.current_ratio = None
        # RND components
        if rnd_cfg is not None:
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(
                params, lr=rnd_cfg.get("learning_rate", 1e-3)
            )
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = (
                symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            )
            # Print that we are not using symmetry
            if not use_symmetry:
                warnings.warn(
                    "Symmetry not used for learning. We will use it for logging instead."
                )
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(
                    symmetry_cfg["data_augmentation_func"]
                )
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(
                symmetry_cfg["data_augmentation_func"]
            ):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        if autoencoder_cfg is not None:
            (
                self.autoencoder,
                self.autoencoder_loss_func,
                self.train_params_dataloader,
                self.eval_params_dataloader,
                self.latent_dim,
            ) = self.build_autoencoder(autoencoder_cfg)
            self.optimize_autoencoder = autoencoder_cfg["train"].get("optimize_autoencoder", True) and not self.optimize_z # no need to optimize encoder when optimize z
            if self.optimize_autoencoder:
                params = self.autoencoder.parameters()
                self.autoencoder_lr = float(autoencoder_cfg["train"].get("autoencoder_lr", self.learning_rate))
                self.autoencoder_lr_ratio = self.autoencoder_lr / self.learning_rate
                self.autoencoder_optimizer = optim.AdamW(params, lr=self.autoencoder_lr, weight_decay=5e-3)
                self.autoencoder_loss_coef = float(autoencoder_cfg["train"].get("autoencoder_loss_coef", autoencoder_loss_coef))
            else:
                self.autoencoder_optimizer = None
                self.autoencoder_loss_coef = 0.0
            self.autoencoder_cfg = autoencoder_cfg
            self.latent_mode = autoencoder_cfg["train"]["latent_mode"] # concat or FiLM
            self.film_lr = float(autoencoder_cfg["train"].get("film_lr", self.learning_rate))
            self.film_lr_ratio = self.film_lr / self.learning_rate
            self.adapt_all_lr = autoencoder_cfg["train"].get("adapt_all_lr", False)
            self.optimize_single_z = self.autoencoder_cfg["train"]["optimize_single_z"]
        else:
            self.autoencoder = None
            self.autoencoder_loss_coef = 0.0
            self.train_params_dataloader = None
            self.eval_params_dataloader = None
            self.latent_dim = None
            self.autoencoder_optimizer = None
            self.latent_mode = None
            self.adapt_all_lr = False
            self.optimize_single_z = False

        self.latent_z = None
        self.latent_optimizer = None
        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        # Create optimizer
        self.build_optimizer()
        # from muon import Muon
        # muon_params = [p for p in self.actor_critic.parameters() if p.ndim >= 2]
        # # Find everything else -- these should be optimized by AdamW
        # adamw_params = [p for p in self.actor_critic.parameters() if p.ndim < 2]
        # # Create the optimizer
        # self.optimizers = [Muon(muon_params, lr=0.02, momentum=0.95, rank=0, world_size=1), torch.optim.AdamW(adamw_params)]
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
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
        self.num_z_envs = 0
        self.all_latents = None
        self.latent_trajectory = {}
        if self.autoencoder is not None:
            self.all_latents, _ = self.get_latents()
            self.all_latents = self.all_latents.detach()

    def build_optimizer(self):
        """Configure trainable parameters based on film_training_mode"""
        if self.autoencoder is None:
            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
            self.film_optimizer = None

        elif self.optimize_z:
            params = []
            if self.autoencoder_cfg["train"]["cotrain_critic"]:
                for param in self.actor_critic.critic.parameters():
                    if param.requires_grad:
                        params.append(param)
            if self.autoencoder_cfg["train"]["cotrain_actors"]:
                for param in self.actor_critic.actor.parameters():
                    if param.requires_grad:
                        params.append(param)
            if len(params):
                self.optimizer = optim.Adam(params, lr=self.learning_rate)
            else:
                self.optimizer = None
            self.film_optimizer = None
        
        elif self.actor_critic.training_mode == "co-train":
            # FiLM learning rate is 1e-5, while others are 1e-4
            film_params = []
            for film_layer in self.actor_critic.film_layers:
                film_params += list(film_layer.parameters())
            # Add other parameters
            other_params = []
            for param in self.actor_critic.actor.parameters():
                if param.requires_grad:
                    other_params.append(param)
            for param in self.actor_critic.critic.parameters():
                if param.requires_grad:
                    other_params.append(param)

            self.optimizer = optim.Adam(other_params, lr=self.learning_rate)
            self.film_optimizer = optim.Adam(film_params, lr=self.film_lr)
        
        elif self.actor_critic.training_mode == "film-only":
            # Only train FiLM layers
            film_params = []
            for film_layer in self.actor_critic.film_layers:
                film_params += list(film_layer.parameters())
            # Add critic parameters
            critic_params = []
            for param in self.actor_critic.critic.parameters():
                if param.requires_grad:
                    critic_params.append(param)
            # Combine parameters
            self.optimizer = optim.Adam(critic_params, lr=self.learning_rate)
            self.film_optimizer = optim.Adam(film_params, lr=self.film_lr, weight_decay=5e-3)
            
        else:
            raise ValueError(f"Invalid film_training_mode: {self.actor_critic.film_training_mode}")

    def build_autoencoder(self, autoencoder_cfg: dict):
        train_cfg = autoencoder_cfg["train"]
        model_cfg = autoencoder_cfg["model"]
        data_cfg = autoencoder_cfg["data"]


        params_dataset = HyperparameterDataset(data_cfg)

        # normalize the parameters
        if data_cfg["normalize_params"]:
            params_dataset.normalize_params()
        
        input_splits = [params_dataset.params.shape[1]]

        autoencoder = EncoderDecoder(
            input_splits,
            model_cfg["n_embd"],
            model_cfg["encoder_depth"],
            model_cfg["decoder_depth"],
            model_cfg["input_noise_factor"],
            model_cfg["latent_noise_factor"],
            model_cfg["is_vae"],
        ).to(self.device)
        if len(autoencoder_cfg["train"].get("pretrain_model", "")) > 0: 
            encoder_ckpt = torch.load(autoencoder_cfg["train"]["pretrain_model"], map_location="cpu")
            weights_dict = {}
            weights = encoder_ckpt["state_dict"]
            for k, v in weights.items():
                new_k = k.replace("model.", "") if "model." in k else k
                weights_dict[new_k] = v
            autoencoder.load_state_dict(weights_dict)
            print("Loading encoders from {}".format(autoencoder_cfg["train"]["pretrain_model"]))
        # Instantiate the reconstruction loss function from the config
        recon_loss_func = nn.MSELoss()
        # Get the KL weight from the config if in VAE mode (default to 1.0 if not specified)
        kl_weight = train_cfg.get("kl_weight", 1.0) if model_cfg["is_vae"] else None

        def loss_func(recon, mean, logvar, target):
            recon_loss = recon_loss_func(recon, target)
            # If in VAE mode and mean/logvar are provided, add KL divergence
            if model_cfg["is_vae"] and mean is not None and logvar is not None:
                # Compute KL divergence loss
                kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
                total_loss = recon_loss + kl_weight * kl_loss
            else:
                # In AE mode, use only the reconstruction loss
                total_loss = recon_loss

            return total_loss

        latent_dim = model_cfg["n_embd"] * len(input_splits)

        return autoencoder, loss_func, params_dataset.train_dataloader, params_dataset.eval_dataloader, latent_dim

    def configure_optimize_z(self, num_envs):
        # get average latent z from the train envs
        use_eval_latents = self.autoencoder_cfg["train"].get("use_eval_latents", False)
        initial_latents, _ = self.get_latents(get_eval_latents=use_eval_latents)
        self.latent_trajectory['initial_latents'] = initial_latents.clone()
        self.latent_trajectory['optimized_latents'] = []

        initial_latents = initial_latents.mean(dim=0, keepdim=True)
        if not self.autoencoder_cfg["train"]["optimize_single_z"]:
            initial_latents = initial_latents.expand(num_envs, -1)
        self.num_z_envs = num_envs
        
        # register the optimizable latents
        # Create a learnable parameter tensor initialized with initial_latents
        self.latent_z = nn.Parameter(initial_latents.clone(), requires_grad=True)
        self.all_latents = self.latent_z.detach()
        # Create an optimizer for the latent parameters
        self.latent_lr = float(self.autoencoder_cfg["train"].get("latent_lr", self.learning_rate))
        self.latent_lr_ratio = self.latent_lr / self.learning_rate
        self.latent_optimizer = optim.AdamW([self.latent_z], lr=self.latent_lr)
        
        print(f"Initialized optimizable latents with shape: {self.latent_z.shape}")
        print(f"Using Adam optimizer with learning rate: {self.latent_lr}")
        
    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # create rollout storage
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            rnd_state_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, latents=None):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions, action_real = self.actor_critic.act(obs, latents)
        self.transition.actions, action_real = self.transition.actions.detach(), action_real.detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return action_real

    def process_env_step(self, rewards, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Obtain curiosity gates / observations from infos
            rnd_state = infos["observations"]["rnd_state"]
            # Compute the intrinsic rewards
            # note: rnd_state is the gated_state after normalization if normalization is used
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards
            # Record the curiosity gates
            self.transition.rnd_state = rnd_state.clone()

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
        )

    def get_latents(self, get_recon_loss=False, get_eval_latents=False):
        autoencoder_loss = torch.tensor(0.0).to(self.device)
        if self.optimize_z and self.latent_z is not None:
            if self.optimize_single_z and self.num_z_envs > 0:
                return self.latent_z.expand(self.num_z_envs, -1), autoencoder_loss
            return self.latent_z, autoencoder_loss
        params_dataloader = self.eval_params_dataloader if get_eval_latents else self.train_params_dataloader
        if self.autoencoder is not None:
            if self.optimize_autoencoder or self.all_latents is None or get_eval_latents:
                all_latents = []
                # recon_loss = 0.0
                for batch in params_dataloader:
                    z, mean, logvar = self.autoencoder.encode(batch)
                    z = torch.clamp(z, -1.0, 1.0)
                    if get_recon_loss and self.autoencoder_loss_coef > 0.0:
                        x_recon = self.autoencoder.decode(z)
                        # Compute reconstruction loss for this batch
                        recon_loss = self.autoencoder_loss_func(
                            x_recon, mean, logvar, batch
                        )
                        autoencoder_loss += recon_loss  
                    all_latents.append(z)
                # Concatenate all latent vectors into a single tensor
                all_latents = torch.cat(all_latents, dim=0).flatten(start_dim=1)
            else:
                all_latents = self.all_latents
        else:
            all_latents = None
        return all_latents, autoencoder_loss
    
    def get_infer_latents(self):
        # TODO: deal with this awkward logic..
        if self.optimize_z and self.latent_z is not None:
            if self.optimize_single_z:
                return self.latent_z.expand(self.num_z_envs, -1).detach().clone()
            return self.latent_z.detach().clone()
        else:
            assert self.all_latents is not None
            return self.all_latents.clone()

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None
        # -- Autoencoder loss
        if self.autoencoder:
            mean_autoencoder_loss = 0
        else:
            mean_autoencoder_loss = None

        # generator for mini batches
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        # iterate over batches
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
            rnd_state_batch,
            env_indices_batch,
        ) in generator:
            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )
            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                    is_critic=False,
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch,
                    actions=None,
                    env=self.symmetry["_env"],
                    is_critic=True,
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(
                    num_aug, 1
                )
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            if self.autoencoder is not None:
                all_latents, autoencoder_loss = self.get_latents(get_recon_loss=self.optimize_autoencoder and self.autoencoder_loss_coef > 0.0)
                all_latents = all_latents[env_indices_batch]
                if self.latent_mode == "concat":
                    obs_batch = torch.cat(
                        [obs_batch[:, : -self.latent_dim], all_latents], dim=1
                    )
                    critic_obs_batch = torch.cat(
                        [critic_obs_batch[:, : -self.latent_dim], all_latents], dim=1
                    )
                    all_latents = None

            else:
                autoencoder_loss = torch.tensor(0.0).to(self.device)
                all_latents = None

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the actor_critic with the new parameters
            # -- actor
            self.actor_critic.act(
                obs_batch, z=all_latents, masks=masks_batch, hidden_states=hid_states_batch[0]
            )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            # -- critic
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.actor_critic.action_mean[:original_batch_size]
            sigma_batch = self.actor_critic.action_std[:original_batch_size]
            entropy_batch = self.actor_critic.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.optimizer is not None:
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate
                    if self.adapt_all_lr:
                        if self.film_optimizer is not None:
                            for param_group in self.film_optimizer.param_groups:
                                param_group["lr"] = self.learning_rate * self.film_lr_ratio
                        if self.autoencoder_optimizer is not None:
                            for param_group in self.autoencoder_optimizer.param_groups:
                                param_group["lr"] = self.learning_rate * self.autoencoder_lr_ratio
                        if self.latent_optimizer is not None:
                            for param_group in self.latent_optimizer.param_groups:
                                param_group["lr"] = self.learning_rate * self.latent_lr_ratio

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            self.current_ratio = ratio.mean().item()
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + self.autoencoder_loss_coef * autoencoder_loss
            )

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch,
                        actions=None,
                        env=self.symmetry["_env"],
                        is_critic=False,
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.actor_critic.act_inference(
                    obs_batch.detach().clone(), all_latents.detach().clone()
                )

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None,
                    actions=action_mean_orig,
                    env=self.symmetry["_env"],
                    is_critic=False,
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:],
                    actions_mean_symm_batch.detach()[original_batch_size:],
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch)
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding.detach())

            # Gradient step
            # -- For PPO
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            if self.autoencoder_optimizer:
                self.autoencoder_optimizer.zero_grad()
            if self.film_optimizer:
                self.film_optimizer.zero_grad()
            if self.latent_optimizer:
                self.latent_optimizer.zero_grad()

            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

            if self.optimizer is not None:
                self.optimizer.step()
            if self.autoencoder_optimizer:
                self.autoencoder_optimizer.step()
            if self.film_optimizer:
                self.film_optimizer.step()
            if self.latent_optimizer:
                self.latent_optimizer.step()
                # self.latent_trajectory['optimized_latents'].append(self.latent_z.detach().cpu().numpy())
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            # -- Autoencoder loss
            if mean_autoencoder_loss is not None:
                mean_autoencoder_loss += autoencoder_loss.item()
            # -- Latent loss


        # -- For PPO
        self.all_latents = self.get_latents(get_recon_loss=True)[0]
        if self.all_latents is not None:
            self.all_latents = self.all_latents.detach()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- For Autoencoder
        if mean_autoencoder_loss is not None:
            mean_autoencoder_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_entropy,
            mean_rnd_loss,
            mean_symmetry_loss,
            mean_autoencoder_loss,
        )

    def visualize_latent_dynamics(self, log_dir, it, dim=2):
        """
        Visualize latent dynamics using t-SNE
        
        Args:
            initial_latents: torch.Tensor of shape (n_samples, latent_dim) - initial latent points
            latent_trajectory: torch.Tensor of shape (n_steps, latent_dim) - optimization trajectory
            output_path: str - path to save the visualization
            dim: int - 2 or 3 for 2D or 3D visualization
        """
        initial_latents = self.latent_trajectory['initial_latents']
        latent_trajectory = self.latent_trajectory['optimized_latents']
        if len(latent_trajectory) == 0:
            print("No latent trajectory to visualize.")
            return
        # Convert to numpy if they're torch tensors
        initial_latents = initial_latents.cpu().numpy()
        latent_trajectory = torch.concatenate(latent_trajectory).cpu().numpy()
        
        # Combine all points for t-SNE
        all_points = np.vstack([initial_latents, latent_trajectory])
        
        # Apply t-SNE
        tsne = TSNE(n_components=dim, random_state=42, perplexity=30)
        embedded = tsne.fit_transform(all_points)
        
        # Split back into initial points and trajectory
        n_initial = initial_latents.shape[0]
        initial_embedded = embedded[:n_initial]
        trajectory_embedded = embedded[n_initial:]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        if dim == 2:
            # Plot initial points
            plt.scatter(initial_embedded[:, 0], initial_embedded[:, 1], 
                    color='blue', alpha=0.5, label='Initial Latents')
            
            # Plot trajectory with color progression
            points = trajectory_embedded.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Create a continuous norm to map from time step to colors
            norm = plt.Normalize(0, len(trajectory_embedded))
            lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(np.arange(len(trajectory_embedded)))
            lc.set_linewidth(2)
            line = plt.gca().add_collection(lc)
            
            # Add colorbar
            plt.colorbar(line, label='Optimization Step')
            
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            
        elif dim == 3:
            ax = plt.axes(projection='3d')
            
            # Plot initial points
            ax.scatter3D(initial_embedded[:, 0], initial_embedded[:, 1], initial_embedded[:, 2],
                        color='blue', alpha=0.5, label='Initial Latents')
            
            # Plot trajectory with color progression
            sc = ax.scatter3D(trajectory_embedded[:, 0], trajectory_embedded[:, 1], trajectory_embedded[:, 2],
                            c=np.arange(len(trajectory_embedded)), cmap='viridis', 
                            label='Optimization Trajectory')
            
            # Add colorbar
            plt.colorbar(sc, label='Optimization Step')
            
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_zlabel('t-SNE 3')
        
        plt.title('Latent Space Dynamics Visualization')
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(log_dir, f"latent_dynamics_it_{it}_{dim}d.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")

