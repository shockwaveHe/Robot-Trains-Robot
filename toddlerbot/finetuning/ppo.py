import os
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tqdm import tqdm

from toddlerbot.finetuning.finetune_config import FinetuneConfig
from toddlerbot.finetuning.logger import FinetuneLogger
from toddlerbot.finetuning.networks import (
    GaussianPolicyNetwork,
    ValueNetwork,
)
from toddlerbot.finetuning.replay_buffer import OnlineReplayBuffer
from toddlerbot.finetuning.utils import CONST_EPS


class PPO:
    def __init__(
        self,
        device: torch.device,
        config: FinetuneConfig,
        policy_net: GaussianPolicyNetwork,
        value_net: ValueNetwork,
        logger: FinetuneLogger,
        base_policy_net: Optional[GaussianPolicyNetwork] = None,
        use_latent: bool = False,
        optimize_z: bool = False,
        optimize_critic: bool = False,
        autoencoder_cfg: Optional[dict] = None,
    ) -> None:
        self.batch_size = config.online.batch_size
        self.mini_batch_size = config.online.mini_batch_size
        self.max_train_step = config.online.max_train_step
        self.lr_a = config.online.lr_a  # Learning rate of actor
        self.lr_c = config.online.lr_c  # Learning rate of critic
        self.gamma = config.online.gamma  # Discount factor
        self.lamda = config.online.lamda  # GAE parameter
        self.epsilon = config.online.epsilon  # PPO clip parameter
        self.K_epochs = config.online.K_epochs  # PPO parameter
        self.entropy_coef = config.online.entropy_coef  # Entropy coefficient
        self.set_adam_eps = config.online.set_adam_eps
        self.use_grad_clip = config.online.use_grad_clip
        self.use_lr_decay = config.online.use_lr_decay
        self.use_adv_norm = config.online.use_adv_norm
        self.is_clip_value = config.online.is_clip_value
        self.device = device
        self.use_latent = use_latent
        self.optimize_z = optimize_z
        self.optimizer_critic = optimize_critic
        self.autoencoder_cfg = autoencoder_cfg
        self.exp_type = config.exp_type

        self._config = config
        self._device = device
        self._logger = logger  # TODO: improve logging, seperate online offline?

        self._policy_net = deepcopy(policy_net).to(
            self.device
        )  # deepcopy to keep policy on inference device
        self._base_policy_net = (
            deepcopy(base_policy_net).to(self.device)
            if base_policy_net is not None
            else None
        )
        # if args.scale_strategy == 'dynamic' or args.scale_strategy == 'number': # DISCUSS
        #     self.critic = ValueReluMLP(args).to(self.device)
        # else:
        self._value_net = value_net.to(self.device)

        # register the optimizable latents
        # Create a learnable parameter tensor initialized with initial_latents

        self.latent_z = None
        self.latent_optimizer = None
        self.optimizer_actor = None
        self.optimizer_critic = None
        if self.exp_type == "walk":
            if use_latent:
                with open("toddlerbot/finetuning/latent_z_release.pt", "rb") as f:
                    initial_latents = torch.load(f)
                    if type(initial_latents) == dict:
                        initial_latents = initial_latents["latent_z"]
                    initial_latents = initial_latents.to(self.device)

                self.latent_z = nn.Parameter(
                    initial_latents.clone(), requires_grad=True
                )

            if optimize_z:
                # self.all_latents = self.latent_z.detach()
                # Create an optimizer for the latent parameters
                # Total number of training steps for decay
                initial_lr = float(self.autoencoder_cfg["train"].get("latent_lr"))
                decay_factor = float(self.autoencoder_cfg["train"].get("latent_decay"))
                decay_steps = self.autoencoder_cfg["train"].get("latent_steps")
                step_size = self.autoencoder_cfg["train"].get("latent_step_size")

                def stepwise_decay(step):
                    decay_count = min(step // step_size, decay_steps // step_size)
                    return max(decay_factor, 1.0 - decay_factor * decay_count)

                self.latent_optimizer = torch.optim.AdamW(
                    [self.latent_z], lr=initial_lr
                )
                self.latent_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.latent_optimizer, stepwise_decay
                )
            else:
                self.optimizer_actor = torch.optim.Adam(
                    self._policy_net.parameters(),
                    lr=self.lr_a,
                    eps=1e-5 if self.set_adam_eps else 1e-8,
                )

            if optimize_critic:
                self.optimizer_critic = torch.optim.Adam(
                    self._value_net.parameters(),
                    lr=self.lr_c,
                    eps=1e-5 if self.set_adam_eps else 1e-8,
                )
        elif self.exp_type == "swing":
            if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
                self.optimizer_actor = torch.optim.Adam(
                    self._policy_net.parameters(), lr=self.lr_a, eps=1e-5
                )
                self.optimizer_critic = torch.optim.Adam(
                    self._value_net.parameters(), lr=self.lr_c, eps=1e-5
                )
            else:
                self.optimizer_actor = torch.optim.Adam(
                    self._policy_net.parameters(), lr=self.lr_a
                )
                self.optimizer_critic = torch.optim.Adam(
                    self._value_net.parameters(), lr=self.lr_c
                )

        self.beta = config.beta
        if self.beta > 0.0:
            from pink.cnrl import ColoredNoiseProcess

            self.cnp = ColoredNoiseProcess(
                beta=self.beta,
                size=(policy_net.action_size, config.episode_length),
                rng=np.random.default_rng(),
            )

    def evaluate_value(self, replay_buffer: OnlineReplayBuffer, steps=20000):
        mean_value = []
        for _ in tqdm(range(steps), desc="check buffer value"):
            _, s_p, _, _, _, _, _, _, _, Return, _ = replay_buffer.sample(512)
            value = self._value_net(s_p)
            mean_value.append(torch.mean(value.cpu().detach()).item())

        print("mean value score: {}".format(np.mean(mean_value)))

    def load(self, path: str) -> None:
        policy_path = os.path.join(path, "policy_net.pt")
        value_path = os.path.join(path, "value_net.pt")
        self._policy_net.load_state_dict(
            torch.load(policy_path, map_location=self._device)
        )
        self._value_net.load_state_dict(
            torch.load(value_path, map_location=self.device)
        )

    def set_networks(self, value_net: ValueNetwork, policy_net: GaussianPolicyNetwork):
        self._value_net.load_state_dict(value_net.state_dict())
        self._policy_net.load_state_dict(policy_net.state_dict())
        print("Successfully set networks from pretraining")

    def get_action(self, s, deterministic=False):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        with torch.no_grad():
            dist = (
                self._policy_net(s, self.latent_z)
                if self.exp_type == "walk"
                else self._policy_net(s)
            )
            if deterministic:
                if isinstance(dist, TransformedDistribution):
                    a_pi = torch.tanh(dist.base_dist.loc)
                else:
                    a_pi = dist.loc
            else:
                if self.beta == 0.0:
                    a_pi = dist.sample()
                else:
                    noise = torch.from_numpy(self.cnp.sample()).to(self.device)
                    if isinstance(dist, TransformedDistribution):
                        a_pi = dist.base_dist.mean + dist.base_dist.stddev * noise
                        for transform in dist.transforms:
                            a_pi = transform(a_pi)
                    else:
                        a_pi = dist.mean + dist.stddev * noise

            a_logprob = dist.log_prob(a_pi).sum(axis=-1, keepdim=True)
        if self.use_residual:
            base_dist = self._base_policy_net(s)
            if isinstance(base_dist, TransformedDistribution):
                a_real = a_pi + torch.tanh(base_dist.base_dist.loc)
            else:
                assert isinstance(base_dist, torch.distributions.Normal)
                a_real = a_pi + base_dist.loc
            a_real.clamp_(-1.0 + CONST_EPS, 1.0 - CONST_EPS)
        else:
            a_real = a_pi
        return (
            a_pi.cpu().numpy().flatten(),
            a_real.cpu().numpy().flatten(),
            a_logprob.cpu().numpy(),
        )

    def update(self, replay_buffer: OnlineReplayBuffer, current_steps):
        states, sp, actions, rewards, s_, sp_, _, terms, truncs, _, a_logprob_old = (
            replay_buffer.sample_all()
        )
        # reset immediately after sampling to let remote client continue collecting data
        replay_buffer.reset()
        # Compute advantage
        gae = 0
        advantage = torch.zeros_like(rewards)
        with torch.no_grad():
            values = self._value_net(
                sp
            )  # "old" value calculated at the start of the update
            next_values = self._value_net(sp_)
            deltas = (
                rewards.flatten()
                + self.gamma * (1.0 - terms.flatten()) * next_values
                - values
            )
            for step in reversed(range(len(deltas))):
                gae = deltas[step] + self.gamma * self.lamda * gae * (
                    1.0 - terms[step]
                ) * (1.0 - truncs[step])
                advantage[step] = gae
            returns = advantage.flatten() + values
            if self.use_adv_norm:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        actor_losses, critic_losses = [], []
        pbar = tqdm(range(self.K_epochs), desc="PPO training")
        for i in pbar:
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(
                SubsetRandomSampler(range(len(states))), self.mini_batch_size, False
            ):
                new_dist = (
                    self._policy_net(states[index], self.latent_z)
                    if self.exp_type == "walk"
                    else self._policy_net(states[index])
                )
                # dist_entropy = new_dist.base_dist.entropy().sum(1, keepdim=True)
                if isinstance(new_dist, TransformedDistribution):
                    pre_tanh_sample = new_dist.transforms[-1].inv(new_dist.rsample())
                    log_det_jac = self._policy_net.forward_log_det_jacobian(
                        pre_tanh_sample
                    )  # return 2.0 * (torch.log(torch.tensor(2.0)) - x - F.softplus(-2.0 * x))
                    dist_entropy = torch.sum(
                        new_dist.base_dist.entropy() + log_det_jac, dim=-1, keepdim=True
                    )
                else:
                    assert isinstance(new_dist, torch.distributions.Normal)
                    dist_entropy = new_dist.entropy().sum(1, keepdim=True)
                a_logprob_now = new_dist.log_prob(actions[index])

                ratios = torch.exp(
                    a_logprob_now.sum(1, keepdim=True) - a_logprob_old[index]
                )
                surr1 = ratios * advantage[index]
                ratios_clipped = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
                surr2 = ratios_clipped * advantage[index]
                actor_loss = (
                    -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                ).mean()
                actor_losses.append(actor_loss.item())

                if self.optimizer_actor:
                    self.optimizer_actor.zero_grad()
                    actor_loss.backward()
                    if self.use_grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            self._policy_net.parameters(), 1.0
                        )
                    self.optimizer_actor.step()

                if self.latent_optimizer:
                    # Update the latent parameters
                    self.latent_optimizer.zero_grad()
                    if not self.optimizer_actor:
                        actor_loss.backward()
                    if self.use_grad_clip:
                        torch.nn.utils.clip_grad_norm_([self.latent_z], 1.0)

                    self.latent_optimizer.step()
                    self.latent_lr_scheduler.step()

                # Critic loss
                current_values = self._value_net(
                    sp[index]
                )  # "new" value calculated after the actor update
                if self.is_clip_value:
                    old_value_clipped = values[index] + (
                        current_values - values[index]
                    ).clamp(-self.epsilon, self.epsilon)
                    value_loss = (current_values - returns[index].detach().float()).pow(
                        2
                    )
                    value_loss_clipped = (
                        old_value_clipped - returns[index].detach().float()
                    ).pow(2)
                    critic_loss = torch.max(value_loss, value_loss_clipped).mean()
                else:
                    critic_loss = F.mse_loss(returns[index], current_values).mean()
                critic_losses.append(critic_loss.item())

                if self.optimizer_critic:
                    self.optimizer_critic.zero_grad()
                    critic_loss.backward()
                    if self.use_grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            self._value_net.parameters(), 1.0
                        )

                    self.optimizer_critic.step()

                clamped_mask = (ratios < 1 - self.epsilon) | (ratios > 1 + self.epsilon)
                clamped_fraction = clamped_mask.sum().item() / self.mini_batch_size
                self._logger.log_update(
                    a_logprob_now=a_logprob_old[index].mean().item(),
                    ratios=ratios.mean().item(),
                    ratios_clipped=ratios_clipped.mean().item(),
                    clamped_fraction=clamped_fraction,
                    adv=advantage[index].mean().item(),
                    current_values=current_values.mean().item(),
                    actor_loss=actor_loss.item(),
                    dist_entropy=dist_entropy.mean().item(),
                    critic_loss=critic_loss.item(),
                    actor_lr=self.optimizer_actor.param_groups[0]["lr"]
                    if self.optimizer_actor
                    else None,
                    critic_lr=self.optimizer_critic.param_groups[0]["lr"]
                    if self.optimizer_critic
                    else None,
                    latent_lr=self.latent_optimizer.param_groups[0]["lr"]
                    if self.latent_optimizer
                    else None,
                    latent_z=self.get_latent(),
                )
                pbar.set_description(
                    f"PPO training step {i} (actor_loss: {actor_loss.item()}, critic_loss: {critic_loss.item()})"
                )
        if self.use_lr_decay:
            self.lr_decay(current_steps)
        return np.mean(actor_losses), np.mean(critic_losses)

    def lr_decay(self, current_steps):
        # TODO: decay by max train steps or train steps per iteration
        lr_a_now = self.lr_a * (1 - current_steps / self.max_train_step)
        lr_c_now = self.lr_c * (1 - current_steps / self.max_train_step)
        if self.optimizer_actor:
            for p in self.optimizer_actor.param_groups:
                p["lr"] = lr_a_now

        if self.optimizer_critic:
            for p in self.optimizer_critic.param_groups:
                p["lr"] = lr_c_now

    def get_latent(self):
        if self.latent_z is None:
            return None

        return self.latent_z.detach().cpu().clone()
