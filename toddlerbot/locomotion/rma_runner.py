# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import heapq
import os
import shutil
import statistics
import time
from collections import deque

import numpy as np

import rsl_rl
import torch
import mediapy as media
from toddlerbot.rsl_rl.rsl_rl.env import VecEnv
from toddlerbot.finetuning.utils import RunningMeanStd
from toddlerbot.rsl_rl.rsl_rl.modules import EmpiricalNormalization
from toddlerbot.rsl_rl.rsl_rl.utils import store_code_state
from toddlerbot.locomotion.actor_critic import ActorCriticRMA, AdaptationModule
from toddlerbot.locomotion.ppo_rma import PPO
import torch.optim as optim

# define the empty context
class EmptyContext:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass
class RMARunner:
    """RMA runner for training and evaluation."""

    def __init__(
        self, train_env: VecEnv, eval_env: VecEnv, train_cfg: dict, run_name: str | None = None, learning_stage: str="1", action_frame_stack=True, device="cpu"
    ):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.train_env = train_env # eval env is used for training when optimizing the latent code
        self.eval_env = eval_env
        self.learning_stage = learning_stage # phase 1
        if "autoencoder_cfg" in self.alg_cfg and not "num_train_envs" in self.alg_cfg["autoencoder_cfg"]["data"]:
            self.alg_cfg['autoencoder_cfg']['data']['num_train_envs'] = train_env.num_envs
            self.alg_cfg['autoencoder_cfg']['data']['num_eval_envs'] = eval_env.num_envs
        # resolve dimensions of observations
        obs, extras = self.train_env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs

        if "autoencoder_cfg" in self.alg_cfg:
            # if self.alg_cfg["autoencoder_cfg"]["train"]["latent_mode"] == "concat":
            latent_dim = self.alg_cfg["autoencoder_cfg"]["model"]["n_embd"] * self.alg_cfg["autoencoder_cfg"]["model"]["num_splits"]
            num_obs += latent_dim
            num_critic_obs += latent_dim
            self.policy_cfg["autoencoder_cfg"] = self.alg_cfg["autoencoder_cfg"]

        actor_critic_class = eval(self.policy_cfg.pop("class_name") + "RMA")  # ActorCriticRMA
        actor_critic: ActorCriticRMA = actor_critic_class(
            num_obs, num_critic_obs, self.train_env.num_actions, train_env.cfg.obs.frame_stack, **self.policy_cfg
        ).to(self.device)

        # resolve dimension of rnd gated state
        if self.alg_cfg.get("rnd_cfg") is not None:
            # check if rnd gated state is present
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError(
                    "Observations for they key 'rnd_state' not found in infos['observations']."
                )
            # get dimension of rnd gated state
            num_rnd_state = rnd_state.shape[1]
            # add rnd gated state to config
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
            self.alg_cfg["rnd_cfg"]["weight"] *= train_env.unwrapped.step_dt

        # if using symmetry then pass the environment config object
        if self.alg_cfg.get("symmetry_cfg") is not None:
            # this is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = train_env # TODO: what's this for


        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.normalization_type = self.cfg["normalization_type"]

        if self.normalization_type == "empirical":
            self.obs_normalizer = EmpiricalNormalization(
                shape=[num_obs], until=1.0e8
            ).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(
                shape=[num_critic_obs], until=1.0e8
            ).to(self.device)
        elif self.normalization_type == "running_mean_std":
            self.obs_normalizer = RunningMeanStd(shape=[num_obs]).to(self.device)
            self.critic_obs_normalizer = RunningMeanStd(shape=[num_critic_obs]).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(
                self.device
            )  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity().to(
                self.device
            )  # no normalization
        # init storage and model
        if action_frame_stack and self.learning_stage == "2":
            self.action_frame_stack = train_env.cfg.obs.frame_stack
            num_actions = self.train_env.num_actions * self.action_frame_stack
        else:
            self.action_frame_stack = 1
            num_actions = self.train_env.num_actions
        # init algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        self.alg: PPO = alg_class(actor_critic, optimize_z=False, device=self.device, learning_stage=self.learning_stage, action_frame_stack=self.action_frame_stack, **self.alg_cfg)
        self.alg.init_storage(
            self.train_env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [num_actions],
        )

        # Log
        self.run_name = run_name
        self.log_dir = (
            os.path.join("results", run_name) if run_name is not None else None
        )
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

        self.best_ckpts = []
        self.latest_train_reward = 0.0
        self.latest_eval_reward = 0.0

        self.eval_freq = self.cfg["eval_freq"]
        self.eval_render_freq = self.cfg["eval_render_freq"]
        self.train_render_freq = self.cfg["train_render_freq"]
        self.eval_steps_per_env = self.cfg["eval_steps_per_env"]

        self.train_action_history = torch.zeros(self.train_env.num_envs, self.alg.actor_critic.stack_frames, self.train_env.num_actions).to(self.device)
        self.eval_action_history = torch.zeros(self.eval_env.num_envs, self.alg.actor_critic.stack_frames, self.eval_env.num_actions).to(self.device)

    def set_stage_two(self):
        self.alg.learning_stage = "2"
        self.adaptor_optimizer = optim.Adam(self.alg.actor_critic.adaptation_module.parameters(), lr=self.alg.film_lr)
        self.alg.adaptor_optimizer = optim.Adam(self.alg.actor_critic.adaptation_module.parameters(), lr=self.alg.film_lr)
        self.alg.actor_critic.actor.eval()
        self.alg.actor_critic.base_latent = self.alg.get_latents()[0].mean(dim=0).detach()
        # self.alg.actor_critic.base_latent = None
        
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                self.writer.log_config(
                    self.train_env.cfg, self.cfg, self.alg_cfg, self.policy_cfg
                )
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                import wandb

                self.writer = WandbSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                wandb.run.name = self.run_name
                self.writer.log_config(
                    self.train_env.cfg, self.cfg, self.alg_cfg, self.policy_cfg
                )
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError(
                    "Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'."
                )

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.train_env.episode_length_buf = torch.randint_like(
                self.train_env.episode_length_buf, high=int(self.train_env.max_episode_length)
            )

        # start learning
        obs, extras = self.train_env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.train_env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.train_env.num_envs, dtype=torch.float, device=self.device
        )
        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(
                self.train_env.num_envs, dtype=torch.float, device=self.device
            )
            cur_ireward_sum = torch.zeros(
                self.train_env.num_envs, dtype=torch.float, device=self.device
            )

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        adaptation_loss_fn = torch.nn.MSELoss()
        all_latents = None
        if self.alg.autoencoder is not None:
            if self.learning_stage == "1" or self.action_frame_stack > 1:
                all_latents = self.alg.get_infer_latents()
            elif self.learning_stage == "2":
                all_latents = self.alg.actor_critic.get_latent(obs, self.train_action_history)
            # if self.alg.latent_mode == "concat": # must be concat for rma
            obs = torch.cat((obs, all_latents), dim=1)
            critic_obs = torch.cat((critic_obs, all_latents), dim=1)
            all_latents = None # only keep all_latents when latent_mode is film
        for it in range(start_iter, tot_iter):
            # TODO: no need to reset environments?
            start = time.time()
            # Rollout
            eval_returns, eval_returns_zero = None, None
            # only enter inference mode when learning_stage == 1
            if self.learning_stage == "1" or self.action_frame_stack > 1:
                mode_context = torch.inference_mode
            else:
                # enter empty context that does noting
                mode_context = EmptyContext
            with mode_context():
                rollout = []
                mean_adaptation_loss = []
                for _ in range(self.num_steps_per_env):

                    # Sample actions from policy
                    # Step environment

                    actions = self.alg.act(obs, critic_obs, all_latents) # shape: num_envs, num_actions
                    self.train_action_history = torch.roll(self.train_action_history, 1, 1)
                    self.train_action_history[:, 0, :] = actions
                    obs, rewards, dones, infos = self.train_env.step(
                        actions.to(self.train_env.device)
                    )
                    if rewards.abs().max() > 1e3:
                        import ipdb; ipdb.set_trace()
                    infos["log"]["action_mean"], infos["log"]["action_std"] = self.alg.actor_critic.action_mean, self.alg.actor_critic.action_std
                    if it % self.train_render_freq == self.train_render_freq - 1:
                        rollout.append(infos["pipeline_state"])
                    # Move to the agent device
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )

                    # Extract critic observations and normalize
                    if "critic" in infos["observations"]:
                        critic_obs = infos["observations"]["critic"].to(self.device)
                    else:
                        critic_obs = obs

                    if self.alg.autoencoder is not None:
                        if self.learning_stage == "1" or self.action_frame_stack > 1:
                            all_latents = self.alg.get_infer_latents()
                        elif self.learning_stage == "2" and self.action_frame_stack == 1:
                            all_latents = self.alg.actor_critic.get_latent(obs, self.train_action_history)
                            target_latents = self.alg.get_infer_latents().detach()
                            adaptation_loss = adaptation_loss_fn(all_latents, target_latents)
                            self.adaptor_optimizer.zero_grad()
                            adaptation_loss.backward()
                            self.adaptor_optimizer.step()
                            mean_adaptation_loss.append(adaptation_loss.item())
                        # if self.alg.latent_mode == "concat": mush be concat for rma
                        obs = torch.cat((obs, all_latents), dim=1)
                        critic_obs = torch.cat((critic_obs, all_latents), dim=1)
                        all_latents = None # only keep all_latents when latent_mode is film
                    # Normalize observations
                    if isinstance(self.obs_normalizer, RunningMeanStd):
                        self.obs_normalizer.update(obs)
                    obs = self.obs_normalizer(obs)
                    if isinstance(self.critic_obs_normalizer, RunningMeanStd):
                        self.critic_obs_normalizer.update(critic_obs)
                    critic_obs = self.critic_obs_normalizer(critic_obs)
                    # Process env step and store in buffer
                    if self.learning_stage == "1" or (self.learning_stage == "2" and self.action_frame_stack > 1):
                        if self.learning_stage == "2":
                            infos["action_history"] = self.train_action_history.flatten(start_dim=1)
                        self.alg.process_env_step(rewards, dones, infos)

                    # Intrinsic rewards (extracted here only for logging)!
                    intrinsic_rewards = (
                        self.alg.intrinsic_rewards if self.alg.rnd else None
                    )   

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- intrinsic and extrinsic rewards
                        if self.alg.rnd:
                            erewbuffer.extend(
                                cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist()
                            )
                            irewbuffer.extend(
                                cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist()
                            )
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0
                mean_adaptation_loss = np.mean(mean_adaptation_loss) if len(mean_adaptation_loss) > 0 else None
                if it % self.eval_freq == self.eval_freq - 1:
                    # import ipdb; ipdb.set_trace()
                    eval_returns, eval_length = self.eval(it)
                    # self.alg.visualize_latent_dynamics(self.log_dir, it)
                    # self.alg.visualize_latent_dynamics(self.log_dir, it, dim=3)
                    if self.alg.autoencoder is not None:
                        eval_returns_zero, eval_length_zero = self.eval(it, zero_z=True)
                    self.latest_eval_reward = eval_returns
                    self.save(
                        os.path.join(
                            self.log_dir,
                            f"model_{it}_tr={round(self.latest_train_reward)}_er={round(self.latest_eval_reward)}.pt",
                        )
                    )
                        
                if len(rollout) > 0:
                    print(f"Visualing data with rollout length: {np.mean(lenbuffer)} and avg reward {np.mean(rewbuffer)}")
                    renders = self.train_env.render(
                        rollout,
                        height=360,
                        width=640,
                        camera="perspective"
                )
                    os.makedirs(os.path.join(self.log_dir, "videos"), exist_ok=True)
                    video_path = os.path.join(self.log_dir, "videos", f"it_{it}_{np.mean(rewbuffer):.2f}.mp4")
                    media.write_video(
                        video_path,
                        renders,
                        fps=50,
                    )

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            # Update policy
            # Note: we keep arguments here since locals() loads them
            if self.learning_stage == "1":
                (
                    mean_value_loss,
                    mean_surrogate_loss,
                    mean_entropy,
                    mean_rnd_loss,
                    mean_symmetry_loss,
                    mean_autoencoder_loss,
                ) = self.alg.update()
                mean_adaptation_loss = None
            elif self.action_frame_stack > 1:
                (
                    mean_value_loss,
                    mean_surrogate_loss,
                    mean_entropy,
                    mean_rnd_loss,
                    mean_symmetry_loss,
                    mean_adaptation_loss,
                ) = self.alg.update()
                mean_autoencoder_loss = None
            else:
                (
                    mean_value_loss,
                    mean_surrogate_loss,
                    mean_entropy,
                    mean_rnd_loss,
                    mean_symmetry_loss,
                    mean_autoencoder_loss,
                ) = None, None, None, None, None, None
                
            latent_z = self.alg.latent_z.detach().mean(axis=0) if self.alg.latent_z is not None else None
            current_ratio = self.alg.current_ratio
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Logging info and save checkpoint
            if self.log_dir is not None:
                # Log information
                self.log(locals())

            # Clear episode infos
            ep_infos.clear()

            # Save code state
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None:
            self.save(
                os.path.join(
                    self.log_dir,
                    f"model_{self.current_learning_iteration}_r={round(self.latest_train_reward)}_er={self.latest_eval_reward}.pt",
                )
            )

    def eval(self, it, zero_z: bool = False, eval_train_env: bool = False):
        self.alg.actor_critic.eval()
        self.eval_env.reset()
        eval_obs, eval_extras = self.eval_env.get_observations()
        critic_obs = eval_extras["observations"].get("critic", eval_obs)
        eval_returns, eval_length, eval_is_done = torch.zeros(self.eval_env.num_envs).to(self.eval_env.device), torch.zeros(self.eval_env.num_envs).to(self.eval_env.device), torch.zeros(self.eval_env.num_envs).to(self.eval_env.device)
        eval_rollout = []
        self.eval_action_history = torch.zeros_like(self.eval_action_history)
        if self.alg.autoencoder is not None:
            if self.learning_stage == "1":
                self.alg.all_latents = None
                eval_latents, _ = self.alg.get_latents(get_eval_latents=not eval_train_env)
            elif self.learning_stage == "2":
                eval_latents = self.alg.actor_critic.get_latent(eval_obs, self.eval_action_history)
            if zero_z:
                eval_latents = torch.zeros_like(eval_latents)
            infer_eval_latents = eval_latents.clone().to(self.device)
        else:
            eval_latents, infer_eval_latents = None, None
        with torch.inference_mode():
            # if self.alg.autoencoder is not None and self.alg.latent_mode == "concat":
            eval_obs = torch.cat((eval_obs, eval_latents), dim=1)
            critic_obs = torch.cat((critic_obs, eval_latents), dim=1)
            infer_eval_latents = None
        # Concatenate all latent vectors into a single tensor
            for _ in range(self.eval_steps_per_env):
                eval_actions = self.alg.act(
                    eval_obs.to(self.device), critic_obs.to(self.device), infer_eval_latents
                )
                self.eval_action_history = torch.roll(self.eval_action_history, 1, 1)
                self.eval_action_history[:, 0, :] = eval_actions
                eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_env.step(
                    eval_actions.to(self.eval_env.device)
                )
                if "critic" in eval_infos["observations"]:
                    critic_obs = eval_infos["observations"]["critic"].to(self.device)
                else:
                    critic_obs = eval_obs
                if self.eval_render_freq > 0 and it % self.eval_render_freq == self.eval_render_freq - 1:
                    eval_rollout.append(eval_infos["pipeline_state"])
                # Move to the agent device
                eval_obs, eval_rewards, eval_dones = (
                    eval_obs.to(self.device),
                    eval_rewards.to(self.device),
                    eval_dones.to(self.device),
                )
                if self.learning_stage == "2":
                    eval_latents = self.alg.actor_critic.get_latent(eval_obs, self.eval_action_history)
                    if zero_z:
                        eval_latents = torch.zeros_like(eval_latents)
                # Normalize observations
                eval_obs = self.obs_normalizer(eval_obs)
                # if self.alg.autoencoder is not None and self.alg.latent_mode == "concat":
                eval_obs = torch.cat((eval_obs, eval_latents), dim=1)
                critic_obs = torch.cat((critic_obs, eval_latents), dim=1)
                
                # process returns
                eval_returns += eval_rewards * (1 - eval_is_done)
                eval_length += (1 - eval_is_done)
                eval_is_done = eval_is_done + eval_dones * (1 - eval_is_done)
                if eval_is_done.sum() == self.eval_env.num_envs:
                    break
        
        eval_returns = eval_returns.cpu().numpy().mean()
        eval_length = eval_length.cpu().numpy().mean()
        eval_is_done = eval_is_done.cpu().numpy()
        if len(eval_rollout) > 0:
            eval_renders = self.eval_env.render(eval_rollout, height=360, width=640, camera="perspective")
            os.makedirs(os.path.join(self.log_dir, "videos"), exist_ok=True)
            if zero_z:
                video_path = os.path.join(self.log_dir, "videos", f"eval_it_{it}_zero_z_{eval_returns:.2f}_{eval_length:.2f}.mp4")
            else:
                video_path = os.path.join(self.log_dir, "videos", f"eval_it_{it}_{eval_returns:.2f}_{eval_length:.2f}.mp4")
            media.write_video(video_path, eval_renders, fps=50)
            print(f"Video saved to {video_path}")
        print(f"Eval avg rollout length: {eval_length} and avg reward {eval_returns}")
        self.alg.actor_critic.train()
        return eval_returns, eval_length
    
    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.train_env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.action_std.mean()
        fps = int(
            self.num_steps_per_env
            * self.train_env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        # -- Losses
        if self.learning_stage == "1":
            self.writer.add_scalar(
                "Loss/value_function", locs["mean_value_loss"], locs["it"]
            )
            self.writer.add_scalar(
                "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
            )
            self.writer.add_scalar("Loss/entropy", locs["mean_entropy"], locs["it"])
            self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        
        if self.alg.rnd:
            self.writer.add_scalar("Loss/rnd", locs["mean_rnd_loss"], locs["it"])
        if self.alg.symmetry:
            self.writer.add_scalar(
                "Loss/symmetry", locs["mean_symmetry_loss"], locs["it"]
            )
        if self.alg.autoencoder and locs["mean_autoencoder_loss"] is not None:
            self.writer.add_scalar(
                "Loss/autoencoder", locs["mean_autoencoder_loss"], locs["it"]
            )

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            # separate logging for intrinsic and extrinsic rewards
            if self.alg.rnd:
                self.writer.add_scalar(
                    "Rnd/mean_extrinsic_reward",
                    statistics.mean(locs["erewbuffer"]),
                    locs["it"],
                )
                self.writer.add_scalar(
                    "Rnd/mean_intrinsic_reward",
                    statistics.mean(locs["irewbuffer"]),
                    locs["it"],
                )
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # everything else
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/median_reward", statistics.median(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/median_episode_length",
                statistics.median(locs["lenbuffer"]),
                locs["it"],
            )
            if locs["eval_returns"] is not None:
                print("Logged eval returns: ", locs["eval_returns"])
                self.writer.add_scalar(
                    "Eval/mean_reward", locs["eval_returns"], locs["it"]
                )
                self.writer.add_scalar(
                    "Eval/mean_length", locs["eval_length"], locs["it"]
                )
            if locs["eval_returns_zero"] is not None:
                self.writer.add_scalar(
                    "Eval/mean_reward_zero_z", locs["eval_returns_zero"], locs["it"]
                )
                self.writer.add_scalar(
                    "Eval/mean_length_zero_z", locs["eval_length_zero"], locs["it"]
                )
            if locs["latent_z"] is not None:
                self.writer.add_scalar(
                    "Latent/mean_z", locs["latent_z"].mean().item(), locs["it"]
                )
                self.writer.add_scalar(
                    "Latent/max_z", locs["latent_z"].max().item(), locs["it"]
                )
                self.writer.add_scalar(
                    "Latent/min_z", locs["latent_z"].min().item(), locs["it"]
                )

            if locs["current_ratio"] is not None:   
                self.writer.add_scalar(
                    "Train/current_ratio", locs["current_ratio"], locs["it"]
                )
            if (
                self.logger_type != "wandb"
            ):  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar(
                    "Train/mean_reward/time",
                    statistics.mean(locs["rewbuffer"]),
                    self.tot_time,
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length/time",
                    statistics.mean(locs["lenbuffer"]),
                    self.tot_time,
                )

            self.latest_train_reward = statistics.mean(locs["rewbuffer"])

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        if "mean_adaptation_loss" in locs and locs["mean_adaptation_loss"] is not None:
            self.writer.add_scalar(
                "RMA/mean_latents_loss", locs["mean_adaptation_loss"], locs["it"]
            )
        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {locs["learn_time"]:.3f}s)\n"""
                # f"""{"Value function loss:":>{pad}} {locs["mean_value_loss"]:.4f}\n"""
                # f"""{"Surrogate loss:":>{pad}} {locs["mean_surrogate_loss"]:.4f}\n"""
            )

            # -- For symmetry
            if self.alg.symmetry:
                log_string += (
                    f"""{"Symmetry loss:":>{pad}} {locs["mean_symmetry_loss"]:.4f}\n"""
                )

            log_string += (
                f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
            )

            # -- For RND
            if self.alg.rnd:
                log_string += (
                    f"""{"Mean extrinsic reward:":>{pad}} {statistics.mean(locs["erewbuffer"]):.2f}\n"""
                    f"""{"Mean intrinsic reward:":>{pad}} {statistics.mean(locs["irewbuffer"]):.2f}\n"""
                )

            log_string += f"""{"Mean total reward:":>{pad}} {statistics.mean(locs["rewbuffer"]):.2f}\n"""
            log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(locs["lenbuffer"]):.2f}\n"""
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                # f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {locs["learn_time"]:.3f}s)\n"""
                # f"""{"Value function loss:":>{pad}} {locs["mean_value_loss"]:.4f}\n"""
                # f"""{"Surrogate loss:":>{pad}} {locs["mean_surrogate_loss"]:.4f}\n"""
            )
            # -- For symmetry
            if self.alg.symmetry:
                log_string += (
                    f"""{"Symmetry loss:":>{pad}} {locs["mean_symmetry_loss"]:.4f}\n"""
                )

            log_string += (
                f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
            )

            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Total time:":>{pad}} {self.tot_time:.2f}s\n"""
            f"""{"ETA:":>{pad}} {self.tot_time / (locs["it"] - locs["start_iter"] + 1) * (locs["start_iter"] + locs["num_learning_iterations"] - locs["it"]):.1f}s\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None, track_eval=True):
        # -- Save PPO model
        
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.alg.optimizer is not None:
            saved_dict["optimizer_state_dict"] = self.alg.optimizer.state_dict()
        # -- Save RND model if used
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # -- save autoencoder model if used
        if self.alg.autoencoder and self.alg_cfg["autoencoder_cfg"]["model"]["save_finetune_model"]:
            saved_dict["autoencoder_state_dict"] = self.alg.autoencoder.state_dict()
            if self.alg.autoencoder_optimizer:
                saved_dict["autoencoder_optimizer_state_dict"] = (
                    self.alg.autoencoder_optimizer.state_dict()
                )
            if self.alg.film_optimizer is not None:
                saved_dict["film_optimizer_state_dict"] = self.alg.film_optimizer.state_dict()
        # -- Save observation normalizer if used
        if self.normalization_type != "none":
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = (
                self.critic_obs_normalizer.state_dict()
            )

        torch.save(saved_dict, path)

        if track_eval:
            # heappush only if current latest_eval_reward is not in the heap
            heapq.heappush(self.best_ckpts, (self.latest_eval_reward, path))
        else:
            heapq.heappush(self.best_ckpts, (self.latest_train_reward, path))
        if len(self.best_ckpts) > 3:
            # Remove worst checkpoint clearly
            worst_reward, worst_ckpt = heapq.heappop(self.best_ckpts)
            if os.path.exists(worst_ckpt):
                os.remove(worst_ckpt)

            best_ckpt_reward, best_ckpt_path = max(self.best_ckpts)
            best_ckpt_dest = os.path.join(self.log_dir, "model_best.pt")
            shutil.copyfile(best_ckpt_path, best_ckpt_dest)

        # Upload model to external logging service
        # if self.logger_type in ["neptune", "wandb"]:
        #     self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        # -- Load PPO model
        # pop the mismatched keys
        keys_to_pop = []
        if "model_state_dict" in loaded_dict:
            for key, value in loaded_dict["model_state_dict"].items():
                if value.shape != self.alg.actor_critic.state_dict()[key].shape:
                    keys_to_pop.append(key)
        for key in keys_to_pop:
            loaded_dict["model_state_dict"].pop(key)
            print(f"Key {key} popped from model_state_dict due to shape mismatch")
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict=False)
        # -- Load RND model if used
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        if self.alg.autoencoder  and "autoencoder_state_dict" in loaded_dict:
            self.alg.autoencoder.load_state_dict(loaded_dict["autoencoder_state_dict"])
        # -- Load observation normalizer if used
        if self.normalization_type != "none":
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(
                loaded_dict["critic_obs_norm_state_dict"]
            )
        # -- Load optimizer if used
        if load_optimizer:
            # -- PPO
            if "optimizer_state_dict" in loaded_dict:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # -- RND optimizer if used
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(
                    loaded_dict["rnd_optimizer_state_dict"]
                )
            # -- Autoencoder optimizer if used
            if self.alg.autoencoder and "autoencoder_optimizer_state_dict" in loaded_dict:
                self.alg.autoencoder_optimizer.load_state_dict(
                    loaded_dict["autoencoder_optimizer_state_dict"]
                )
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        if self.normalization_type != "none":
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(  # noqa: E731
                self.obs_normalizer(x)
            )
        return policy

    def train_mode(self):
        # -- PPO
        self.alg.actor_critic.train()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.train()
        # -- Autoencoder
        if self.alg.autoencoder:
            self.alg.autoencoder.train()
        # -- Normalization
        if self.normalization_type != "none":
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        # -- PPO
        self.alg.actor_critic.eval()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.eval()
        # -- Autoencoder
        if self.alg.autoencoder:
            self.alg.autoencoder.eval()
        # -- Normalization
        if self.normalization_type != "none":
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
