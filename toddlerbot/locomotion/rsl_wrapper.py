import io
import os
import queue
import threading
import time
from typing import Optional

import jax
import lz4.frame
import torch
from brax.io.torch import jax_to_torch, torch_to_jax
from rsl_rl.env.vec_env import VecEnv
from dotmap import DotMap
from toddlerbot.locomotion.mjx_env import MJXEnv
from toddlerbot.locomotion.ppo_config import PPOConfig

# from toddlerbot.utils.math_utils import soft_clamp


class AsyncLogger:
    def __init__(self, log_dir, flush_interval=50):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.data_queue = queue.Queue(maxsize=1000)
        self.flush_interval = flush_interval
        self.buffer = []
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def log(self, data):
        self.data_queue.put(data)

    def _worker(self):
        while True:
            try:
                data = self.data_queue.get(timeout=1.0)
                self.buffer.append(data)

                if len(self.buffer) >= self.flush_interval:
                    self.flush()
            except queue.Empty:
                if self.buffer:
                    self.flush()

    def flush(self):
        timestamp = int(time.time() * 1000)
        filename = os.path.join(self.log_dir, f"log_{timestamp}.pt.lz4")

        buffer = io.BytesIO()
        torch.save(self.buffer, buffer)
        compressed = lz4.frame.compress(buffer.getvalue())

        with open(filename, "wb") as f:
            f.write(compressed)

        self.buffer.clear()

    def close(self):
        self.flush()


class RSLWrapper(VecEnv):
    def __init__(
        self,
        env: MJXEnv,
        device: torch.device,
        exp_folder_path: str = "",
        train_cfg: Optional[PPOConfig] = None,
        num_envs: int = 1,
    ):
        self.env = env
        self.device = device
        self.num_actions = env.action_size
        self.cfg = env.cfg

        self.num_obs = env.obs_size

        if train_cfg is None:
            self.num_envs = 1
            self.max_episode_length = 1000
            self.key_envs = jax.random.PRNGKey(0)
        else:
            self.num_envs = num_envs
            self.max_episode_length = train_cfg.episode_length
            key = jax.random.PRNGKey(train_cfg.seed)
            self.key_envs = jax.random.split(key, self.num_envs)

        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=device
        )

        self.reset_fn = jax.jit(env.reset)
        self.step_fn = jax.jit(env.step)

        self.last_state = self.reset()

        # Initialize async logger
        if exp_folder_path:
            self.logger = AsyncLogger(os.path.join(exp_folder_path, "transition_data"))
        else:
            self.logger = None

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        obs = self.last_state.obs
        privileged_obs = self.last_state.privileged_obs
        obs_torch = jax_to_torch(obs, device=self.device)
        privileged_obs_torch = jax_to_torch(privileged_obs, device=self.device)
        if len(obs_torch.shape) == 1:
            obs_torch = obs_torch.unsqueeze(0)
            privileged_obs_torch = privileged_obs_torch.unsqueeze(0)

        return obs_torch, {"observations": {"critic": privileged_obs_torch}}

    def reset(self):
        self.last_state = self.reset_fn(self.key_envs)
        return self.last_state

    def render(self, trajectory, **kwargs):
        return self.env.render(trajectory, **kwargs)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        actions_jax = torch_to_jax(actions)
        state_curr = self.step_fn(self.last_state, actions_jax)
        obs = jax_to_torch(state_curr.obs, device=self.device)
        privileged_obs = jax_to_torch(state_curr.privileged_obs, device=self.device)
        rewards = jax_to_torch(state_curr.reward, device=self.device)
        dones = jax_to_torch(state_curr.done, device=self.device)
        infos = {
            "observations": {"critic": privileged_obs},
            "log": jax_to_torch(state_curr.metrics, device=self.device),
            "info": state_curr.info,
            "pipeline_state": DotMap({
                "q": state_curr.pipeline_state.q,
                "qd": state_curr.pipeline_state.qd,
            }),
        }
        self.last_state = state_curr

        if self.logger is not None:
            # Efficiently log data asynchronously
            log_entry = {
                "obs": obs[..., : self.num_obs].cpu(),
                "actions": actions.cpu(),
                "rewards": rewards.cpu(),
                "dones": dones.cpu(),
            }
            # for key, value in log_entry.items():
            #     print(f"{key}: {value.shape}")

            self.logger.log(log_entry)

        return obs, rewards, dones, infos

    def close(self):
        self.logger.close()
