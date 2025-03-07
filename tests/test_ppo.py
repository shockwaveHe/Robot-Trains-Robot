import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.utils.data import BatchSampler, SubsetRandomSampler
import gin
from copy import deepcopy

from tqdm import tqdm
from toddlerbot.finetuning.logger import FinetuneLogger
import torch.nn.functional as F

# Assuming these are defined elsewhere in your codebase
class MLP(nn.Module):
    def __init__(self, layers, activation_fn=nn.SiLU, layer_norm=False, activate_final=False):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2 or activate_final:
                self.layers.append(activation_fn())
        if layer_norm:
            self.layers.append(nn.LayerNorm(layers[-1]))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def soft_clamp(x, low, high):
    return low + (high - low) * torch.sigmoid((x - low) / (high - low))

# Your provided classes (pasted here for completeness)
class OnlineReplayBuffer:
    def __init__(self, device: torch.device, obs_dim: int, privileged_obs_dim: int, action_dim: int, max_size: int, seed: int = 0, enlarge_when_full: int = 0):
        self._device = device
        self._dtype = np.float32
        self._obs = np.zeros((max_size, obs_dim), dtype=self._dtype)
        self._privileged_obs = np.zeros((max_size, privileged_obs_dim), dtype=self._dtype)
        self._action = np.zeros((max_size, action_dim), dtype=self._dtype)
        self._reward = np.zeros((max_size, 1), dtype=self._dtype)
        self._terminated = np.zeros((max_size, 1), dtype=self._dtype)
        self._truncated = np.zeros((max_size, 1), dtype=self._dtype)
        self._return = np.zeros((max_size, 1), dtype=self._dtype)
        self._action_logprob = np.zeros((max_size, 1), dtype=self._dtype)
        self.rng = np.random.default_rng(seed=seed)
        self.enlarge_when_full = enlarge_when_full
        self.is_overwriting = False
        self._truncated_temp = False
        self._size = 0
        self._max_size = max_size

    def __len__(self):
        return self._size

    def store(self, s: np.ndarray, s_p: np.ndarray, a: np.ndarray, r: np.ndarray, done: bool, truncated: bool, a_logprob: np.ndarray, raw_obs=None):
        self._obs[self._size] = s
        self._action[self._size] = a
        self._reward[self._size] = r
        self._privileged_obs[self._size] = s_p
        self._action_logprob[self._size] = a_logprob
        self._terminated[self._size] = done
        if self.is_overwriting:
            self._truncated[self._size] = True
            if self._size > 1:
                self._truncated[self._size - 1] = self._truncated_temp
        else:
            self._truncated[self._size] = truncated
        self._size += 1
        self._truncated_temp = truncated
        if self._size >= self._obs.shape[0]:
            if self.enlarge_when_full > 0:
                self.enlarge(self.enlarge_when_full)
            else:
                print("Buffer is full, replacing the old data")
                self.is_overwriting = True
                self._size = 0

    def reset(self):
        self._size = 0
        self._truncated_temp = False
        self.is_overwriting = False

    def sample_all(self):
        return (
            torch.FloatTensor(self._obs[:self._size-1]).to(self._device),
            torch.FloatTensor(self._privileged_obs[:self._size-1]).to(self._device),
            torch.FloatTensor(self._action[:self._size-1]).to(self._device),
            torch.FloatTensor(self._reward[:self._size-1]).to(self._device),
            torch.FloatTensor(self._obs[1:self._size]).to(self._device) if self._size < self._max_size else torch.FloatTensor(self._obs[:1]).to(self._device),
            torch.FloatTensor(self._privileged_obs[1:self._size]).to(self._device) if self._size < self._max_size else torch.FloatTensor(self._privileged_obs[:1]).to(self._device),
            torch.FloatTensor(self._action[1:self._size]).to(self._device) if self._size < self._max_size else torch.FloatTensor(self._action[:1]).to(self._device),
            torch.FloatTensor(self._terminated[:self._size-1]).to(self._device),
            torch.FloatTensor(self._truncated[:self._size-1]).to(self._device),
            torch.FloatTensor(self._return[:self._size-1]).to(self._device),
            torch.FloatTensor(self._action_logprob[:self._size-1]).to(self._device),
        )

@gin.configurable
class OnlineConfig:
    max_train_step: int = int(1e6)
    batch_size: int = 2048
    mini_batch_size: int = 128
    K_epochs: int = 10
    gamma: float = 0.99
    lamda: float = 0.95
    epsilon: float = 0.2  # Adjusted from 0.05 for less conservative updates
    entropy_coef: float = 0.05
    lr_a: float = 3e-4  # Increased from 1e-4
    lr_c: float = 3e-4  # Increased from 1e-4
    use_adv_norm: bool = True
    use_grad_clip: bool = True
    use_lr_decay: bool = True
    is_clip_value: bool = False
    set_adam_eps: bool = True

class GaussianPolicyNetwork(nn.Module):
    def __init__(self, observation_size: int, hidden_layers: tuple, action_size: int, preprocess_observations_fn, activation_fn=nn.SiLU):
        super().__init__()
        self.preprocess_observations_fn = preprocess_observations_fn
        self.mlp = MLP([observation_size] + list(hidden_layers) + [action_size * 2], activation_fn=activation_fn)
        self._log_std_bound = (-10., 2.)

    def forward(self, obs: torch.Tensor, processer_params=None):
        obs = self.preprocess_observations_fn(obs, processer_params)
        mu, log_std = self.mlp(obs).chunk(2, dim=-1)
        log_std = soft_clamp(log_std, self._log_std_bound[0], self._log_std_bound[1])
        std = log_std.exp()
        dist = Normal(mu, std)
        dist = TransformedDistribution(dist, [TanhTransform(cache_size=1)])
        return dist
    
    def forward_log_det_jacobian(self, x):
        # 2 * (log(2) - x - softplus(-2x))
        return 2.0 * (torch.log(torch.tensor(2.0)) - x - F.softplus(-2.0 * x))

class ValueNetwork(nn.Module):
    def __init__(self, observation_size, preprocess_observations_fn, hidden_layers, activation_fn=nn.SiLU):
        super().__init__()
        self.preprocess_observations_fn = preprocess_observations_fn
        self.mlp = MLP([observation_size] + list(hidden_layers) + [1], activation_fn=activation_fn)

    def forward(self, obs, processer_params=None):
        obs = self.preprocess_observations_fn(obs, processer_params)
        return self.mlp(obs).squeeze(-1)

class PPO:
    def __init__(self, device: torch.device, config: OnlineConfig, policy_net: GaussianPolicyNetwork, value_net: ValueNetwork):
        self.batch_size = config.batch_size
        self.mini_batch_size = config.mini_batch_size
        self.max_train_step = config.max_train_step
        self.lr_a = config.lr_a
        self.lr_c = config.lr_c
        self.gamma = config.gamma
        self.lamda = config.lamda
        self.epsilon = config.epsilon
        self.K_epochs = config.K_epochs
        self.entropy_coef = config.entropy_coef
        self.set_adam_eps = config.set_adam_eps
        self.use_grad_clip = config.use_grad_clip
        self.use_lr_decay = config.use_lr_decay
        self.use_adv_norm = config.use_adv_norm
        self.is_clip_value = config.is_clip_value
        self.device = device
        self._policy_net = deepcopy(policy_net).to(self.device)
        self._value_net = value_net.to(self.device)
        self.optimizer_actor = torch.optim.Adam(self._policy_net.parameters(), lr=self.lr_a, eps=1e-5 if self.set_adam_eps else 1e-8)
        self.optimizer_critic = torch.optim.Adam(self._value_net.parameters(), lr=self.lr_c, eps=1e-5 if self.set_adam_eps else 1e-8)
        self._logger = FinetuneLogger('tests/logging/ppo')

    def get_action(self, s, deterministic=False):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        with torch.no_grad():
            dist = self._policy_net(s)
            if deterministic:
                a = torch.tanh(dist.base_dist.loc)
                a_logprob = dist.log_prob(a)
            else:
                a = dist.sample()
                a_logprob = dist.log_prob(a)
        return a.cpu().numpy().flatten(), a_logprob.sum(axis=-1, keepdim=True).cpu().numpy()

    def update(self, replay_buffer: OnlineReplayBuffer, current_steps):
        states, sp, actions, rewards, s_, sp_, _, terms, truncs, _, a_logprob_old = replay_buffer.sample_all()
        gae = 0
        advantage = torch.zeros_like(rewards)
        with torch.no_grad():
            values = self._value_net(sp) # "old" value calculated at the start of the update
            next_values = self._value_net(sp_)
            deltas = rewards.flatten() + self.gamma * (1.0 - terms.flatten()) * next_values - values
            for step in reversed(range(len(deltas))):
                gae = deltas[step] + self.gamma * self.lamda * gae * (1.0 - terms[step]) * (1.0 - truncs[step])
                advantage[step] = gae
            returns = advantage.flatten() + values
            if self.use_adv_norm:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)
        actor_losses, critic_losses = [], []
        pbar = tqdm(range(self.K_epochs), desc='PPO training')
        for i in pbar:
            for index in BatchSampler(SubsetRandomSampler(range(len(states))), self.mini_batch_size, False):  # Fixed indexing
                new_dist = self._policy_net(states[index])
                # dist_entropy = new_dist.base_dist.entropy().sum(1, keepdim=True)
                if isinstance(new_dist, TransformedDistribution):
                    pre_tanh_sample = new_dist.transforms[-1].inv(new_dist.rsample())
                    log_det_jac = self._policy_net.forward_log_det_jacobian(pre_tanh_sample) # return 2.0 * (torch.log(torch.tensor(2.0)) - x - F.softplus(-2.0 * x))
                    dist_entropy = torch.sum(new_dist.base_dist.entropy() + log_det_jac, dim=-1, keepdim=True)
                else:
                    dist_entropy = new_dist.entropy().sum(1, keepdim=True)
                a_logprob_now = new_dist.log_prob(actions[index])

                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob_old[index])
                surr1 = ratios * advantage[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantage[index]
                actor_loss = (-torch.min(surr1, surr2) - self.entropy_coef * dist_entropy).mean()
                actor_losses.append(actor_loss.item())
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self._policy_net.parameters(), 0.5)
                self.optimizer_actor.step()
                current_values = self._value_net(sp[index]) # "new" value calculated after the actor update
                if self.is_clip_value:
                    old_value_clipped = values[index] + (current_values - values[index]).clamp(-self.epsilon, self.epsilon)
                    value_loss = (current_values - returns[index].detach().float()).pow(2)
                    value_loss_clipped = (old_value_clipped - returns[index].detach().float()).pow(2)
                    critic_loss = torch.max(value_loss, value_loss_clipped).mean()
                else:
                    critic_loss = nn.functional.mse_loss(returns[index], current_values).mean()
                critic_losses.append(critic_loss.item())
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self._value_net.parameters(), 0.5)
                self.optimizer_critic.step()
                self._logger.log_update(
                    a_logprob_now=a_logprob_old[index].mean().item(),
                    ratios=ratios.mean().item(),
                    adv=advantage[index].mean().item(),
                    current_values=current_values.mean().item(),
                    actor_loss=actor_loss.item(),
                    dist_entropy=dist_entropy.mean().item(),
                    critic_loss=critic_loss.item()
                )
                pbar.set_description(f'PPO training step {i} (actor_loss: {actor_loss.item()}, critic_loss: {critic_loss.item()})')
        if self.use_lr_decay:
            self.lr_decay(current_steps)
        replay_buffer.reset()
        return np.mean(actor_losses), np.mean(critic_losses)

    def lr_decay(self, current_steps):
        lr_a_now = self.lr_a * (1 - current_steps / self.max_train_step)
        lr_c_now = self.lr_c * (1 - current_steps / self.max_train_step)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now


# Training Loop
def train():
    env = gym.make("Pendulum-v1")
    print(env.action_space)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = (env.action_space.high[0] - env.action_space.low[0]) / 2  # 2.0 for Pendulum
    action_bias = (env.action_space.high[0] + env.action_space.low[0]) / 2  # 0.0 for Pendulum

    config = OnlineConfig()
    replay_buffer = OnlineReplayBuffer(device, obs_dim, obs_dim, action_dim, max_size=2 * config.batch_size)
    policy_net = GaussianPolicyNetwork(obs_dim, (64, 64), action_dim, lambda x, _: x)
    value_net = ValueNetwork(obs_dim, lambda x, _: x, (64, 64))
    ppo = PPO(device, config, policy_net, value_net)

    max_steps = int(1e6)
    episode_reward = 0
    episode_steps = 0
    total_steps = 0
    s, _ = env.reset()

    while total_steps < max_steps:
        a, a_logprob = ppo.get_action(s)
        a_env = a * action_scale + action_bias  # Scale action to [-2, 2]
        s_next, r, terminated, truncated, _ = env.step(a_env)
        episode_reward += r
        episode_steps += 1
        total_steps += 1

        # Store experience (obs and privileged_obs are the same)
        replay_buffer.store(s, s, a, r, terminated, truncated, a_logprob)

        if len(replay_buffer) >= config.batch_size:
            ppo.update(replay_buffer, total_steps)

        s = s_next

        if terminated or truncated:
            # print(f"Episode ended at step {total_steps}, Reward: {episode_reward}")
            ppo._logger.log_update(episode_reward=episode_reward, episode_steps=episode_steps)
            episode_reward = 0
            episode_steps = 0
            s, _ = env.reset()

if __name__ == "__main__":
    train()