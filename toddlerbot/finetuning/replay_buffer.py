import os
import time
import torch
import numpy as np
from tqdm import tqdm 
from toddlerbot.finetuning.utils import CONST_EPS, RewardScaling, normalize
from copy import deepcopy

class OnlineReplayBuffer:
    def __init__(
        self, 
        device: torch.device, 
        obs_dim: int, privileged_obs_dim: int, action_dim: int, max_size: int, seed: int = 0
    ) -> None:
        self._device = device
        self._dtype = np.float32

        self._obs = np.zeros((max_size, obs_dim), dtype=self._dtype)
        self._privileged_obs = np.zeros((max_size, privileged_obs_dim), dtype=self._dtype)
        self._action = np.zeros((max_size, action_dim))
        self._reward = np.zeros((max_size, 1))
        self._done = np.zeros((max_size, 1))
        self._return = np.zeros((max_size, 1))
        self._advantage = np.zeros((max_size, 1))
        self.rng = np.random.default_rng(seed=seed)

        self._size = 0
        self.start_collection = False

    def __len__(self):
        return self._size
    
    def store(
        self,
        s: np.ndarray,
        s_p: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        done: bool
    ) -> None:

        if not self.start_collection:
            self.start_collection = True
            self.init_time = time.time()
        self._obs[self._size] = s
        self._action[self._size] = a
        self._reward[self._size] = r
        self._privileged_obs[self._size] = s_p
        self._done[self._size] = done
        self._size += 1
        if self._size % 400 == 0:
            print(f"Data size: {self._size}, Data fps: {self._size/(time.time() - self.init_time)}")


    def compute_return(
        self, gamma: float
    ) -> None:

        pre_return = 0
        for i in tqdm(reversed(range(self._size)), desc='Computing the returns'):
            self._return[i] = self._reward[i] + gamma * pre_return * (1 - self._done[i])
            pre_return = self._return[i]



    def compute_advantage(
        self, gamma:float, lamda: float, value
    ) -> None:
        delta = np.zeros_like(self._reward)

        pre_value = 0
        pre_advantage = 0

        for i in tqdm(reversed(range(self._size)), 'Computing the advantage'):
            current_state = torch.FloatTensor(self._obs[i]).to(self._device)
            current_value = value(current_state).cpu().data.numpy().flatten()

            delta[i] = self._reward[i] + gamma * pre_value * (1 - self._done[i]) - current_value
            self._advantage[i] = delta[i] + gamma * lamda * pre_advantage * (1 - self._done[i])

            pre_value = current_value
            pre_advantage = self._advantage[i]

        self._advantage = (self._advantage - self._advantage.mean()) / (self._advantage.std() + CONST_EPS)


    def shuffle(self,):
        indices = np.arange(self._obs.shape[0])
        self.rng.shuffle(indices)
        self._obs = self._obs[indices]
        self._action = self._action[indices]
        self._reward = self._reward[indices]
        self._done = self._done[indices]
        self._privileged_obs = self._privileged_obs[indices]
        self._return = self._return[indices]
        self._advantage = self._advantage[indices]

    def sample_all(self,):
        return {
            "observations": self._obs[:self._size - 1].copy(),
            "actions": self._action[:self._size - 1].copy(),
            "next_observations": self._obs[1:self._size].copy(),
            "terminals": self._done[:self._size - 1].copy(),
            "rewards": self._reward[:self._size - 1].copy()
        }
    
    def sample_aug_all(self,):
        self._obs = np.concatenate((self._obs, self._aug_state), axis=0)
        self._action = np.concatenate((self._action, self._action), axis = 0)
        self._done = np.concatenate((self._done, self._done), axis = 0)
        self._reward = np.concatenate((self._reward, self._reward), axis = 0)
        indices = np.arange(self._obs.shape[0])
        self.rng.shuffle(indices)
        return {
            "observations": self._obs[indices].copy(),
            "actions": self._action[indices].copy(),
            "next_observations": self._obs[indices + 1].copy(),
            "terminals": self._done[indices].copy(),
            "rewards": self._reward[indices].copy()
        }
    
    def sample(
        self, batch_size: int
    ) -> tuple:

        ind = self.rng.integers(0, int(self._size), size=batch_size)

        return (
            torch.FloatTensor(self._obs[ind]).to(self._device),
            torch.FloatTensor(self._privileged_obs[ind]).to(self._device),
            torch.FloatTensor(self._action[ind]).to(self._device),
            torch.FloatTensor(self._reward[ind]).to(self._device),
            torch.FloatTensor(self._obs[ind + 1]).to(self._device),
            torch.FloatTensor(self._privileged_obs[ind + 1]).to(self._device),
            torch.FloatTensor(self._action[ind + 1]).to(self._device),
            torch.FloatTensor(self._done[ind]).to(self._device),
            torch.FloatTensor(self._return[ind]).to(self._device),
            torch.FloatTensor(self._advantage[ind]).to(self._device)
        )
    
    def sample_aug_state(self, batch_size: int):
        self._obs = np.concatenate((self._obs, self._aug_state), axis=0)
        ind = self.rng.integers(0, int(self._size * 2), size=batch_size)
        return (
            torch.FloatTensor(self._obs[ind]).to(self._device)
        )


    def augmentaion(self, alpha = 0.75, beta = 1.25):
        z = self.rng.uniform(low=alpha, high=beta, size=self._obs.shape)
        self._aug_state = deepcopy(self._obs) * z
        self._aug_next_state = deepcopy(self._next_obs) * z


class OfflineReplayBuffer(OnlineReplayBuffer):

    def __init__(
        self, device: torch.device, 
        obs_dim: int, privileged_obs_dim: int, action_dim: int, max_size: int, seed: int = 0
    ) -> None:
        super().__init__(device, obs_dim, privileged_obs_dim, action_dim, max_size, seed)


    def load_dataset(
        self, dataset: dict, clip = False
    ) -> None:
        if clip:
            lim = 1. - 1e-5
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)
        self._state = dataset['observations'][:-1, :]
 
        self._action = dataset['actions'][:-1, :]
        self._reward = dataset['rewards'].reshape(-1, 1)[:-1, :]
        self._next_state = dataset['observations'][1:, :]
        self._next_action = dataset['actions'][1:, :]
        self._done = 1. - (dataset['terminals'].reshape(-1, 1)[:-1, :] | dataset['timeouts'].reshape(-1, 1)[:-1, :])

        self._size = len(dataset['actions']) - 1

    def reward_normalize(self, gamma = 0.99, scaling = 'dynamic'): # dynamic/normal/number
        if scaling == 'dynamic':
            print('scaling reward dynamically')
            reward_norm = RewardScaling(1, gamma)
            rewards = self._reward.flatten()
            for i, not_done in enumerate(self._done.flatten()):
                if not not_done:
                    reward_norm.reset()
                else:
                    rewards[i] = reward_norm(rewards[i])
            self._reward = rewards.reshape(-1, 1)

            return reward_norm
        elif scaling == 'normal':
            print('use normal reward scaling')
            normalized_rewards = normalize(self._state, self._action, deepcopy(self._reward.flatten()), self._done.flatten(), 1 - self._done.flatten(), self._next_state)
            self._reward = normalized_rewards.reshape(-1, 1)

        elif scaling == 'number':
            print('use a fixed number reward scaling')
            self._reward = self._reward * 0.1
        else:
            print('donnot use any reward scaling')
            self._reward = self._reward 


    def normalize_state(
        self
    ) -> tuple:

        mean = self._state.mean(0, keepdims=True)
        std = self._state.std(0, keepdims=True) + CONST_EPS
        self._state = (self._state - mean) / std
        self._next_state = (self._next_state - mean) / std
        return (mean, std)