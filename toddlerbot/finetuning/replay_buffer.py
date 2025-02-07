from collections import deque
import torch
import numpy as np
from tqdm import tqdm
from toddlerbot.finetuning.utils import CONST_EPS, RewardScaling, normalize
from toddlerbot.sim import Obs
from copy import deepcopy


class OnlineReplayBuffer:
    def __init__(
        self, device: torch.device, 
        obs_dim: int, privileged_obs_dim: int, action_dim: int, max_size: int, seed: int = 0
    ) -> None:
        self._device = device
        self._dtype = np.float32

        self._obs = np.zeros((max_size, obs_dim), dtype=self._dtype)
        self._privileged_obs = np.zeros(
            (max_size, privileged_obs_dim), dtype=self._dtype
        )
        self._action = np.zeros((max_size, action_dim))
        self._reward = np.zeros((max_size, 1))
        self._done = np.zeros((max_size, 1))
        self._return = np.zeros((max_size, 1))
        self._advantage = np.zeros((max_size, 1))
        self._raw_obs = deque(maxlen=max_size)
        self.rng = np.random.default_rng(seed=seed)

        self.enlarge_when_full = True

        self._size = 0
        self._max_size = max_size
        self.start_collection = False

    def __len__(self):
        return self._size

    def store(
        self,
        s: np.ndarray,
        s_p: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        done: bool,
        raw_obs: Obs = None,
    ) -> None:
        if not self.start_collection:
            self.start_collection = True
        self._obs[self._size] = s
        self._action[self._size] = a
        self._reward[self._size] = r
        self._privileged_obs[self._size] = s_p
        self._done[self._size] = done
        self._size += 1
        self._raw_obs.append(raw_obs)
        if self._size >= self._obs.shape[0]:
            if self.enlarge_when_full:
                enlarge = input(f"Buffer is full, enlarge the buffer from {self._max_size} to {self._max_size * 2}? y/n:")
                while enlarge not in ["y", "n"]:
                    enlarge = input(f"Enlarge the buffer from {self._max_size} to {self._max_size * 2}? y/n:")
                if enlarge == "y":
                    self.enlarge(self._max_size)
                else:
                    self.enlarge_when_full = False
            else:
                print("Buffer is full, replacing the old data")
                self._obs[:-1] = self._obs[1:]
                self._action[:-1] = self._action[1:]
                self._reward[:-1] = self._reward[1:]
                self._privileged_obs[:-1] = self._privileged_obs[1:]
                self._done[:-1] = self._done[1:]
                self._size -= 1

    def compute_return(self, gamma: float) -> None:
        pre_return = 0
        for i in tqdm(reversed(range(self._size)), desc='Computing the returns'):
            self._return[i] = self._reward[i] + gamma * pre_return * (1 - self._done[i])
            pre_return = self._return[i]

    def compute_advantage(self, gamma: float, lamda: float, value) -> None:
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
            "rewards": self._reward[indices].copy(),
        }

    def sample(self, batch_size: int) -> tuple:
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
            torch.FloatTensor(self._advantage[ind]).to(self._device),
        )

    def sample_aug_state(self, batch_size: int):
        self._obs = np.concatenate((self._obs, self._aug_state), axis=0)
        ind = self.rng.integers(0, int(self._size * 2), size=batch_size)
        return torch.FloatTensor(self._obs[ind]).to(self._device)

    def augmentaion(self, alpha=0.75, beta=1.25):
        z = self.rng.uniform(low=alpha, high=beta, size=self._obs.shape)
        self._aug_state = deepcopy(self._obs) * z

    def save_compressed(self, path):
        np.savez_compressed(
            path,
            observations=self._obs[: self._size],
            privileged_obs=self._privileged_obs[: self._size],
            actions=self._action[: self._size],
            rewards=self._reward[: self._size],
            terminals=self._done[: self._size],
            returns=self._return[: self._size],
            anvanatage=self._advantage[: self._size],
            size=self._size,
            raw_obs=self._raw_obs,
        )

    def load_compressed(self, path):
        data = np.load(path, allow_pickle=True)
        data_size = data["size"]
        if self._size + data_size > self._max_size:
            print("Data size is larger than the buffer size, enlarge the buffer")
            self.enlarge(data_size - self._max_size + self._size)
        self._obs[self._size : self._size + data_size] = data["observations"]
        self._privileged_obs[self._size : self._size + data_size] = data["privileged_obs"]
        self._action[self._size : self._size + data_size] = data["actions"]
        self._reward[self._size : self._size + data_size] = data["rewards"]
        self._done[self._size : self._size + data_size] = data["terminals"]
        self._return[self._size : self._size + data_size] = data["returns"]
        self._advantage[self._size : self._size + data_size] = data["anvanatage"]
        self._raw_obs.extend(data["raw_obs"])
        self._size += data_size

    def enlarge(self, new_size):
        self._obs = np.concatenate((self._obs, np.zeros((new_size, self._obs.shape[1]))), axis=0)
        self._privileged_obs = np.concatenate((self._privileged_obs, np.zeros((new_size, self._privileged_obs.shape[1]))), axis=0)
        self._action = np.concatenate((self._action, np.zeros((new_size, self._action.shape[1]))), axis=0)
        self._reward = np.concatenate((self._reward, np.zeros((new_size, self._reward.shape[1]))), axis=0)
        self._done = np.concatenate((self._done, np.zeros((new_size, self._done.shape[1]))), axis=0)
        self._return = np.concatenate((self._return, np.zeros((new_size, self._return.shape[1]))), axis=0)
        self._advantage = np.concatenate((self._advantage, np.zeros((new_size, self._advantage.shape[1]))), axis=0)
        self._max_size += new_size


class OfflineReplayBuffer(OnlineReplayBuffer):
    def __init__(
        self, device: torch.device, 
        obs_dim: int, privileged_obs_dim: int, action_dim: int, max_size: int, seed: int = 0
    ) -> None:
        super().__init__(device, obs_dim, privileged_obs_dim, action_dim, max_size, seed)


    def load_dataset(self, dataset: dict, clip=False) -> None:
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

    def normalize_state(self) -> tuple:
        mean = self._state.mean(0, keepdims=True)
        std = self._state.std(0, keepdims=True) + CONST_EPS
        self._state = (self._state - mean) / std
        self._next_state = (self._next_state - mean) / std
        return (mean, std)
