import os
import pickle
from collections import deque
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from toddlerbot.finetuning.server_client import RemoteClient, dump_experience_to_base64
from toddlerbot.finetuning.utils import CONST_EPS, RewardScaling, normalize
from toddlerbot.sim import Obs


class RemoteReplayBuffer:
    def __init__(
        self,
        client: RemoteClient,
        buffer_size: int,
        num_obs_history: int,
        num_privileged_obs_history: int,
        enlarge_when_full: int = 0,
    ):
        self.client = client
        self.num_obs_history = num_obs_history  # e.g., 15
        self.num_privileged_obs_history = num_privileged_obs_history  # e.g., 15
        self._max_size = buffer_size
        self._count = 0
        self.is_overwriting = False
        self.enlarge_when_full = enlarge_when_full

    def store(
        self,
        obs_arr: np.ndarray,
        privileged_obs_arr: np.ndarray,
        action: np.ndarray,
        reward,
        done,
        truncated,
        action_logprob,
        raw_obs,
    ):
        # Instead of sending the full stacked version,
        # send only the latest frame of each.
        # Assume obs_arr is a 1D np.array of length (obs_dim * num_obs_history)
        obs_frame_dim = obs_arr.size // self.num_obs_history  # e.g., 1245/15 = 83
        priv_frame_dim = (
            privileged_obs_arr.size // self.num_privileged_obs_history
        )  # e.g., 1890/15 = 126

        latest_obs = obs_arr[:obs_frame_dim]
        latest_priv = privileged_obs_arr[:priv_frame_dim]
        # import ipdb; ipdb.set_trace()
        data = {
            "s": latest_obs,
            "s_p": latest_priv,
            "a": action,
            "r": float(reward),
            "done": bool(done),
            "truncated": bool(truncated),
            "a_logprob": action_logprob,
            "raw_obs": raw_obs,
            # Optionally, include additional fields (e.g., a sequence number) for debugging.
        }
        b64_experience = dump_experience_to_base64(data)
        payload = {
            "type": "experience",
            "data_b64": b64_experience,
        }
        self.client.send_experience(payload)
        # print(f"Sent message of length {len(pickle.dumps(data))} at {time.time()} to {self.remote_address}")
        self._count += 1
        # When the count reaches max_size, set overwriting and reset _count
        if self._count >= self._max_size and not self.enlarge_when_full > 0:
            print("Remote buffer full, replacing old data")
            self.is_overwriting = True
            self._count = 0

    def __len__(self) -> int:
        # If overwriting, the remote buffer holds max_size items.
        if self.is_overwriting:
            return self._max_size
        return self._count

    def reset(self):
        self._count = 0
        self.is_overwriting = False
        # self.client.send_experience({"type": "reset"})


class OnlineReplayBuffer:
    def __init__(
        self,
        device: torch.device,
        obs_dim: int,
        privileged_obs_dim: int,
        action_dim: int,
        max_size: int,
        validation_size: int = 0,
        seed: int = 0,
        enlarge_when_full: int = 0,
        keep_data_after_reset: bool = True,  # New parameter
    ):
        self._device = device
        self._dtype = np.float32

        self._obs = np.zeros((max_size, obs_dim), dtype=self._dtype)
        self._privileged_obs = np.zeros(
            (max_size, privileged_obs_dim), dtype=self._dtype
        )
        self._action = np.zeros((max_size, action_dim), dtype=self._dtype)
        self._reward = np.zeros((max_size, 1), dtype=self._dtype)
        self._terminated = np.zeros((max_size, 1), dtype=self._dtype)
        self._truncated = np.zeros((max_size, 1), dtype=self._dtype)
        self._return = np.zeros((max_size, 1), dtype=self._dtype)
        self._action_logprob = np.zeros((max_size, 1), dtype=self._dtype)
        self._raw_obs = deque(maxlen=max_size)
        self.rng = np.random.default_rng(seed=seed)

        self.enlarge_when_full = enlarge_when_full
        self.is_overwriting = False
        self._truncated_temp = False

        self._size = 0
        self._validation_size = validation_size
        self._max_size = max_size
        self._window_size = 1
        self.keep_data_after_reset = keep_data_after_reset
        self._current_start = 0  # Tracks sampling start index

    def __len__(self):
        return self._size - self._current_start

    def store(
        self,
        s: np.ndarray,
        s_p: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        done: bool,
        truncated: bool,
        a_logprob: np.ndarray,
        raw_obs: Obs = None,
    ) -> None:
        self._obs[self._size] = s
        self._action[self._size] = a
        self._reward[self._size] = r
        self._privileged_obs[self._size] = s_p
        self._action_logprob[self._size] = a_logprob
        self._terminated[self._size] = done
        self._raw_obs.append(raw_obs)
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

    def shift_action(self, shift_steps: int) -> None:
        self._action = np.roll(self._action, -shift_steps)
        self._size -= shift_steps

    def reset(self):
        if self.keep_data_after_reset:
            self._current_start = self._size
        else:
            self._raw_obs.clear()
            self._size = 0
        self._truncated_temp = False
        self.is_overwriting = False

    def compute_return(self, gamma: float) -> None:
        pre_return = 0
        for i in tqdm(reversed(range(self._size)), desc="Computing the returns"):
            self._return[i] = self._reward[i] + gamma * pre_return * (
                1 - self._terminated[i]
            )
            pre_return = self._return[i]

    def sample_all(self) -> tuple:
        current_size = self._size
        start = max(
            0,
            self._current_start
            - (current_size - self._current_start) * (self._window_size - 1),
        )
        return (
            torch.FloatTensor(self._obs[start : current_size - 1]).to(self._device),
            torch.FloatTensor(self._privileged_obs[start : current_size - 1]).to(
                self._device
            ),
            torch.FloatTensor(self._action[start : current_size - 1]).to(self._device),
            torch.FloatTensor(self._reward[start : current_size - 1]).to(self._device),
            torch.FloatTensor(self._obs[start + 1 : current_size]).to(self._device),
            torch.FloatTensor(self._privileged_obs[start + 1 : current_size]).to(
                self._device
            ),
            torch.FloatTensor(self._action[start + 1 : current_size]).to(self._device),
            torch.FloatTensor(self._terminated[start : current_size - 1]).to(
                self._device
            ),
            torch.FloatTensor(self._truncated[start : current_size - 1]).to(
                self._device
            ),
            torch.FloatTensor(self._return[start : current_size - 1]).to(self._device),
            torch.FloatTensor(self._action_logprob[start : current_size - 1]).to(
                self._device
            ),
        )

    def sample(
        self, batch_size: int, sample_validation: bool = False, no_term=False
    ) -> tuple:
        assert self._size > self._validation_size
        if no_term:
            condition = (1 - self._terminated[self._current_start : self._size]) * (
                1 - self._truncated[self._current_start : self._size]
            )
        else:
            condition = 1 - self._truncated[self._current_start : self._size]
        valid_indices = np.flatnonzero(condition) + self._current_start
        if sample_validation:
            valid_indices = valid_indices[: self._validation_size - 1]
        else:
            valid_indices = valid_indices[self._validation_size : -1]
        if valid_indices.size < batch_size:
            ind = self.rng.choice(valid_indices, size=batch_size, replace=True)
        else:
            ind = self.rng.choice(valid_indices, size=batch_size, replace=False)
        return (
            torch.FloatTensor(self._obs[ind]).to(self._device),
            torch.FloatTensor(self._privileged_obs[ind]).to(self._device),
            torch.FloatTensor(self._action[ind]).to(self._device),
            torch.FloatTensor(self._reward[ind]).to(self._device),
            torch.FloatTensor(self._obs[ind + 1]).to(self._device),
            torch.FloatTensor(self._privileged_obs[ind + 1]).to(self._device),
            torch.FloatTensor(self._action[ind + 1]).to(self._device),
            torch.FloatTensor(self._terminated[ind]).to(self._device),
            torch.FloatTensor(self._truncated[ind]).to(self._device),
            torch.FloatTensor(self._return[ind]).to(self._device),
            torch.FloatTensor(self._action_logprob[ind]).to(self._device),
        )

    def split_train_valid(self, valid_ratio: float) -> tuple:
        self._validation_size = int(self._size * valid_ratio)

    def __getitem__(self, index):
        return (
            self._obs[index],
            self._privileged_obs[index],
            self._action[index],
            self._reward[index],
            self._terminated[index],
            self._truncated[index],
            self._action_logprob[index],
            self._raw_obs[index],
        )

    def save_compressed(self, path):
        np.savez_compressed(
            os.path.join(path, "buffer.npz"),
            observations=self._obs[: self._size],
            privileged_obs=self._privileged_obs[: self._size],
            actions=self._action[: self._size],
            rewards=self._reward[: self._size],
            terminals=self._terminated[: self._size],
            truncated=self._truncated[: self._size],
            returns=self._return[: self._size],
            advantage=self._action_logprob[: self._size],
            size=self._size,
        )
        with open(os.path.join(path, "raw_obs.pkl"), "wb") as f:
            pickle.dump(self._raw_obs, f)

    def load_compressed(self, path):
        data = np.load(os.path.join(path, "buffer.npz"), allow_pickle=True)
        data_size = data["size"]
        if self._size + data_size > self._max_size:
            print("Data size is larger than the buffer size, enlarge the buffer")
            self.enlarge(data_size - self._max_size + self._size)
        self._obs[self._size : self._size + data_size] = data["observations"]
        self._privileged_obs[self._size : self._size + data_size] = data[
            "privileged_obs"
        ]
        self._action[self._size : self._size + data_size] = data["actions"]
        self._reward[self._size : self._size + data_size] = data["rewards"]
        self._terminated[self._size : self._size + data_size] = data["terminals"]
        if "truncated" in data.keys():
            self._truncated[self._size : self._size + data_size] = data["truncated"]
        self._return[self._size : self._size + data_size] = data["returns"]
        self._action_logprob[self._size : self._size + data_size] = data["advantage"]
        self._truncated[self._size + data_size - 1] = True
        if os.path.exists(os.path.join(path, "raw_obs.pkl")):
            with open(os.path.join(path, "raw_obs.pkl"), "rb") as f:
                raw_obs = pickle.load(f)
        else:
            assert "raw_obs" in data.keys()
            raw_obs = data["raw_obs"]
        for done_idx in np.flatnonzero(data["terminals"]):
            raw_obs[done_idx].is_done = True
        self._raw_obs.extend(raw_obs)
        self._size += data_size
        if self._size >= self._max_size:
            if self.enlarge_when_full > 0:
                self.enlarge(self.enlarge_when_full)
            else:
                print("Buffer is full, replacing the old data")
                self.is_overwriting = True
                self._size = 0

    def enlarge(self, new_size):
        self._obs = np.concatenate(
            (self._obs, np.zeros((new_size, self._obs.shape[1]))), axis=0
        )
        self._privileged_obs = np.concatenate(
            (self._privileged_obs, np.zeros((new_size, self._privileged_obs.shape[1]))),
            axis=0,
        )
        self._action = np.concatenate(
            (self._action, np.zeros((new_size, self._action.shape[1]))), axis=0
        )
        self._reward = np.concatenate(
            (self._reward, np.zeros((new_size, self._reward.shape[1]))), axis=0
        )
        self._terminated = np.concatenate(
            (self._terminated, np.zeros((new_size, self._terminated.shape[1]))), axis=0
        )
        self._truncated = np.concatenate(
            (self._truncated, np.zeros((new_size, self._truncated.shape[1]))), axis=0
        )
        self._return = np.concatenate(
            (self._return, np.zeros((new_size, self._return.shape[1]))), axis=0
        )
        self._action_logprob = np.concatenate(
            (self._action_logprob, np.zeros((new_size, self._action_logprob.shape[1]))),
            axis=0,
        )
        self._max_size += new_size
        self._raw_obs = deque(self._raw_obs, maxlen=int(self._max_size))
        print(f"Buffer size is enlarged to {self._max_size}")


class OfflineReplayBuffer(OnlineReplayBuffer):
    def __init__(
        self,
        device: torch.device,
        obs_dim: int,
        privileged_obs_dim: int,
        action_dim: int,
        max_size: int,
        seed: int = 0,
    ) -> None:
        super().__init__(
            device, obs_dim, privileged_obs_dim, action_dim, max_size, seed
        )

    def normalize_reward(self, gamma=0.99, scaling="dynamic"):  # dynamic/normal/number
        if scaling == "dynamic":
            print("scaling reward dynamically")
            reward_norm = RewardScaling(1, gamma)
            rewards = self._reward.flatten()
            for i, not_done in enumerate(self._done.flatten()):
                if not not_done:
                    reward_norm.reset()
                else:
                    rewards[i] = reward_norm(rewards[i])
            self._reward = rewards.reshape(-1, 1)

            return reward_norm
        elif scaling == "normal":
            print("use normal reward scaling")
            normalized_rewards = normalize(
                self._state,
                self._action,
                deepcopy(self._reward.flatten()),
                self._done.flatten(),
                1 - self._done.flatten(),
                self._next_state,
            )
            self._reward = normalized_rewards.reshape(-1, 1)

        elif scaling == "number":
            print("use a fixed number reward scaling")
            self._reward = self._reward * 0.1
        else:
            print("donnot use any reward scaling")
            self._reward = self._reward

    def normalize_state(self) -> tuple:
        mean = self._state.mean(0, keepdims=True)
        std = self._state.std(0, keepdims=True) + CONST_EPS
        self._state = (self._state - mean) / std
        self._next_state = (self._next_state - mean) / std
        return (mean, std)
