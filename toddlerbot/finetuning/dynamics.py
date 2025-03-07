
import pickle
from typing import Dict, Tuple
from collections import defaultdict
import numpy as np
import torch
from toddlerbot.finetuning.finetune_config import FinetuneConfig
import os
import torch.nn.functional as F
from toddlerbot.finetuning.networks import ValueNetwork, QNetwork, DynamicsNetwork, GaussianPolicyNetwork, load_jax_params, load_jax_params_into_pytorch
from toddlerbot.finetuning.replay_buffer import OfflineReplayBuffer
from toddlerbot.finetuning.ppo import PPO

def _terminate_fn(privileged_obs: np.ndarray, config: FinetuneConfig) -> np.ndarray:
    # last obs is the frist obs in the frames_stack frames
    if privileged_obs.ndim == 1:
        privileged_obs = privileged_obs[None, :]
    last_privileged_obs = np.split(privileged_obs, config.frame_stack, axis=1)[0]
    ee_force = last_privileged_obs[:, -6:-3]
    torso_euler = last_privileged_obs[:, -9:-6]
    motor_pos_error = last_privileged_obs[:, -45:-15] # TODO: verify this

    def terminate_fn_single(ee_force, torso_euler, motor_pos_error):
        if ee_force[2] < config.healty_ee_force_z[0] or ee_force[2] > config.healty_ee_force_z[1]:
            return True
        if ee_force[0] > config.healty_ee_force_xy[1] or ee_force[1] > config.healty_ee_force_xy[1]:
            return True
        if ee_force[0] < config.healty_ee_force_xy[0] or ee_force[1] < config.healty_ee_force_xy[0]:
            return True
        if torso_euler[0] < config.healty_torso_roll[0] or torso_euler[0] > config.healty_torso_roll[1]:
            return True
        if torso_euler[1] < config.healty_torso_pitch[0] or torso_euler[1] > config.healty_torso_pitch[1]:
            return True
        # if np.linalg.norm(motor_pos_error) > config.pos_error_threshold:
        #     return True
        return False

    # import ipdb; ipdb.set_trace()
    return np.array([terminate_fn_single(ee_force, torso_euler, motor_pos_error) for ee_force, torso_euler, motor_pos_error in zip(ee_force, torso_euler, motor_pos_error)])

def get_obs_from_priviledged_obs(privileged_obs: torch.Tensor | np.ndarray, stack_frame: int = 15) -> np.ndarray:
    concat_func = torch.concat if isinstance(privileged_obs, torch.Tensor) else np.concatenate
    if privileged_obs.ndim == 1:
        privileged_obs = privileged_obs.reshape(stack_frame, -1)
        obs = concat_func([privileged_obs[:, :77], privileged_obs[:, 110:116]], axis=-1) # 110: 116
        return obs.reshape(-1)
    elif privileged_obs.ndim == 2:
        batch_size = privileged_obs.shape[0]
        privileged_obs = privileged_obs.reshape(batch_size, stack_frame, -1)
        obs = concat_func([privileged_obs[:, :, :77], privileged_obs[:, :, 110:116]], axis=-1)
        return obs.reshape(batch_size, -1)

class StandardScaler(object):
    def __init__(self):
        pass

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        return self.std * data + self.mu
    
    def save_scaler(self, save_path):
        mu_path = os.path.join(save_path, "mu.npy")
        std_path = os.path.join(save_path, "std.npy")
        np.save(mu_path, self.mu)
        np.save(std_path, self.std)
    
    def load_scaler(self, load_path):
        mu_path = os.path.join(load_path, "mu.npy")
        std_path = os.path.join(load_path, "std.npy")
        self.mu = np.load(mu_path)
        self.std = np.load(std_path)

    def transform_tensor(self, obs_action: torch.Tensor, device):
        obs_action = obs_action.cpu().numpy()
        obs_action = self.transform(obs_action)
        obs_action = torch.tensor(obs_action, device=device)
        return obs_action

class BaseDynamics:
    def __init__(
        self, 
        device: torch.device, 
        model: DynamicsNetwork,
        config: FinetuneConfig,
        terminate_fn: callable = _terminate_fn,
        extract_obs_fn: callable = get_obs_from_priviledged_obs
    ) -> None:
        super().__init__()
        self._device = device
        self._model = model.to(device)
        self._optim = torch.optim.Adam(
            self._model.parameters(), 
            lr=config.dynamics_lr,
            )
        self._config = config
        self._batch_size = config.dynamics_batch_size
        self._terminate_fn = terminate_fn
        self._extract_obs_fn = extract_obs_fn

    def update(
        self, replay_buffer: OfflineReplayBuffer
    ) -> float:
        _, s, a, r, _, s_n, _, _, _, _, _ = replay_buffer.sample(self._batch_size)
        dynamics_loss = F.mse_loss(self._model(s, a), torch.concatenate([s_n, r], dim=1))
        self._optim.zero_grad()
        dynamics_loss.backward()
        self._optim.step()

        return dynamics_loss.item()

    @torch.no_grad()
    def step(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        obs = torch.FloatTensor(obs).to(self._device)
        act = torch.FloatTensor(action).to(self._device)
        predicted_states = self._model(obs, act).cpu().numpy()
        next_obs = predicted_states[..., :-1]
        rewards = predicted_states[..., -1:]
        terminals = self._terminate_fn(obs, self._config)
        info = {}
        return next_obs, rewards, terminals, info
    
    def save(
        self, path: str
    ) -> None:
        torch.save(self._model.state_dict(), path)
        print('Dynamics parameters saved in {}'.format(path))


    def load(
        self, path: str
    ) -> None:
        self._model.load_state_dict(torch.load(path, map_location=self._device))
        print('Dynamics parameters loaded')


def rollout(
        Q: QNetwork,
        policy: PPO,
        dynamics: BaseDynamics,
        init_obs: np.ndarray,
        init_privileged_obs: np.ndarray,
        rollout_length: int,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        total_q = np.array([])
        rollout_transitions = defaultdict(list)
        # rollout
        obs = init_obs.to(dynamics._device)
        priviledged_obs = init_privileged_obs.to(dynamics._device)
        assert torch.allclose(dynamics._extract_obs_fn(priviledged_obs), obs)
        with torch.no_grad():
            for length in range(rollout_length):
                # import ipdb; ipdb.set_trace()
                obs = dynamics._extract_obs_fn(priviledged_obs)
                actions, _ = policy.get_action(obs, deterministic=True)

                Q_value = Q(priviledged_obs, actions)
                next_priviledged_obs, rewards, terminals, info = dynamics.step(priviledged_obs.cpu().data.numpy(), actions.cpu().data.numpy())

                rollout_transitions["obs"].append(priviledged_obs)
                rollout_transitions["next_obs"].append(next_priviledged_obs)
                rollout_transitions["actions"].append(actions)
                rollout_transitions["rewards"].append(rewards)
                rollout_transitions["terminals"].append(terminals)

                num_transitions += len(priviledged_obs)
                rewards_arr = np.append(rewards_arr, rewards.flatten())
                total_q = np.append(total_q, Q_value.cpu().data.numpy().flatten())
                nonterm_mask = (~terminals).flatten()
                if nonterm_mask.sum() == 0:
                    # print('terminal length: {}'.format(length))
                    break

                priviledged_obs = torch.FloatTensor(next_priviledged_obs[nonterm_mask]).to(dynamics._device)

        return total_q.mean(), rewards_arr.mean(), length
    

def dynamics_eval(config: FinetuneConfig, policy: PPO, Q: QNetwork, dynamics: BaseDynamics, replay_buffer: OfflineReplayBuffer):
    s, s_p, _, _, _, _, _, _, _, _, _ = replay_buffer.sample(config.rollout_batch_size)
    Q_mean, reward_mean, rollout_length = rollout(Q, policy, dynamics, s, s_p, config.ope_rollout_length)
    return Q_mean, reward_mean, rollout_length

if __name__ == "__main__":
    from toddlerbot.finetuning.replay_buffer import OfflineReplayBuffer
    from toddlerbot.finetuning.learners import IQL_QV_Learner
    config = FinetuneConfig()

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open('buffer_mock.pkl', 'rb') as f:
        replay_buffer: OfflineReplayBuffer = pickle.load(f)
    replay_buffer.compute_return(config.gamma)
    replay_buffer._device = device
    dynamics_model = DynamicsNetwork(replay_buffer._privileged_obs.shape[1], replay_buffer._action.shape[1], lambda x, y: x, config.dynamics_hidden_layer_sizes)
    base_dynamics = BaseDynamics(device, dynamics_model, config)
    for i in range(config.dynamics_update_steps):
        loss = base_dynamics.update(replay_buffer)
        print("dynamics update step: {}, loss: {}".format(i, loss))
    policy_net = GaussianPolicyNetwork(
        action_size=replay_buffer._action.shape[1],
        observation_size=replay_buffer._obs.shape[1],
        preprocess_observations_fn=lambda x, y: x,
        hidden_layers=config.policy_hidden_layer_sizes,
    )
    policy_path = os.path.join(
        "toddlerbot",
        "policies",
        "checkpoints",
        "toddlerbot_walk_policy",
    )
    jax_params = load_jax_params(policy_path)
    load_jax_params_into_pytorch(policy_net, jax_params[1]["params"])
    policy_net.eval()
    policy = PPO(
        device=device,
        policy_net=policy_net,
        config=config,
    )
    value_net = ValueNetwork(
        replay_buffer._privileged_obs.shape[1],
        lambda x, y: x,
        config.value_hidden_layer_sizes,
    )
    q_net = QNetwork(
        replay_buffer._privileged_obs.shape[1],
        replay_buffer._action.shape[1],
        lambda x, y: x,
        config.value_hidden_layer_sizes,
    )
    config.use_double_q = False
    iql_learner = IQL_QV_Learner(
        device,
        q_net,
        value_net,
        config,
    )
    for i in range(config.value_update_steps):
        q_loss, value_loss = iql_learner.update(replay_buffer)
        print(i, value_loss, q_loss)
    initial_obs, initial_privileged_obs, _, _, _, _, _, _, _, _, _ = replay_buffer.sample(1)
    obs = initial_obs.reshape(15, -1)
    pri_obs = initial_privileged_obs.reshape(15, -1)
    obs_cat = np.concatenate([pri_obs[:, :77], pri_obs[:, -12:-6]], axis=1) # 110: 116
    
    assert np.allclose(initial_obs, get_obs_from_priviledged_obs(initial_privileged_obs))
    rollout(q_net, policy, base_dynamics, initial_privileged_obs, 1000)