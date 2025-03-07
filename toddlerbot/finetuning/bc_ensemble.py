import torch
import numpy as np
import gymnasium as gym
from toddlerbot.finetuning.replay_buffer import OnlineReplayBuffer
from toddlerbot.finetuning.networks import GaussianPolicyNetwork
from toddlerbot.finetuning.utils import log_prob_func, orthogonal_initWeights
import os
from copy import deepcopy
from typing import List

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class BehaviorCloning:
    def __init__(
        self,
        device: torch.device,
        observation_size: List[int],
        hidden_layers: int,
        action_size: int,
        preprocess_observations_fn: callable,
        policy_lr: float,
        batch_size: int,
        policy_id: int,
        num_policy: int,
        kl_type: str,
    ) -> None:
        super().__init__()
        self._device = device
        self._num_policy = num_policy
        self.kl_type = kl_type
        self._policy_net = GaussianPolicyNetwork(
            observation_size, hidden_layers, action_size, preprocess_observations_fn
        ).to(device)
        orthogonal_initWeights(self._policy_net)
        self._optimizer = torch.optim.Adam(self._policy_net.parameters(), lr=policy_lr)
        self._lr = policy_lr
        self._batch_size = batch_size
        self._policy_id = policy_id

    def loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        dist = self.get_dist(state)
        log_prob = log_prob_func(dist, action)
        loss = (-log_prob).mean()

        return loss

    def get_dist(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return self._policy_net(state)

    def single_update(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> float:
        policy_loss = self.loss(state, action)

        self._optimizer.zero_grad()
        policy_loss.backward()
        self._optimizer.step()

        return policy_loss.item()

    def joint_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        policies: List["BehaviorCloning"],
        alpha: float,
        bc_kl: str,
        all_action_probs: list,
        all_others: List["BehaviorCloning"],
        pi_action: torch.Tensor,
        pre_id: int,
    ) -> torch.Tensor:
        if bc_kl == "pi":
            all_action_probs = []
            for bc in all_others:  # if sampled from pi, we need to resample the action for each bc other than use the same action from dataset
                dist = bc.get_dist(state)
                action_prob = log_prob_func(dist, pi_action)
                all_action_probs.append(action_prob)

            all_prob_a_t = torch.cat(
                all_action_probs, dim=1
            )  # tensor: (batch size, num_policies)
            max_action_prob, policy_id = all_prob_a_t.max(-1)  # tensor: (batch_size)

        elif bc_kl == "data":
            if len(policies) != 1 or len(all_action_probs) == 0:
                for bc in policies:
                    dist = bc.get_dist(state)
                    action_prob = log_prob_func(dist, action)
                    all_action_probs.append(action_prob)

            else:
                dist = policies[0].get_dist(state)
                action_prob = log_prob_func(dist, action)
                all_action_probs[pre_id] = action_prob

            all_prob_a_t = torch.cat(
                all_action_probs, dim=1
            )  # tensor: (batch size, num_policies)
            max_action_prob, policy_id = all_prob_a_t.max(-1)  # tensor: (batch_size)

        # calculate bc loss
        dist_pi = self.get_dist(state)
        log_prob = log_prob_func(dist_pi, action)
        bc_loss = -log_prob

        if self.kl_type == "heuristic":
            loss = bc_loss - alpha * (log_prob - max_action_prob.detach())
        else:
            loss = torch.where(
                log_prob > max_action_prob,
                bc_loss - alpha * (log_prob),
                bc_loss + alpha * (log_prob),
            )

        return loss.mean(), all_action_probs

    def update(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        policies: list,
        alpha: float,
        is_single_train: bool = False,
        all_action_probs: list = None,
        all_others: list = None,
        pi_action: torch.Tensor = None,
        bc_kl: str = None,
        pre_id: int = None,
    ) -> float:
        if is_single_train:
            policy_loss = self.loss(state, action)
            all_action_probs = 0
        else:
            policy_loss, all_action_probs = self.joint_loss(
                state,
                action,
                policies=policies,
                alpha=alpha,
                bc_kl=bc_kl,
                all_action_probs=all_action_probs,
                all_others=all_others,
                pi_action=pi_action,
                pre_id=pre_id,
            )

        self._optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy_net.parameters(), 1.0)

        self._optimizer.step()

        return policy_loss.item(), all_action_probs

    def select_action(
        self, s: torch.Tensor, is_sample: bool
    ) -> torch.Tensor:
        # dist = self.get_dist(s)
        # if is_sample:
        #     action = dist.sample()
        # else:    
        #     action = dist.mean
        # # clip 
        # action = action.clamp(-1., 1.)
        return self._policy_net.select_action(s, is_sample)
    
    def offline_evaluate(
        self,
        env: gym.Env,
        seed: int,
        state_mean: np.ndarray = 0.0,
        state_std: np.ndarray = 1.0,
        eval_episodes: int = 10,
    ) -> float:
        total_reward = 0
        for _ in range(eval_episodes):
            state, _ = env.reset(seed=seed)
            done = False
            while not done:
                state = torch.FloatTensor((np.array(state).reshape(1, -1) - state_mean) / state_std).to(
                    self._device
                )
                action = self.select_action(state, is_sample=False).cpu().data.numpy().flatten()
                state, reward, term, trunc, _ = env.step(action)
                total_reward += reward
                done = term or trunc

        avg_reward = total_reward / eval_episodes
        return avg_reward

    def save(self, bc_folder: str, save_id: int) -> None:
        os.makedirs(bc_folder, exist_ok=True)
        torch.save(self._policy_net.state_dict(), os.path.join(bc_folder, "bc_{}.pt".format(save_id)))
        print("Behavior policy {} parameters saved in {}".format(save_id, bc_folder))

    def load(self, bc_folder: str, save_id: int) -> None:
        self._policy_net.load_state_dict(
            torch.load(os.path.join(bc_folder, "bc_{}.pt".format(save_id)))
        )
        print("Behavior policy {} parameters loaded from {}".format(save_id, bc_folder))



class BehaviorCloningEnsemble:
    def __init__(
        self,
        num_policy: int,
        device: torch.device,
        observation_size: int,
        hidden_layers: List[int],
        action_size: int,
        preprocess_observations_fn: callable,
        policy_lr: float,
        batch_size: int,
        bc_kl: str = "data",
        kl_type: str = "heuristic",
    ) -> None:
        super().__init__()

        self.num = num_policy
        self.batch_size = batch_size
        self.device = device
        self.state_dim = observation_size
        self.action_dim = action_size
        self.bc_kl = bc_kl

        ensemble = []
        for i in range(self.num):
            bc = BehaviorCloning(
                device=device,
                observation_size=observation_size,
                hidden_layers=hidden_layers,
                action_size=action_size,
                policy_lr=policy_lr,
                preprocess_observations_fn=preprocess_observations_fn,
                batch_size=batch_size,
                policy_id=i,
                num_policy=num_policy,
                kl_type=kl_type,
            )
            ensemble.append(bc)
        self._ensemble: List[BehaviorCloning] = ensemble

    def get_ensemble(
        self,
    ) -> list:
        return self._ensemble

    def joint_train(
        self, replay_buffer: OnlineReplayBuffer, alpha: float, shuffle: bool = True
    ) -> float:
        state, _, action, _, _, _, _, _, _, _, _ = replay_buffer.sample(self.batch_size)

        losses = []
        # separately train each polciy
        if alpha == 0.0 or self.num == 1:
            for bc in self._ensemble:
                each_loss = bc.single_update(state, action)
                losses.append(each_loss)
        # jointly train each behavior policy
        else:
            all_action_probs = []
            pi_ids = np.arange(0, self.num)
            # shuffle pi's order
            if shuffle:
                np.random.shuffle(pi_ids)

                for i, pi_id in enumerate(pi_ids):
                    bc = self._ensemble[pi_id]
                    if self.bc_kl == "pi":
                        pi_action = bc.select_action(state, is_sample = True)
                        all_others = deepcopy(self._ensemble)
                    else:
                        pi_action, all_others = None, None

                    if i == 0:
                        others = deepcopy(self._ensemble)
                        del others[pi_id]
                        first_pi, pre_id = pi_id, pi_id
                    else:
                        others = [self._ensemble[pi_ids[i - 1]]]
                        if pi_id > first_pi:
                            pre_id = pi_id - 1
                        else:
                            pre_id = pi_id

                    each_loss, all_action_probs= bc.update(
                        state=state,
                        action=action,
                        policies=others,
                        alpha=alpha,
                        all_action_probs=all_action_probs,
                        all_others=all_others,
                        pi_action=pi_action,
                        bc_kl=self.bc_kl,
                        pre_id=pre_id,
                    )
                    losses.append(each_loss)
            else:
                for i, bc in enumerate(self._ensemble):
                    if self.bc_kl == "pi":
                        pi_action = bc.select_action(state, is_sample=True)
                        all_others = deepcopy(self._ensemble)
                    else:
                        pi_action, all_others = None, None
                    if i == 0:
                        others = deepcopy(self._ensemble)
                        del others[i]
                        pre_id = None
                    else:
                        others = [self._ensemble[i - 1]]
                        pre_id = i - 1

                    each_loss, all_action_probs = bc.update(
                        state=state,
                        action=action,
                        policies=others,
                        alpha=alpha,
                        all_action_probs=all_action_probs,
                        all_others=all_others,
                        pi_action=pi_action,
                        bc_kl=self.bc_kl,
                        pre_id=pre_id,
                    )
                    losses.append(each_loss)

        return np.array(losses)

    def evaluation(
        self,
        env: gym.Env,
        seed: int,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        eval_episodes: int = 10,
    ) -> list:
        scores = []
        for i in range(self.num):
            bc = self._ensemble[i]
            each_score = bc.offline_evaluate(
                env, seed, state_mean, state_std, eval_episodes=eval_episodes
            )
            scores.append(each_score)
        return np.array(scores)

    def ensemble_save(self, bc_folder: str, save_id: list) -> None:
        for i in save_id:
            bc = self._ensemble[i]
            bc.save(bc_folder, i)

    def load_pi(self, bc_folder: str) -> None:
        for i in range(self.num):
            self._ensemble[i].load(bc_folder)
