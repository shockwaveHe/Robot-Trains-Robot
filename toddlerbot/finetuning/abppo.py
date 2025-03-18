import os
from typing import List
import torch
import numpy as np
from copy import deepcopy

from tqdm import tqdm
from toddlerbot.finetuning.networks import GaussianPolicyNetwork, QNetwork, ValueNetwork
from toddlerbot.finetuning.replay_buffer import OnlineReplayBuffer
from toddlerbot.finetuning.learners import ValueLearner, QLearner, IQL_QV_Learner
from toddlerbot.finetuning.finetune_config import FinetuneConfig
from toddlerbot.finetuning.dynamics import BaseDynamics, dynamics_eval
from toddlerbot.finetuning.logger import FinetuneLogger
from torch.distributions import Distribution
from torch.distributions.transformed_distribution import TransformedDistribution

CONST_EPS = 1e-8

def log_prob_func(
    dist: Distribution, action: torch.Tensor
    ) -> torch.Tensor:
    log_prob = dist.log_prob(action)
    if len(log_prob.shape) == 1:
        return log_prob
    else:
        return log_prob.sum(-1, keepdim=True)


class BehaviorProximalPolicyOptimization:
    def __init__(
        self,
        device: torch.device,
        policy_net: GaussianPolicyNetwork,
        config: FinetuneConfig,
    ) -> None:
        self._is_iql = config.is_iql
        self._device = device
        self._policy = policy_net
        self._policy.to(device)
        #orthogonal_initWeights(self._policy)
        self._optimizer = torch.optim.Adam(
            self._policy.parameters(),
            lr=config.policy_lr
            )
        self._policy_lr = config.policy_lr
        self._old_policy: GaussianPolicyNetwork = deepcopy(self._policy)
        
        self._clip_ratio = config.clip_ratio
        self._entropy_weight = config.entropy_weight
        self._decay = config.decay
        self._omega = config.omega
        self._batch_size = config.policy_batch_size
        self._config = config
        self.temperature = config.temperature

    def loss(
        self, 
        s: torch.Tensor,
        advantage: torch.Tensor,
        a: torch.Tensor,
        old_dist: torch.distributions.transformed_distribution.TransformedDistribution,
        clip_ratio_now: float = None,
        kl_logprob_a: torch.Tensor = None
    ) -> torch.Tensor:

        new_dist = self._policy(s)

        new_log_prob = torch.sum(new_dist.log_prob(a), -1)
        old_log_prob = torch.sum(old_dist.log_prob(a), -1)
        ratio = (new_log_prob - old_log_prob).exp()
        # print(ratio.mean().item(), old_log_prob.mean().item(), new_log_prob.mean().item(), advantage.abs().mean().item())
        loss1 =  ratio * advantage 

        if self._config.is_clip_decay:
            if self._config.is_linear_decay:
                self._clip_ratio = clip_ratio_now
            else:
                self._clip_ratio = self._clip_ratio * self._decay
        else:
            self._clip_ratio = self._clip_ratio

        loss2 = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio) * advantage 
        
        # import ipdb; ipdb.set_trace()
        # entropy_loss = torch.sum(new_dist.base_dist.entropy(), dim=-1) * self._entropy_weight
        entropy_loss = self.get_entropy_loss(new_dist)
        loss = -(torch.min(loss1, loss2) + entropy_loss)
        # print(loss1.mean().item(), loss2.mean().item(), entropy_loss.mean().item(), loss.mean().item())
        if self._config.kl_update:
            kl_loss = - self._config.kl_alpha * (new_log_prob - kl_logprob_a.detach())
            loss = loss + kl_loss

        return loss.mean(), ratio.mean().item()

    def weighted_advantage(
        self,
        advantage: torch.Tensor
    ) -> torch.Tensor:
        if self._omega == 0.5:
            return advantage
        else:
            weight = torch.where(advantage > 0, self._omega, (1 - self._omega))
            weight.to(self._device)
            return weight * advantage

    def get_entropy_loss(self, new_dist):
        pre_tanh_sample = new_dist.rsample()
        if isinstance(new_dist, TransformedDistribution):
            pre_tanh_sample = new_dist.transforms[-1].inv(pre_tanh_sample)
            log_det_jac = self._policy.forward_log_det_jacobian(pre_tanh_sample) # return 2.0 * (torch.log(torch.tensor(2.0)) - x - F.softplus(-2.0 * x))
            entropy_loss = torch.sum(new_dist.base_dist.entropy() + log_det_jac, dim=-1) * self._entropy_weight
        else:
            assert isinstance(new_dist, torch.distributions.normal.Normal)
            entropy_loss = torch.sum(new_dist.entropy(), dim=-1) * self._entropy_weight
        return entropy_loss
    
    def set_old_policy(
        self,
    ) -> None:
        self._old_policy.load_state_dict(self._policy.state_dict())

    def load(self, path: str) -> None:
        self._policy.load_state_dict(torch.load(path))

    def update(
        self, 
        s: torch.Tensor,
        advantage: torch.Tensor,
        action: torch.Tensor,
        old_dist: torch.Tensor,
        bppo_lr_now: float = None,
        clip_ratio_now: float = None,
        kl_logprob_a: torch.Tensor = None
    ) -> float:
        policy_loss, ratios = self.loss(s, advantage, action, old_dist, clip_ratio_now, kl_logprob_a)
        
        self._optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy.parameters(), 1.0)
        self._optimizer.step()
        
        if self._config.is_bppo_lr_decay:
            self._scheduler.step()
        if self._config.is_linear_decay:
            for p in self._optimizer.param_groups:
                p['lr'] = bppo_lr_now    
        return policy_loss.item(), ratios

    def get_action(
        self, s: torch.Tensor, deterministic: bool
    ) -> torch.Tensor:
        s.to(self._device)
        dist = self._policy(s)
        if not deterministic:
            action = dist.sample()
        else:
            if isinstance(dist, TransformedDistribution):
                action = dist.base_dist.mode
                for transform in dist.transforms:
                    action = transform(action)
            else:
                assert isinstance(dist, torch.distributions.normal.Normal)
                action = dist.mean    
        # clip 
        action = action.clamp(-1., 1.)
        log_prob = dist.log_prob(action)
        return action, log_prob

class AdaptiveBehaviorProximalPolicyOptimization:
    def __init__(
        self,
        device: torch.device,
        policy_net: GaussianPolicyNetwork,
        config: FinetuneConfig,
    ) -> None:
        self._device = device
        self._config = config
        self._policy_lr = config.policy_lr
        self._clip_ratio= config.clip_ratio
        self._entropy_weight = config.entropy_weight
        self._decay = config.decay
        self._omega = config.omega
        self._discount = config.gamma
        self._batch_size = config.policy_batch_size
        self._num_policy = config.num_policy
        self._is_iql = config.is_iql
        self._kl_update = config.kl_update
        self._kl_strategy = config.kl_strategy
        self._alpha = config.kl_alpha
        self._is_clip_action = config.is_clip_action
        self._policy_net = policy_net
        self._temperature = config.temperature
        self.bppo_ensemble: List[BehaviorProximalPolicyOptimization] = []
        for _ in range(self._num_policy):
            bppo = BehaviorProximalPolicyOptimization(device, deepcopy(policy_net), config)
            self.bppo_ensemble.append(bppo)
    

    def joint_train(self, 
        replay_buffer: OnlineReplayBuffer,
        iql: IQL_QV_Learner = None,
        bppo_lr_now: float = None,
        clip_ratio_now: float = None
        ) -> np.ndarray:
        s, s_p, _, _, _, _, _, _, _, _, _ = replay_buffer.sample(self._batch_size)
        # import ipdb; ipdb.set_trace()
        actions, advantages, dists, kl_logprob_a = self.kl_update(iql, s, s_p, self._kl_update, self._kl_strategy)
        losses, ratios = [], []
        for i, bppo in enumerate(self.bppo_ensemble):
            loss, ratio = bppo.update(s, advantages[i], actions[i], dists[i], bppo_lr_now, clip_ratio_now, kl_logprob_a[i])
            losses.append(loss)
            ratios.append(ratio)
        return np.array(losses), np.array(ratios).mean()
    

    def behavior_update(self, iql: IQL_QV_Learner, privileged_obs: torch.Tensor, obs: torch.Tensor)-> None:

        advantages, actions, dists = [], [], []
        if not self._is_iql:
            s_value = iql._value_net(privileged_obs)

        for i in range(self._num_policy):
            dist = self.bppo_ensemble[i]._old_policy(obs)
            action = dist.rsample()
            if self._is_clip_action:
                action = action.clamp(-(1.-1e-5), 1.+1e-5)
            actions.append(action)
            if self._is_iql:
                advantage = iql.get_advantage(privileged_obs, action)
                if self._temperature:
                    print('using advantage with exp temperature')
                    advantage = torch.minimum(torch.exp(advantage * self.temperature), torch.ones_like(advantage).to(self._device)*100.0)
                advantage = (advantage - advantage.mean()) / (advantage.std() + CONST_EPS)
            else:
                advantage = iql._Q_net(privileged_obs, action) - s_value
                advantage = (advantage - advantage.mean()) / (advantage.std() + CONST_EPS)
                advantage = self.weighted_advantage(advantage)

            advantages.append(advantage)
            dists.append(dist)

        return actions, advantages, dists

    @torch.no_grad()
    def kl_update(self, iql: IQL_QV_Learner, obs: torch.Tensor, privileged_obs: torch.Tensor, kl_update, kl_strategy: str = 'sample')-> None:
        advantages, actions, dists, kl_logprob_a = [], [], [], []
        if not self._is_iql:
            s_value = iql._value_net(privileged_obs)
        policy_ids = [i_d for i_d in range(self._num_policy)]
        for i in range(self._num_policy):
            dist = self.bppo_ensemble[i]._old_policy(obs)
            dists.append(dist)
            action = dist.sample()
            if self._is_clip_action:
                action = action.clamp(-(1.-1e-5), 1.-1e-5)
            if kl_update:
                other_ids = deepcopy(policy_ids)
                del other_ids[i]
                if kl_strategy == 'sample':
                    sample_id = np.random.randint(low=0, high=len(other_ids))
                    other_dist = self.bppo_ensemble[other_ids[sample_id]]._old_policy(obs)
                    logprob_a = log_prob_func(other_dist, action)                 
                    kl_logprob_a.append(logprob_a)
                elif kl_strategy == 'max':
                    all_logprob_a = []
                    for i_d in other_ids:
                        others_dist = self.bppo_ensemble[i_d]._old_policy(obs)
                        logprob_a = log_prob_func(others_dist, action) 
                        all_logprob_a.append(logprob_a)
                    all_logprob_a_t = torch.cat(all_logprob_a, dim=1) #tensor: (batch size, num_policies - 1)
                    max_prob_a, _ = all_logprob_a_t.max(-1) # tensor: (batch_size)
                    kl_logprob_a.append(max_prob_a)
            else:
                kl_logprob_a = [0 for i_d in range(self._num_policy)]

            #action = action.clamp(-1., 1.)
            actions.append(action)
            if self._is_iql:
                advantage = iql.get_advantage(privileged_obs, action)
                if self._temperature:
                    print('using advantage with exp temperature')
                    advantage = torch.minimum(torch.exp(advantage * self._temperature), torch.ones_like(advantage).to(self._device)*100.0)
                #advantage = self.weighted_advantage(advantage)
                advantage = (advantage - advantage.mean()) / (advantage.std() + CONST_EPS)
            else:
                advantage = iql._Q_net(privileged_obs, action) - s_value
                advantage = (advantage - advantage.mean()) / (advantage.std() + CONST_EPS)
                advantage = self.weighted_advantage(advantage)

            advantages.append(advantage)

        return actions, advantages, dists, kl_logprob_a


    def replace(self, index: list) -> None:
        for i in index:
            self.bppo_ensemble[i].set_old_policy()

    def ensemble_save(self, path: str, save_id: list) -> None:
        for i in save_id:
            bc = self.bppo_ensemble[i]
            bc.save(path,i)

    def ensemble_save_body(self, path: str, save_id: list) -> None:
        for i in save_id:
            bc = self.bppo_ensemble[i]
            bc.save_body(path,i)

    def ope_dynamics_eval(self, q_eval, dynamics, eval_buffer):
        best_mean_qs =  []
        best_mean_rewards = []
        rollout_lengths = []
        for bppo in self.bppo_ensemble:
            best_mean_q, best_mean_reward, rollout_length = dynamics_eval(self._config, bppo, q_eval, dynamics, eval_buffer)
            best_mean_qs.append(best_mean_q)
            best_mean_rewards.append(best_mean_reward)
            rollout_lengths.append(rollout_length)
        return np.array(best_mean_qs), np.array(best_mean_rewards), np.array(rollout_lengths)
    
    def weighted_advantage(
        self,
        advantage: torch.Tensor
    ) -> torch.Tensor:
        if self._omega == 0.5:
            return advantage
        else:
            weight = torch.zeros_like(advantage)
            index = torch.where(advantage > 0)[0]
            weight[index] = self._omega
            weight[torch.where(weight == 0)[0]] = 1 - self._omega
            weight.to(self._device)
            return weight * advantage
        
    def load_bc(self, policy_folder: str, policy_prefix: str="pi")-> None:
        for i in range(self._num_policy):
            self.bppo_ensemble[i].load(os.path.join(policy_folder, f'{policy_prefix}_{i}.pt'))
    
# ABPPO offline learner
class ABPPO_Offline_Learner:
    def __init__(
            self, 
            device: torch.device, 
            config: FinetuneConfig, 
            abppo: AdaptiveBehaviorProximalPolicyOptimization,
            q_net: QNetwork,
            value_net: ValueNetwork,
            dynamics: BaseDynamics,
            logger: FinetuneLogger
        ):
        self._device = device
        self._config = config
        self._abppo = abppo
        self._q_net = q_net
        self._value_net = value_net
        self._logger = logger
        self._iql_learner = IQL_QV_Learner(device, q_net, value_net, config)
        if not self._config.is_iql:
            self._q_learner = QLearner(device, q_net, config)
            self._value_learner = ValueLearner(device, value_net, config)
        self._dynamics = dynamics

    def fit_q_v(self, replay_buffer: OnlineReplayBuffer):
        print("fitting q_v ......")
        value_loss, Q_loss, valid_loss = 0.0, 0.0, 0.0
        pbar = tqdm(range(int(self._config.value_update_steps)))
        if not self._config.is_iql:
            replay_buffer.compute_return(self._config.gamma)
        for step in pbar: 
            if self._config.is_iql:
                Q_loss, value_loss = self._iql_learner.update(replay_buffer=replay_buffer)
            else:
                # Q_loss = self._q_learner.update(replay_buffer=replay_buffer)
                Q_loss = 0.0
                value_loss = self._value_learner.update(replay_buffer=replay_buffer)
                if step % self._config.valid_freq == 0:
                    valid_loss = self._value_learner.valid(replay_buffer)
                    self._logger.log_update(valid_loss=valid_loss)
            if step % self._config.log_freq == 0:
                self._logger.log_update(q_loss=Q_loss, value_loss=value_loss)
            pbar.set_description(f'value loss {value_loss:.6f}, Q loss {Q_loss:.6f}, valid loss {valid_loss:.6f}')
        # if not self._config.is_iql:
        #     self._value_net.load_state_dict(self._value_learner._best_model_dict)
        #     print('best value model loaded')

    def fit_dynamics(self, replay_buffer: OnlineReplayBuffer):
        print('fitting dynamics ......')
        dynamics_loss = 0.0
        pbar = tqdm(range(int(self._config.dynamics_update_steps)), desc=f'dynamics loss {dynamics_loss:.6f}')
        for step in pbar: 
            dynamics_loss = self._dynamics.update(replay_buffer=replay_buffer)
            if step % self._config.log_freq == 0:
                self._logger.log_update(dynamics_loss=dynamics_loss)
            pbar.set_description(f'dynamics loss {dynamics_loss:.6f}')
    
    def update(self, replay_buffer: OnlineReplayBuffer):
        self.fit_q_v(replay_buffer)
        self.fit_dynamics(replay_buffer)
        if self._abppo._num_policy > 1:
            best_mean_qs, best_mean_rewards, rollout_lengths = self._abppo.ope_dynamics_eval(self._q_net, self._dynamics, replay_buffer)
            self._logger.log_update(ope_length_mean=rollout_lengths.mean(), ope_rewards_mean=best_mean_rewards.mean(), ope_Q_mean=best_mean_qs.mean(), ope_Q_std=best_mean_qs.std())
        
        print('fitting bppo ......')
        losses = np.zeros(self._config.num_policy)
        joint_losses = []
        pbar = tqdm(range(self._config.bppo_steps), desc=f'bppo loss {losses.mean():.6f}')
        for step in pbar:
            if self._config.is_linear_decay:
                bppo_lr_now = self._config.bppo_lr * (1 - step / self._config.bppo_steps)
                clip_ratio_now = self._config.clip_ratio * (1 - step / self._config.bppo_steps)
            else:
                bppo_lr_now = None
                clip_ratio_now = None
        
            losses, ratios = self._abppo.joint_train(replay_buffer, self._iql_learner, bppo_lr_now, clip_ratio_now)
            self._logger.log_update(policy_loss_mean=losses.mean(), policy_loss_std=losses.std(), bppo_lr=bppo_lr_now, clip_ratio=clip_ratio_now, ratio_mean=ratios)
            joint_losses.append(losses)
            pbar.set_description(f'bppo loss {losses.mean():.6f}')

            if (step+1) % self._config.eval_step == 0:
                if self._abppo._num_policy > 1:
                    current_mean_qs, current_mean_rewards, rollout_lengths = self._abppo.ope_dynamics_eval(self._q_net, self._dynamics, replay_buffer)
                    self._logger.log_update(ope_length_mean=rollout_lengths.mean(), ope_rewards_mean=current_mean_rewards.mean(), ope_Q_mean=current_mean_qs.mean(), ope_Q_std=current_mean_qs.std())
                    print(f"Step: {step}, rollout trajectory q mean: {current_mean_rewards}, length: {rollout_lengths}")
                    
                    index = np.where(current_mean_qs > best_mean_qs)[0]  
                    # index = np.arange(self._config.num_policy)
                    if len(index) != 0:
                        if self._config.is_update_old_policy:
                            for i_d in index:
                                self._abppo.replace(index=index)
                                print('------------------------------update behavior policy {}----------------------------------------'.format(i_d))
                                best_mean_qs[i_d] = current_mean_qs[i_d]
                        best_policy_idx = np.argmax(current_mean_qs)
                        self._abppo._policy_net.load_state_dict(self._abppo.bppo_ensemble[best_policy_idx]._policy.state_dict()) 
                else:
                    self._abppo.replace(index=[0])
                    self._abppo._policy_net.load_state_dict(self._abppo.bppo_ensemble[0]._policy.state_dict())
        return np.mean(joint_losses)