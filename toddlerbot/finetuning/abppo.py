import torch
import numpy as np
from copy import deepcopy
from toddlerbot.finetuning.networks import GaussianPolicyNetwork
from toddlerbot.finetuning.replay_buffer import OnlineReplayBuffer
from toddlerbot.finetuning.learners import ValueLearner, QLearner, IQL_Q_V
from torch.distributions.categorical import Distribution, Categorical

CONST_EPS = 1e-8

def log_prob_func(
    dist: Distribution, action: torch.Tensor
    ) -> torch.Tensor:
    log_prob = dist.log_prob(action)
    if len(log_prob.shape) == 1:
        return log_prob
    else:
        return log_prob.sum(-1, keepdim=True)


class ProximalPolicyOptimization:
    _device: torch.device
    _policy: GaussianPolicyNetwork
    _optimizer: torch.optim
    _policy_lr: float
    _old_policy: GaussianPolicyNetwork
    _scheduler: torch.optim
    _clip_ratio: float
    _entropy_weight: float
    _decay: float
    _omega: float
    _batch_size: int


    def __init__(
        self,
        device: torch.device,
        policy_net: GaussianPolicyNetwork,
        policy_lr: float,
        clip_ratio: float,
        entropy_weight: float,
        decay: float,
        omega: float,
        batch_size: int,
        is_iql: bool
    ) -> None:
        super().__init__()
        self._is_iql = is_iql
        self._device = device
        self._policy = policy_net
        #orthogonal_initWeights(self._policy)
        self._optimizer = torch.optim.Adam(
            self._policy.parameters(),
            lr=policy_lr
            )
        self._policy_lr = policy_lr
        self._old_policy = deepcopy(self._policy)
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer,
            step_size=2,
            gamma=0.98
            )
        
        self._clip_ratio = clip_ratio
        self._entropy_weight = entropy_weight
        self._decay = decay
        self._omega = omega
        self._batch_size = batch_size


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


    def loss(
        self, 
        replay_buffer: OnlineReplayBuffer,
        Q: QLearner,
        value: ValueLearner,
        is_clip_decay: bool,
        is_linear_decay, clip_ratio_now
    ) -> torch.Tensor:
        # -------------------------------------Advantage-------------------------------------
        s, a, _, _, _, _, _, advantage = replay_buffer.sample(self._batch_size)
        old_dist = self._old_policy(s)
        # -------------------------------------Advantage-------------------------------------
        new_dist = self._policy(s)
        
        new_log_prob = log_prob_func(new_dist, a)
        old_log_prob = log_prob_func(old_dist, a)
        ratio = (new_log_prob - old_log_prob).exp()
        
        advantage = self.weighted_advantage(advantage)

        loss1 =  ratio * advantage 

        if is_clip_decay:
            if is_linear_decay:
                self._clip_ratio = clip_ratio_now
            else:
                self._clip_ratio = self._clip_ratio * self._decay
        else:
            self._clip_ratio = self._clip_ratio

        loss2 = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio) * advantage 

        entropy_loss = new_dist.entropy().sum(-1, keepdim=True) * self._entropy_weight
        
        loss = -(torch.min(loss1, loss2) + entropy_loss).mean()

        return loss


    def update(
        self, 
        replay_buffer: OnlineReplayBuffer,
        Q: QLearner,
        value: ValueLearner,
        is_clip_decay: bool,
        is_lr_decay: bool,
        iql =  None,
        is_linear_decay =  None,
        bppo_lr_now =  None, 
        clip_ratio_now =  None
    ) -> float:
        policy_loss = self.loss(replay_buffer, Q, value, is_clip_decay, iql, is_linear_decay, clip_ratio_now)
        
        self._optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy.parameters(), 0.5)
        self._optimizer.step()
        
        if is_lr_decay:
            self._scheduler.step()
        if is_linear_decay:
            for p in self._optimizer.param_groups:
                p['lr'] = bppo_lr_now    
        return policy_loss.item()


    def select_action(
        self, s: torch.Tensor, is_sample: bool
    ) -> torch.Tensor:
        dist = self._policy(s)
        if is_sample:
            action = dist.sample()
        else:    
            action = dist.base_dist.mode
            for transform in dist.transforms:
                action = transform(action)
        # clip 
        action = action.clamp(-1., 1.)
        return action


    def save(
        self, path: str
    ) -> None:
        torch.save(self._policy.state_dict(), path)
        print('Policy parameters saved in {}'.format(path))
    

    def load(
        self, path: str
    ) -> None:
        self._policy.load_state_dict(torch.load(path, map_location=self._device))
        self._old_policy.load_state_dict(self._policy.state_dict())
        #self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=self._policy_lr)
        print('Policy parameters loaded')

    def set_old_policy(
        self,
    ) -> None:
        self._old_policy.load_state_dict(self._policy.state_dict())

class BehaviorProximalPolicyOptimization(ProximalPolicyOptimization):

    def __init__(
        self,
        device: torch.device,
        policy_net: GaussianPolicyNetwork,
        policy_lr: float,
        clip_ratio: float,
        entropy_weight: float,
        decay: float,
        omega: float,
        batch_size: int,
        is_iql: bool,
        temperature: float = None
    ) -> None:
        super().__init__(
            device = device,
            policy_net = policy_net,
            policy_lr = policy_lr,
            clip_ratio = clip_ratio,
            entropy_weight = entropy_weight,
            decay = decay,
            omega = omega,
            batch_size = batch_size,
            is_iql=is_iql)
        self.temperature = temperature

    def loss(
        self, 
        s: torch.Tensor,
        advantage: torch.Tensor,
        a: torch.Tensor,
        old_dist: torch.Tensor,
        is_clip_decay: bool,
        is_linear_decay: bool =  None,
        clip_ratio_now: float =  None,
        kl_update: bool = False,
        kl_alpha: float = None,
        kl_logprob_a: torch.Tensor = None
    ) -> torch.Tensor:

        new_dist = self._policy(s)

        new_log_prob = log_prob_func(new_dist, a)
        old_log_prob = log_prob_func(old_dist, a)
        ratio = (new_log_prob - old_log_prob).exp()
        
        loss1 =  ratio * advantage 

        if is_clip_decay:
            if is_linear_decay:
                self._clip_ratio = clip_ratio_now
            else:
                self._clip_ratio = self._clip_ratio * self._decay
        else:
            self._clip_ratio = self._clip_ratio

        loss2 = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio) * advantage 
        
        entropy_loss = new_dist.entropy().sum(-1, keepdim=True) * self._entropy_weight
        
        loss = -(torch.min(loss1, loss2) + entropy_loss)

        if kl_update:
            kl_loss = - kl_alpha * (new_log_prob - kl_logprob_a.detach())
            loss = loss + kl_loss

        return loss.mean()

    def update(
        self, 
        s: torch.Tensor,
        advantage: torch.Tensor,
        action: torch.Tensor,
        old_dist: torch.Tensor,
        is_clip_decay: bool,
        is_lr_decay: bool,
        is_linear_decay: bool =  None,
        bppo_lr_now: float =  None, 
        clip_ratio_now: float =  None,
        kl_update: bool = False,
        kl_alpha: float = None,
        kl_logprob_a: torch.Tensor = None
    ) -> float:
        policy_loss = self.loss(s, advantage, action, old_dist, is_clip_decay, is_linear_decay, clip_ratio_now, kl_update, kl_alpha, kl_logprob_a)
        
        self._optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy.parameters(), 0.5)
        self._optimizer.step()
        
        if is_lr_decay:
            self._scheduler.step()
        if is_linear_decay:
            for p in self._optimizer.param_groups:
                p['lr'] = bppo_lr_now    
        return policy_loss.item()


class AdaptiveBehaviorProximalPolicyOptimization():
    def __init__(
        self,
        device: torch.device,
        policy_net: GaussianPolicyNetwork,
        policy_lr: float,
        clip_ratio: float,
        entropy_weight: float,
        decay: float,
        omega: float,
        batch_size: int,
        belief_dim: int,
        discount: float,
        is_iql: bool,
        kl_update: bool,
        kl_strategy: str,
        alpha_bppo: float,
        is_clip_action: float =False, 
        temperature: float = None
    ) -> None:
        self._device = device
        self._policy_lr = policy_lr
        self._clip_ratio= clip_ratio
        self._entropy_weight = entropy_weight
        self._decay = decay
        self._omega = omega
        self._discount = discount
        self._batch_size = batch_size
        self._num_policy = belief_dim
        self._is_iql = is_iql
        self._kl_update = kl_update
        self._kl_strategy = kl_strategy
        self._alpha = alpha_bppo
        self._is_clip_action = is_clip_action
        self._temperature = temperature
        self.bppo_ensemble = []
        for _ in range(belief_dim):
            bppo = BehaviorProximalPolicyOptimization(device, deepcopy(policy_net), policy_lr, clip_ratio, entropy_weight, decay, omega, batch_size, is_iql)
            self.bppo_ensemble.append(bppo)
    

    def joint_train(self, 
        replay_buffer: OnlineReplayBuffer,
        value: ValueLearner,
        is_clip_decay: bool, 
        is_lr_decay: bool,
        is_linear_decay: bool,
        bppo_lr_now: float = None,
        clip_ratio_now: float = None, 
        Q: QLearner = None,
        iql: IQL_Q_V = None
        ) -> None:
        _, s, _, _, _, _, _, _, _, _ = replay_buffer.sample(self._batch_size)
        
        actions, advantages, dists, kl_logprob_a = self.kl_update(Q, value, iql, s, self._kl_update, self._kl_strategy)
        losses = []
        for i, bppo in enumerate(self.bppo_ensemble):

            loss = bppo.update(s, advantages[i], actions[i], dists[i], is_clip_decay, is_lr_decay, is_linear_decay, bppo_lr_now, clip_ratio_now, self._kl_update, self._alpha, kl_logprob_a[i])
            losses.append(loss)

        return np.array(losses)
    

    def behavior_update(self, Q, value, iql, s: torch.Tensor)-> None:

        advantages, actions, dists = [], [], []
        if not self._is_iql:
            s_value = value(s)

        for i in range(self._num_policy):
            dist = self.bppo_ensemble[i]._old_policy(s)
            action = dist.rsample()
            if self._is_clip_action:
                action = action.clamp(-(1.-1e-5), 1.+1e-5)
            actions.append(action)
            if self._is_iql:
                advantage = iql.get_advantage(s, action)
                if self._temperature:
                    print('using advantage with exp temperature')
                    advantage = torch.minimum(torch.exp(advantage * self.temperature), torch.ones_like(advantage).to(self._device)*100.0)
                advantage = (advantage - advantage.mean()) / (advantage.std() + CONST_EPS)
            else:
                advantage = Q(s, action) - s_value
                advantage = (advantage - advantage.mean()) / (advantage.std() + CONST_EPS)
                advantage = self.weighted_advantage(advantage)

            advantages.append(advantage)
            dists.append(dist)

        return actions, advantages, dists

    def kl_update(self, Q, value, iql, s: torch.Tensor, kl_update, kl_strategy: str = 'sample')-> None:
        with torch.no_grad():
            advantages, actions, dists, kl_logprob_a = [], [], [], []
            if not self._is_iql:
                s_value = value(s)
            policy_ids = [i_d for i_d in range(self._num_policy)]
            for i in range(self._num_policy):
                dist = self.bppo_ensemble[i]._old_policy(s)
                dists.append(dist)
                action = dist.rsample()
                if self._is_clip_action:
                    action = action.clamp(-(1.-1e-5), 1.+1e-5)
                if kl_update:
                    other_ids = deepcopy(policy_ids)
                    del other_ids[i]
                    if kl_strategy == 'sample':
                        sample_id = np.random.randint(low=0, high=len(other_ids))
                        other_dist = self.bppo_ensemble[other_ids[sample_id]]._old_policy(s)
                        logprob_a = log_prob_func(other_dist, action)                 
                        kl_logprob_a.append(logprob_a)
                    elif kl_strategy == 'max':
                        all_logprob_a = []
                        for i_d in other_ids:
                            others_dist = self.bppo_ensemble[i_d]._old_policy(s)
                            logprob_a = log_prob_func(others_dist, action) 
                            all_logprob_a.append(logprob_a)
                        all_logprob_a_t = torch.cat(all_logprob_a,dim=1) #tensor: (batch size, num_policies - 1)
                        max_prob_a, _ = all_logprob_a_t.max(-1) # tensor: (batch_size)
                        kl_logprob_a.append(max_prob_a)
                else:
                    kl_logprob_a = [0 for i_d in range(self._num_policy)]

                #action = action.clamp(-1., 1.)
                actions.append(action)
                if self._is_iql:
                    advantage = iql.get_advantage(s, action)
                    if self._temperature:
                        print('using advantage with exp temperature')
                        advantage = torch.minimum(torch.exp(advantage * self.temperature), torch.ones_like(advantage).to(self._device)*100.0)
                    #advantage = self.weighted_advantage(advantage)
                    advantage = (advantage - advantage.mean()) / (advantage.std() + CONST_EPS)
                else:
                    advantage = Q(s, action) - s_value
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

    def ope_dynamics_eval(self, args, dynamics_eval, q_eval, dynamics, eval_buffer,  mean, std):
        best_mean_qs =  []
        for bppo in self.bppo_ensemble:
            best_mean_q, _ = dynamics_eval(args, bppo, q_eval, dynamics, eval_buffer,  mean, std)
            best_mean_qs.append(best_mean_q)
        return np.array(best_mean_qs)
    
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