import torch
from copy import deepcopy

from toddlerbot.finetuning.networks import GaussianPolicyNetwork
from toddlerbot.finetuning.replay_buffer import OnlineReplayBuffer
from toddlerbot.finetuning.finetune_config import FinetuneConfig
from torch.distributions import Distribution
from torch.distributions.transformed_distribution import TransformedDistribution

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
        config: FinetuneConfig,
    ) -> None:
        super().__init__()
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
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer,
            step_size=2,
            gamma=0.98
            )
        
        self._clip_ratio = config.clip_ratio
        self._entropy_weight = config.entropy_weight
        self._decay = config.decay
        self._omega = config.omega
        self._batch_size = config.policy_batch_size
        self._config = config


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
            for transform in reversed(new_dist.transforms):
                pre_tanh_sample = transform.inv(pre_tanh_sample)
        log_det_jac = self._policy.forward_log_det_jacobian(pre_tanh_sample)
        entropy_loss = torch.sum(new_dist.base_dist.entropy() + log_det_jac, dim=-1) * self._entropy_weight
        return entropy_loss
    
    def loss(
        self, 
        replay_buffer: OnlineReplayBuffer,
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

        entropy_loss = self.get_entropy_loss(new_dist)
        
        loss = -(torch.min(loss1, loss2) + entropy_loss).mean()

        return loss


    def update(
        self, 
        replay_buffer: OnlineReplayBuffer,
        is_clip_decay: bool,
        is_lr_decay: bool,
        iql =  None,
        is_linear_decay =  None,
        bppo_lr_now =  None, 
        clip_ratio_now =  None
    ) -> float:
        policy_loss = self.loss(replay_buffer, is_clip_decay, iql, is_linear_decay, clip_ratio_now)
        
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
        s.to(self._device)
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
