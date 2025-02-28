from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import numpy as np
from tqdm import tqdm
import os
from toddlerbot.finetuning.networks import GaussianPolicyNetwork, ValueNetwork
from toddlerbot.finetuning.replay_buffer import OnlineReplayBuffer
from toddlerbot.finetuning.finetune_config import FinetuneConfig
from toddlerbot.finetuning.logger import FinetuneLogger

class PPO:
    def __init__(
        self, 
        device: torch.device,
        config: FinetuneConfig,
        policy_net: GaussianPolicyNetwork,
        value_net: ValueNetwork,
        logger: FinetuneLogger
    ):
        self.batch_size = config.online.batch_size
        self.mini_batch_size = config.online.mini_batch_size
        self.max_train_step = config.online.max_train_step
        self.lr_a = config.online.lr_a  # Learning rate of actor
        self.lr_c = config.online.lr_c  # Learning rate of critic
        self.gamma = config.online.gamma  # Discount factor
        self.lamda = config.online.lamda  # GAE parameter
        self.epsilon = config.online.epsilon  # PPO clip parameter
        self.K_epochs = config.online.K_epochs  # PPO parameter
        self.entropy_coef = config.online.entropy_coef  # Entropy coefficient
        self.set_adam_eps = config.online.set_adam_eps
        self.use_grad_clip = config.online.use_grad_clip
        self.use_lr_decay = config.online.use_lr_decay
        self.use_adv_norm = config.online.use_adv_norm
        self.is_clip_value = config.online.is_clip_value
        self.device = device
        self.has_set_critic = False
        self._config = config
        self._device = device
        self._logger = logger # TODO: improve logging, seperate online offline?

        self._policy_net = deepcopy(policy_net).to(self.device)
        # if args.scale_strategy == 'dynamic' or args.scale_strategy == 'number': # DISCUSS
        #     self.critic = ValueReluMLP(args).to(self.device)
        # else:
        self._value_net = value_net.to(self.device)
        
        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self._policy_net.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self._value_net.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self._policy_net.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self._value_net.parameters(), lr=self.lr_c)

    def evaluate_value(self, replay_buffer: OnlineReplayBuffer, steps = 20000):
        mean_value = []
        for _ in tqdm(range(steps), desc='check buffer value'):
            _, s_p, _, _, _, _, _, _, _, Return, _ = replay_buffer.sample(512)
            value = self._value_net(s_p)
            mean_value.append(torch.mean(value.cpu().detach()).item())

        print('mean value score: {}'.format(np.mean(mean_value)))

    def load(self, path: str) -> None:
        policy_path = os.path.join(path, 'policy_net.pt')
        value_path = os.path.join(path, 'value_net.pt')
        self._policy_net.load_state_dict(torch.load(policy_path, map_location=self._device))
        self._value_net.load_state_dict(torch.load(value_path, map_location=self.device))


    def set_critic(self, value_net: ValueNetwork):
        if not self.has_set_critic:
            self._value_net.load_state_dict(value_net.state_dict())
            self.has_set_critic = True
            print('Successfully set critic from pretraining')

    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        
        a = self._policy_net(s).detach().cpu().numpy().flatten()
        return a
    
    def get_action(self, s, deterministic=False):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        
        with torch.no_grad():
            dist = self._policy_net(s)
            if deterministic:
                a = dist.mean
                a_logprob = dist.log_prob(a)
            else:
                a = dist.sample()  # Sample the action according to the probability distribution
                a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()

    def update(self, replay_buffer: OnlineReplayBuffer, current_steps):
        s, sp, a, r, s_, sp_, _, terms, truncs, _, a_logprob_old = replay_buffer.sample(self.batch_size)  # Get training data
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self._value_net(sp)
            vs_ = self._value_net(sp_)
            deltas = r + self.gamma * (1.0 - terms) * vs_ - vs
            for delta, term, trunc in zip(reversed(deltas.flatten().cpu().numpy()), reversed(terms.flatten().cpu().numpy()), reversed(truncs.flatten().cpu().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - term) * (1.0 - trunc)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(self.device)
            v_target = adv.flatten() + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        actor_losses, critic_losses = [], []
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                new_dist = self._policy_net(s[index])
                dist_entropy = new_dist.base_dist.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                # TODO: entropy modification for tanh
                a_logprob_now = new_dist.log_prob(a[index])

                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob_old[index])  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                actor_losses.append(actor_loss.mean().item())
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self._policy_net.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self._value_net(sp[index])
                if self.is_clip_value:
                    old_value_clipped = vs[index] + (v_s - vs[index]).clamp(-self.epsilon, self.epsilon)
                    value_loss = (v_s - v_target[index].detach().float()).pow(2)
                    value_loss_clipped = (old_value_clipped - v_target[index].detach().float()).pow(2)
                    critic_loss = torch.max(value_loss,value_loss_clipped).mean()
                else:
                    critic_loss = F.mse_loss(v_target[index], v_s)
                critic_losses.append(critic_loss.mean().item())
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self._value_net.parameters(), 0.5)
                self.optimizer_critic.step()
                self._logger.log_update(
                    a_logprob_now=a_logprob_old[index].mean().item(),
                    ratios=ratios.mean().item(),
                    adv=adv[index].mean().item(),
                    v_s=v_s.mean().item(),
                    actor_loss=actor_loss.mean().item(),
                    dist_entropy=dist_entropy.mean().item(),
                    critic_loss=critic_loss.mean().item()
                )

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(current_steps)
        import ipdb; ipdb.set_trace()
        replay_buffer.reset()
        return np.mean(actor_losses), np.mean(critic_losses)
    
    def lr_decay(self, current_steps):
        # TODO: decay by max train steps or train steps per iteration
        lr_a_now = self.lr_a * (1 - current_steps / self.max_train_step)
        lr_c_now = self.lr_c * (1 - current_steps / self.max_train_step)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
