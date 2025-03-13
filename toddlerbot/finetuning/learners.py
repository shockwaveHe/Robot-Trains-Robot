from copy import deepcopy
import torch
import torch.nn.functional as F
from tqdm import tqdm

from toddlerbot.finetuning.networks import ValueNetwork, QNetwork, DoubleQNetwork, DynamicsNetwork
from toddlerbot.finetuning.replay_buffer import OnlineReplayBuffer

import torch.nn as nn

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from toddlerbot.finetuning.finetune_config import FinetuneConfig

class LinearCosineScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, decay_steps: int, last_epoch: int = -1):
        """
        A scheduler that linearly increases the learning rate for a specified number of steps 
        and then decreases it following a cosine schedule.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            total_steps (int): Total number of training steps.
            warmup_steps (int): Number of steps for the linear warmup phase.
            last_epoch (int): The index of the last epoch. Default: -1.
        """
        self.total_steps = decay_steps + warmup_steps
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup phase
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return [base_lr * 0.5 * (1 + math.cos(math.pi * progress)) for base_lr in self.base_lrs]


class ValueLearner:
    def __init__(
        self, 
        device: torch.device, 
        value_net: ValueNetwork,
        config: FinetuneConfig
    ) -> None:
        super().__init__()
        self._device = device
        self._value = value_net.to(device)
        self._value_opt = torch.compile(self._value)
        self._optimizer = torch.optim.Adam(
            self._value.parameters(), 
            lr=config.value_lr,
            )
        # self._scheduler = LinearCosineScheduler(self._optimizer, config.warmup_steps, config.decay_steps)
        self._batch_size = config.value_batch_size
        self._best_valid_loss = float('inf')
        self._best_model_dict = deepcopy(self._value.state_dict())


    def __call__(
        self, s: torch.Tensor
    ) -> torch.Tensor:
        return self._value(s)


    def update(
        self, replay_buffer: OnlineReplayBuffer
    ) -> float:
        _, s, _, _, _, _, _, _, _, Return, _ = replay_buffer.sample(self._batch_size)
        value_loss = F.mse_loss(self._value_opt(s).squeeze(), Return.squeeze())
        # import ipdb; ipdb.set_trace()
        self._optimizer.zero_grad()
        value_loss.backward()
        self._optimizer.step()
        # self._scheduler.step()

        return value_loss.item()

    def valid(
        self, replay_buffer: OnlineReplayBuffer
    ) -> float:
        valid_losses = []
        for _ in range(100):
            _, s, _, _, _, _, _, _, _, Return, _ = replay_buffer.sample(self._batch_size, sample_validation=True)
            value_loss = F.mse_loss(self._value(s).squeeze(), Return.squeeze())
            valid_losses.append(value_loss.item())
        valid_loss = sum(valid_losses) / len(valid_losses)
        if valid_loss < self._best_valid_loss:
            self._best_valid_loss = valid_loss
            self._best_model_dict = deepcopy(self._value.state_dict())
            print(f'Best model updated with valid loss {valid_loss}')
        return valid_loss

    def save(
        self, path: str
    ) -> None:
        torch.save(self._value.state_dict(), path)
        print('Value parameters saved in {}'.format(path))


    def load(
        self, path: str
    ) -> None:
        self._value.load_state_dict(torch.load(path, map_location=self._device))
        print('Value parameters loaded')



class QLearner:
    def __init__(
        self,
        device: torch.device,
        Q_net: QNetwork,
        config: FinetuneConfig
    ) -> None:
        super().__init__()
        self._device = device
        self._Q = Q_net.to(device)
        self._optimizer = torch.optim.Adam(
            self._Q.parameters(),
            lr=config.Q_lr,
            )

        self._target_Q = deepcopy(self._Q)
        self._target_Q.load_state_dict(self._Q.state_dict())
        self._total_update_step = 0
        self._target_update_freq = config.target_update_freq
        self._tau = config.tau

        self._gamma = config.gamma
        self._batch_size = config.value_batch_size
        self._best_valid_loss = float('inf')


    def __call__(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> torch.Tensor:
        return self._Q(s, a)


    def update(
        self, replay_buffer: OnlineReplayBuffer
    ) -> float:
        _, s, a, r, _, s_n, a_n, term, _, Return, _ = replay_buffer.sample(self._batch_size)
        # with torch.no_grad():
        #     target_Q = r.squeeze() + (1 - term.squeeze()) * self._gamma * self._target_Q(s_n, a_n)
        Q = self._Q(s, a).squeeze()
        Q_loss = F.mse_loss(Q, Return.squeeze())
        self._optimizer.zero_grad()
        Q_loss.backward()
        self._optimizer.step()

        self._total_update_step += 1
        if self._total_update_step % self._target_update_freq == 0:
            for param, target_param in zip(self._Q.parameters(), self._target_Q.parameters()):
                target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

        return Q_loss.item()

    def valid(
        self, replay_buffer: OnlineReplayBuffer
    ) -> float:
        valid_losses = []
        for _ in range(100):
            _, s, a, r, _, s_n, a_n, term, _, Return, _ = replay_buffer.sample(self._batch_size, sample_validation=True)
            with torch.no_grad():
                Q = self._Q(s, a).squeeze()
                Q_loss = F.mse_loss(Q, Return.squeeze())
                valid_losses.append(Q_loss.item())
        valid_loss = sum(valid_losses) / len(valid_losses)
        if valid_loss < self._best_valid_loss:
            self._best_valid_loss = valid_loss
            self._best_model_dict = deepcopy(self._Q.state_dict())
            print(f'Best model updated with valid loss {valid_loss}')
        return valid_loss

    def save(
        self, path: str
    ) -> None:
        torch.save(self._Q.state_dict(), path)
        print('Q function parameters saved in {}'.format(path))
    

    def load(
        self, path: str
    ) -> None:
        self._Q.load_state_dict(torch.load(path, map_location=self._device))
        self._target_Q.load_state_dict(self._Q.state_dict())
        print('Q function parameters loaded')


    
class IQL_QV_Learner:
    def __init__(
        self,
        device: torch.device,
        Q_net: QNetwork | DoubleQNetwork,
        value_net: ValueNetwork,
        config: FinetuneConfig
    ) -> None:
        
        self._device = device
        self._omega = config.omega
        self._is_double_q = config.use_double_q
        #for q
        self._Q_net = Q_net.to(device)
        self._Q_target = deepcopy(self._Q_net).to(device)
        self._q_optimizer = torch.optim.Adam(
            self._Q_net.parameters(),
            lr=config.Q_lr,
            )
        
        self._Q_target.load_state_dict(self._Q_net.state_dict())
        self._total_update_step = 0
        self._target_update_freq = config.target_update_freq
        self._tau = config.tau
        self._gamma = config.gamma
        self._batch_size = config.value_batch_size
        #for v
        self._value_net = value_net.to(device)
        self._v_optimizer = torch.optim.Adam(
            self._value_net.parameters(), 
            lr=config.value_lr,
            )
        self.best_q_loss = float('inf')
        self.best_v_loss = float('inf')
        self._best_q_state = deepcopy(self._Q_net.state_dict())
        self._best_v_state = deepcopy(self._value_net.state_dict())

    def expectile_loss(self, loss: torch.Tensor)->torch.Tensor:
        weight = torch.where(loss > 0, self._omega, (1 - self._omega))
        return weight * (loss**2)
    
    def update(self, replay_buffer: OnlineReplayBuffer) -> float:
        _, s, a, r, _, s_n, _, term, _, _, _ = replay_buffer.sample(self._batch_size)
        # Compute value loss
        with torch.no_grad():
            self._Q_target.eval()
            target_q = self._Q_target(s, a)
        value = self._value_net(s)
        value_loss = self.expectile_loss(target_q - value).mean()

        #update v
        self._v_optimizer.zero_grad()
        value_loss.backward()
        self._v_optimizer.step()

        # Compute critic loss
        with torch.no_grad():
            self._value_net.eval()
            next_v = self._value_net(s_n)
            
        target_q = r + (1 - term) * self._gamma * next_v
        if self._is_double_q: 
            current_q1, current_q2 = self._Q_net(s, a, return_min=False)
            q_loss = ((current_q1 - target_q)**2 + (current_q2 - target_q)**2).mean()
        else:
            Q = self._Q_net(s, a)
            q_loss = F.mse_loss(Q, target_q)

        #update q and target q
        self._q_optimizer.zero_grad()
        q_loss.backward()
        self._q_optimizer.step()

        self._total_update_step += 1
        if self._total_update_step % self._target_update_freq == 0:
            for param, target_param in zip(self._Q_net.parameters(), self._Q_target.parameters()):
                target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)


        return q_loss.item(), value_loss.item()
    
    @torch.no_grad()
    def valid(self, replay_buffer: OnlineReplayBuffer) -> float:
        valid_q_losses = []
        valid_v_losses = []
        for _ in range(100):
            _, s, a, r, _, s_n, _, term, _, _, _ = replay_buffer.sample(self._batch_size, sample_validation=True)
            # Compute value loss
            self._Q_net.eval()
            self._value_net.eval()
            with torch.no_grad():
                target_q = self._Q_net(s, a)
                value = self._value_net(s)
                value_loss = self.expectile_loss(target_q - value).mean()
                valid_v_losses.append(value_loss.item())
            # Compute critic loss
            with torch.no_grad():
                next_v = self._value_net(s_n)
                target_q = r + (1 - term) * self._gamma * next_v
                if self._is_double_q: 
                    current_q1, current_q2 = self._Q_net(s, a, return_min=False)
                    q_loss = ((current_q1 - target_q)**2 + (current_q2 - target_q)**2).mean()
                else:
                    Q = self._Q_net(s, a)
                    q_loss = F.mse_loss(Q, target_q)
                valid_q_losses.append(q_loss.item())
        valid_q_loss = sum(valid_q_losses) / len(valid_q_losses)
        valid_v_loss = sum(valid_v_losses) / len(valid_v_losses)
        if valid_q_loss < self.best_q_loss:
            self.best_q_loss = valid_q_loss
            self._best_q_state = deepcopy(self._Q_net.state_dict())
            print(f'Best Q model updated with valid loss {valid_q_loss}')
        if valid_v_loss < self.best_v_loss:
            self.best_v_loss = valid_v_loss
            self._best_v_state = deepcopy(self._value_net.state_dict())
            print(f'Best V model updated with valid loss {valid_v_loss}')
        return valid_q_loss, valid_v_loss
    
    def get_advantage(self, s, a)->torch.Tensor:
        return self._Q_net(s, a) - self._value_net(s)
    
    def save(
        self, q_path: str, v_path: str
    ) -> None:
        torch.save(self._Q_net.state_dict(), q_path)
        print('Q function parameters saved in {}'.format(q_path))
        torch.save(self._value_net.state_dict(), v_path)
        print('Value parameters saved in {}'.format(v_path))

    def load(
        self, q_path: str, v_path: str
    ) -> None:
        self._Q_net.load_state_dict(torch.load(q_path, map_location=self._device))
        self._Q_target.load_state_dict(self._Q_net.state_dict())
        print('Q function parameters loaded')
        self._value_net.load_state_dict(torch.load(v_path, map_location=self._device))
        print('Value parameters loaded')


if __name__ == '__main__':
    import pickle
    with open('buffer.pkl', 'rb') as f:
        buffer = pickle.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    buffer._device = device
    buffer.compute_return(0.99)
    config = FinetuneConfig()
    value_net = ValueNetwork(buffer._privileged_obs.shape[1], lambda x, y: x, (512, 256, 128))
    q_net = DoubleQNetwork(buffer._privileged_obs.shape[1], buffer._action.shape[1], lambda x, y: x, (512, 256, 128))
    iql_learner = IQL_QV_Learner(device, q_net, value_net, config)

    
    from tqdm import tqdm
    for i in tqdm(range(10000)):
        q_loss, value_loss = iql_learner.update(buffer)