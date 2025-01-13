from dataclasses import dataclass
from typing import Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from pathlib import Path
from torch.distributions import Normal, TanhTransform
from torch.distributions.transformed_distribution import TransformedDistribution

def load_jax_params(path: str) -> Any:
    with Path(path).open('rb') as fin:
        buf = fin.read()
    # params[0] is running statistics, params[1] is the actual parameters
    return pickle.loads(buf)

def load_jax_params_into_pytorch(pt_model: torch.nn.Module, jax_params: dict):
    """
    Copies parameters from a JAX parameter dictionary into the corresponding 
    PyTorch model. This function assumes that pt_model is an MLP-like structure 
    with a series of Linear layers in the same order as JAX's 'hidden_0', 'hidden_1', etc.
    """
    # Extract layers from the PyTorch model.
    # We assume a Sequential or a Module with ordered layers that correspond to the JAX layers.
    # If your model is structured differently, adapt this accordingly.
    linear_layers = [m for m in pt_model.modules() if isinstance(m, torch.nn.Linear)]
    
    with torch.no_grad():
        for i, layer in enumerate(linear_layers):
            # For each linear layer in PyTorch, find the corresponding hidden_i block in JAX params
            kernel_key = f"hidden_{i}"
            jax_kernel = jax_params[kernel_key]['kernel']  # shape (in_dim, out_dim)
            jax_bias = jax_params[kernel_key]['bias']      # shape (out_dim,)
            
            # Convert to torch tensors
            w = torch.from_numpy(np.array(jax_kernel)).T  # transpose to (out_dim, in_dim)
            b = torch.from_numpy(np.array(jax_bias))
            
            layer.weight.copy_(w)
            layer.bias.copy_(b)

def soft_clamp(
    x: torch.Tensor, bound: tuple
    ) -> torch.Tensor:
    low, high = bound
    #x = torch.tanh(x)
    x = low + 0.5 * (high - low) * (x + 1)
    return x


class MLP(nn.Module):
    def __init__(self, layer_sizes, activation, activate_final=False, layer_norm=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if layer_norm else None
        self.activate_final = activate_final
        self.activation = activation
        for i in range(len(layer_sizes)-1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            linear = nn.Linear(in_dim, out_dim, bias=True)
            # In Flax's lecun_uniform initialization, variance is scaled by fan_in.
            # PyTorch's default init is Kaiming uniform which is similar but not identical.
            # For testing, we can just rely on parameters being copied from JAX rather than new init.
            self.layers.append(linear)
            if layer_norm:
                self.layer_norms.append(nn.LayerNorm(out_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers)-1 or self.activate_final:
                x = self.activation(x)
                if self.layer_norms is not None:
                    x = self.layer_norms[i](x)
        return x
    

class GaussianPolicyNetwork(nn.Module):
    def __init__(
        self, observation_size: int, hidden_layers: Tuple[int], action_size: int, preprocess_observations_fn, activation: str = 'elu', para_std = True
    ) -> None:
        super().__init__()
        self.para_std = para_std
        self.preprocess_observations_fn = preprocess_observations_fn
        self.mlp = MLP([observation_size] + list(hidden_layers) + [action_size * 2], activation=activation, layer_norm=False, activate_final=False)
        self._log_std_bound = (-10., 2.)

        
    def forward(
        self, obs: torch.Tensor, processer_params=None
    ) -> torch.distributions.transformed_distribution.TransformedDistribution:
        obs = self.preprocess_observations_fn(obs, processer_params)
        mu, log_std = self.mlp(obs).chunk(2, dim=-1)
        log_std = soft_clamp(log_std, self._log_std_bound)
        std = log_std.exp()
        dist = Normal(mu, std)
        dist = TransformedDistribution(dist, [TanhTransform(cache_size=1)])
        return dist
    
class ValueNetwork(nn.Module):
    def __init__(self, observation_size, preprocess_observations_fn, hidden_layers, activation='swish'):
        super().__init__()
        self.preprocess_observations_fn = preprocess_observations_fn
        self.mlp = MLP([observation_size] + list(hidden_layers) + [1],
                              activation=activation,
                              layer_norm=False,
                              activate_final=False)

    def forward(self, obs, processer_params=None):
        obs = self.preprocess_observations_fn(obs, processer_params)
        return self.mlp(obs).squeeze(-1)

class QNetwork(nn.Module):
    def __init__(self, observation_size, action_size, preprocess_observations_fn, hidden_layers, activation='swish'):
        super().__init__()
        self.preprocess_observations_fn = preprocess_observations_fn
        self.mlp = MLP([observation_size + action_size] + list(hidden_layers) + [1],
                              activation=activation,
                              layer_norm=False,
                              activate_final=False)

    def forward(self, obs, action, processer_params=None):
        obs = self.preprocess_observations_fn(obs, processer_params)
        x = torch.cat([obs, action], dim=-1)
        return self.mlp(x).squeeze(-1)
    
class DoubleQNetwork(nn.Module):
    def __init__(self, observation_size, action_size, preprocess_observations_fn, hidden_layers, activation='swish'):
        super().__init__()
        self.q1 = QNetwork(observation_size, action_size, preprocess_observations_fn, hidden_layers, activation)
        self.q2 = QNetwork(observation_size, action_size, preprocess_observations_fn, hidden_layers, activation)

    def forward(self, obs, action, processer_params=None):
        return self.q1(obs, action, processer_params), self.q2(obs, action, processer_params)

class DynamicsNetwork(nn.Module):
    def __init__(self, observation_size, action_size, preprocess_observations_fn, hidden_layers, activation='swish'):
        super().__init__()
        self.preprocess_observations_fn = preprocess_observations_fn
        self.mlp = MLP([observation_size + action_size] + list(hidden_layers) + [observation_size],
                              activation=activation,
                              layer_norm=False,
                              activate_final=False)

    def forward(self, obs, action, processer_params=None):
        obs = self.preprocess_observations_fn(obs, processer_params)
        x = torch.cat([obs, action], dim=-1)
        return self.mlp(x)