from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from pathlib import Path

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
    
    
class TanhBijector:
    def forward(self, x):
        return torch.tanh(x)

    def inverse(self, y):
        # atanh: 0.5 * log((1+y)/(1-y))
        # For numerical stability, use torch.atanh if available (PyTorch 2.0+), else implement manually:
        return 0.5 * torch.log((1+y)/(1-y))

    def forward_log_det_jacobian(self, x):
        # Matches JAX implementation
        # 2 * (log(2) - x - softplus(-2x))
        return 2.0 * (torch.log(torch.tensor(2.0)) - x - F.softplus(-2.0 * x))


class NormalDistribution:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def sample(self):
        eps = torch.randn_like(self.loc)
        return eps * self.scale + self.loc

    def mode(self):
        return self.loc

    def log_prob(self, x):
        # same formula as JAX
        log_unnormalized = -0.5 * ((x - self.loc) / self.scale)**2
        log_normalization = 0.5 * np.log(2.0 * np.pi) + torch.log(self.scale)
        return log_unnormalized - log_normalization

    def entropy(self):
        log_normalization = 0.5 * np.log(2.0 * np.pi) + torch.log(self.scale)
        return (0.5 + log_normalization) * torch.ones_like(self.loc)


class NormalTanhDistribution:
    def __init__(self, event_size, min_std=0.001, var_scale=1.0):
        self.min_std = min_std
        self.var_scale = var_scale
        self.bijector = TanhBijector()
        self.event_size = event_size
        self.param_size = 2 * event_size

    def create_dist(self, parameters):
        loc, scale = torch.chunk(parameters, 2, dim=-1)
        scale = (F.softplus(scale) + self.min_std) * self.var_scale
        return NormalDistribution(loc=loc, scale=scale)

    def sample(self, parameters):
        dist = self.create_dist(parameters)
        pre_tanh = dist.sample()
        return self.bijector.forward(pre_tanh)

    def sample_no_postprocessing(self, parameters):
        dist = self.create_dist(parameters)
        return dist.sample()
    
    def mode(self, parameters):
        dist = self.create_dist(parameters)
        pre_tanh_mode = dist.mode()
        return self.bijector.forward(pre_tanh_mode)

    def log_prob(self, parameters, raw_actions):
        dist = self.create_dist(parameters)
        # pre_tanh_actions = self.bijector.inverse(actions)
        log_prob_x = dist.log_prob(raw_actions)
        log_prob_x -= self.bijector.forward_log_det_jacobian(raw_actions)
        # sum over the last dimension
        if self._event_ndims == 1:
            log_prob_x = torch.sum(log_prob_x, dim=-1)
        return log_prob_x

    def postprocess(self, raw_actions):
        return self.bijector.forward(raw_actions)

    def entropy(self, parameters):
        # to mimic JAX entropy calculation, we sample once and estimate
        # for deterministic comparison, let's sample with a fixed seed:
        dist = self.create_dist(parameters)
        base_entropy = dist.entropy()
        pre_tanh_sample = dist.sample()
        base_entropy += self.bijector.forward_log_det_jacobian(pre_tanh_sample)
        if self._event_ndims == 1:
            base_entropy = torch.sum(base_entropy, axis=-1)
        return base_entropy
    

class PolicyNetwork(nn.Module):
    def __init__(self, observation_size, hidden_layers, output_size, preprocess_observations_fn, activation, layer_norm=False):
        super().__init__()
        self.preprocess_observations_fn = preprocess_observations_fn
        # Construct the MLP with hidden_layers + [output_size]
        self.mlp = MLP([observation_size] + list(hidden_layers) + [output_size],
                              activation=activation,
                              layer_norm=layer_norm,
                              activate_final=False)

    def forward(self, obs, processer_params=None):
        obs = self.preprocess_observations_fn(obs, processer_params)
        return self.mlp(obs)


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

@dataclass 
class PPONetworks:
    def __init__(self, policy_network: PolicyNetwork, value_network: ValueNetwork, parametric_action_distribution: NormalTanhDistribution):
        self.policy_network = policy_network
        self.value_network = value_network
        self.parametric_action_distribution = parametric_action_distribution


def make_ppo_networks(
    observation_size: int,
    privileged_observation_size: int,
    action_size: int,
    preprocess_observations_fn: Callable = lambda x, y: x,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: Callable = torch.nn.SiLU,  # PyTorch equivalent of linen.swish is SiLU
    device: str = 'cpu'
) -> PPONetworks:
    """Make PPO networks with a PyTorch implementation."""

    # Create parametric action distribution (assumes you've already implemented this)
    parametric_action_distribution = NormalTanhDistribution(
        event_size=action_size
    )

    # Create policy network
    # Note: Ensure that your policy MLP ends with parametric_action_distribution.param_size units
    policy_network = PolicyNetwork(
        observation_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layers=policy_hidden_layer_sizes,
        output_size=parametric_action_distribution.param_size,
        activation=activation()
    ).to(device)

    # Create value network
    value_network = ValueNetwork(
        observation_size=privileged_observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layers=value_hidden_layer_sizes,
        activation=activation()
    ).to(device)

    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution
    )

def make_inference_fn(ppo_networks: PPONetworks, deterministic: bool = False) -> Callable:
    """Creates a function that returns a policy callable for inference."""
    policy_network = ppo_networks.policy_network
    parametric_action_distribution = ppo_networks.parametric_action_distribution

    def policy(observations: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            observations: torch.Tensor [batch, observation_dim]
        Returns:
            A tuple of (actions, extra_info_dict)
        """
        with torch.no_grad():
            logits = policy_network(observations)  # [batch, param_size]
            print(logits, observations)
            if deterministic:
                # Deterministic: use mode
                actions = parametric_action_distribution.mode(logits)
                return actions, {}
            else:
                # Stochastic: sample raw pre-tanh actions
                raw_actions = parametric_action_distribution.sample_no_postprocessing(logits)
                log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
                postprocessed_actions = parametric_action_distribution.postprocess(raw_actions)
                return postprocessed_actions, {
                    'log_prob': log_prob,
                    'raw_action': raw_actions
                }

    return policy
