import pickle
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TanhTransform
from torch.distributions.transformed_distribution import TransformedDistribution
# from toddlerbot.utils.misc_utils import profile


def load_jax_params(path: str) -> Any:
    with Path(path).open("rb") as fin:
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
            jax_kernel = jax_params[kernel_key]["kernel"]  # shape (in_dim, out_dim)
            jax_bias = jax_params[kernel_key]["bias"]  # shape (out_dim,)

            # Convert to torch tensors
            w = torch.from_numpy(
                np.array(jax_kernel)
            ).T  # transpose to (out_dim, in_dim)
            b = torch.from_numpy(np.array(jax_bias))

            layer.weight.copy_(w)
            layer.bias.copy_(b)


def load_rsl_params_into_pytorch(
    pt_model: torch.nn.Module, value_net: torch.nn.Module, rsl_params: dict
):
    """
    Copies parameters from a JAX parameter dictionary into the corresponding
    PyTorch model. This function assumes that pt_model is an MLP-like structure
    with a series of Linear layers in the same order as JAX's 'hidden_0', 'hidden_1', etc.
    """
    # Extract layers from the PyTorch model.
    # We assume a Sequential or a Module with ordered layers that correspond to the JAX layers.
    # If your model is structured differently, adapt this accordingly.
    policy_linear_layers = [
        m for m in pt_model.modules() if isinstance(m, torch.nn.Linear)
    ]
    # value_linear_layers = [
    #     m for m in value_net.modules() if isinstance(m, torch.nn.Linear)
    # ]

    with torch.no_grad():
        for i, layer in enumerate(policy_linear_layers):
            # For each linear layer in PyTorch, find the corresponding hidden_i block in JAX params
            if i < 4:
                kernel_key = f"actor.{2 * i}"
            else:
                kernel_key = f"film_layers.{i - 4}.film"

            rsl_kernel = rsl_params[f"{kernel_key}.weight"]  # shape (in_dim, out_dim)
            rsl_bias = rsl_params[f"{kernel_key}.bias"]  # shape (out_dim,)

            rsl_kernel_torch = torch.tensor(rsl_kernel, dtype=layer.weight.dtype)
            rsl_bias_torch = torch.tensor(rsl_bias, dtype=layer.bias.dtype)

            # Resize kernel
            target_weight = torch.zeros_like(layer.weight)
            min_rows = min(layer.weight.shape[0], rsl_kernel_torch.shape[0])
            min_cols = min(layer.weight.shape[1], rsl_kernel_torch.shape[1])
            target_weight[:min_rows, :min_cols] = rsl_kernel_torch[:min_rows, :min_cols]
            layer.weight.copy_(target_weight)

            # Resize bias
            target_bias = torch.zeros_like(layer.bias)
            min_bias = min(layer.bias.shape[0], rsl_bias_torch.shape[0])
            target_bias[:min_bias] = rsl_bias_torch[:min_bias]
            layer.bias.copy_(target_bias)

        # for i, layer in enumerate(value_linear_layers):
        #     kernel_key = f"critic.{2 * i}"

        #     rsl_kernel = rsl_params[f"{kernel_key}.weight"]  # shape (in_dim, out_dim)
        #     rsl_bias = rsl_params[f"{kernel_key}.bias"]  # shape (out_dim,)

        #     rsl_kernel_torch = torch.tensor(rsl_kernel, dtype=layer.weight.dtype)
        #     rsl_bias_torch = torch.tensor(rsl_bias, dtype=layer.bias.dtype)

        #     # Resize kernel
        #     target_weight = torch.zeros_like(layer.weight)
        #     min_rows = min(layer.weight.shape[0], rsl_kernel_torch.shape[0])
        #     min_cols = min(layer.weight.shape[1], rsl_kernel_torch.shape[1])
        #     target_weight[:min_rows, :min_cols] = rsl_kernel_torch[:min_rows, :min_cols]
        #     layer.weight.copy_(target_weight)

        #     # Resize bias
        #     target_bias = torch.zeros_like(layer.bias)
        #     min_bias = min(layer.bias.shape[0], rsl_bias_torch.shape[0])
        #     target_bias[:min_bias] = rsl_bias_torch[:min_bias]
        #     layer.bias.copy_(target_bias)


def soft_clamp(
    x: torch.Tensor,
    _min: Optional[torch.Tensor] = None,
    _max: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer with statistics tracking."""

    def __init__(self, latent_dim, hidden_dim, debug=False):
        super().__init__()
        self.film = nn.Linear(latent_dim, 2 * hidden_dim)
        self.register_buffer("gamma_mean", torch.zeros(1))
        self.register_buffer("beta_mean", torch.zeros(1))
        self.register_buffer("count", torch.zeros(1))
        self.debug = debug

    # @profile()
    def forward(self, hidden, z):
        # Generate modulation parameters
        film_params = self.film(z)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)

        # Update running statistics (for monitoring only)
        if self.training and self.debug:
            with torch.no_grad():
                batch_mean_gamma = gamma.mean()
                batch_mean_beta = beta.mean()
                total = self.count + gamma.numel()
                self.gamma_mean = (
                    self.gamma_mean * self.count + batch_mean_gamma * gamma.numel()
                ) / total
                self.beta_mean = (
                    self.beta_mean * self.count + batch_mean_beta * gamma.numel()
                ) / total
                self.count = total

        # Apply feature-wise transformation
        return gamma * hidden + beta

    def reset_stats(self):
        """Reset tracking statistics"""
        self.gamma_mean.zero_()
        self.beta_mean.zero_()
        self.count.zero_()


class MLP(nn.Module):
    def __init__(
        self, layer_sizes, activation_fn, activate_final=False, layer_norm=False
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if layer_norm else None
        self.activate_final = activate_final
        self.activation = activation_fn()
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
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
            if i < len(self.layers) - 1 or self.activate_final:
                x = self.activation(x)
                if self.layer_norms is not None:
                    x = self.layer_norms[i](x)
        return x


class EnsembleLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_ensemble: int,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble

        self.register_parameter(
            "weight", nn.Parameter(torch.zeros(num_ensemble, input_dim, output_dim))
        )
        self.register_parameter(
            "bias", nn.Parameter(torch.zeros(num_ensemble, 1, output_dim))
        )

        nn.init.trunc_normal_(self.weight, std=1 / (2 * input_dim**0.5))

        self.register_parameter(
            "saved_weight", nn.Parameter(self.weight.detach().clone())
        )
        self.register_parameter("saved_bias", nn.Parameter(self.bias.detach().clone()))

        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias

        if len(x.shape) == 2:
            x = torch.einsum("ij,bjk->bik", x, weight)
        else:
            x = torch.einsum("bij,bjk->bik", x, weight)

        x = x + bias

        return x

    def load_save(self) -> None:
        self.weight.data.copy_(self.saved_weight.data)
        self.bias.data.copy_(self.saved_bias.data)

    def update_save(self, indexes: List[int]) -> None:
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]

    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = self.weight_decay * (0.5 * ((self.weight**2).sum()))
        return decay_loss


class EnsembleDynamicsNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        num_ensemble: int = 7,
        num_elites: int = 5,
        activation_fn: nn.Module = torch.nn.SiLU,
        weight_decays: Optional[Union[List[float], Tuple[float]]] = None,
        with_reward: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        self._with_reward = with_reward
        self.device = torch.device(device)

        self.activation = activation_fn()

        assert len(weight_decays) == (len(hidden_dims) + 1)

        module_list = []
        hidden_dims = [obs_dim + action_dim] + list(hidden_dims)
        if weight_decays is None:
            weight_decays = [0.0] * (len(hidden_dims) + 1)
        for in_dim, out_dim, weight_decay in zip(
            hidden_dims[:-1], hidden_dims[1:], weight_decays[:-1]
        ):
            module_list.append(
                EnsembleLinear(in_dim, out_dim, num_ensemble, weight_decay)
            )
        self.backbones = nn.ModuleList(module_list)

        self.output_layer = EnsembleLinear(
            hidden_dims[-1],
            2 * (obs_dim + self._with_reward),
            num_ensemble,
            weight_decays[-1],
        )

        self.register_parameter(
            "max_logvar",
            nn.Parameter(
                torch.ones(obs_dim + self._with_reward) * 0.5, requires_grad=True
            ),
        )
        self.register_parameter(
            "min_logvar",
            nn.Parameter(
                torch.ones(obs_dim + self._with_reward) * -10, requires_grad=True
            ),
        )

        self.register_parameter(
            "elites",
            nn.Parameter(
                torch.tensor(list(range(0, self.num_elites))), requires_grad=False
            ),
        )

        self.to(self.device)

    def forward(self, obs_action: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.as_tensor(obs_action, dtype=torch.float32).to(self.device)
        output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
        mean, logvar = torch.chunk(self.output_layer(output), 2, dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar

    def load_save(self) -> None:
        for layer in self.backbones:
            layer.load_save()
        self.output_layer.load_save()

    def update_save(self, indexes: List[int]) -> None:
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)

    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = 0
        for layer in self.backbones:
            decay_loss += layer.get_decay_loss()
        decay_loss += self.output_layer.get_decay_loss()
        return decay_loss

    def set_elites(self, indexes: List[int]) -> None:
        assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
        self.register_parameter(
            "elites", nn.Parameter(torch.tensor(indexes), requires_grad=False)
        )

    def random_elite_idxs(self, batch_size: int) -> np.ndarray:
        idxs = np.random.choice(self.elites.data.cpu().numpy(), size=batch_size)
        return idxs


class GaussianPolicyNetwork(nn.Module):
    def __init__(
        self,
        observation_size: int,
        hidden_layers: Tuple[int],
        action_size: int,
        preprocess_observations_fn,
        activation_fn=torch.nn.SiLU,
        noise_std_type: str = "learned",
        init_noise_std: float = 1.0,
        use_tanh: bool = True,
        film_layers: Optional[List[FiLMLayer]] = None,
    ) -> None:
        super().__init__()
        self.use_tanh = use_tanh
        self.preprocess_observations_fn = preprocess_observations_fn

        # Action noise
        self.noise_std_type = noise_std_type
        self.action_size = action_size
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(action_size))
            self.mlp = MLP(
                [observation_size] + list(hidden_layers) + [action_size],
                activation_fn=activation_fn,
                layer_norm=False,
                activate_final=False,
            )
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(action_size))
            )
            self.mlp = MLP(
                [observation_size] + list(hidden_layers) + [action_size],
                activation_fn=activation_fn,
                layer_norm=False,
                activate_final=False,
            )
        elif self.noise_std_type == "learned":
            self.mlp = MLP(
                [observation_size] + list(hidden_layers) + [action_size * 2],
                activation_fn=activation_fn,
                layer_norm=False,
                activate_final=False,
            )

        self._log_std_bound = (-10.0, 2.0)
        self.film_layers = film_layers

    # @profile()
    def forward(
        self, obs: torch.Tensor, z=None, processer_params=None
    ) -> torch.distributions.transformed_distribution.TransformedDistribution:
        obs = self.preprocess_observations_fn(obs, processer_params)
        if self.noise_std_type == "scalar":
            mu = self.mlp(obs)
            std = soft_clamp(
                self.std,
                torch.exp(self._log_std_bound[0]),
                torch.exp(self._log_std_bound[1]),
            )
        elif self.noise_std_type == "log":
            mu = self.mlp(obs)
            log_std = soft_clamp(
                self.log_std, self._log_std_bound[0], self._log_std_bound[1]
            )
            std = log_std.exp()
        elif self.noise_std_type == "learned":
            if z is not None:
                mu, log_std = self.actor_forward(obs, z).chunk(2, dim=-1)
            else:
                mu, log_std = self.mlp(obs).chunk(2, dim=-1)

            log_std = soft_clamp(
                log_std, self._log_std_bound[0], self._log_std_bound[1]
            )
            std = log_std.exp()
        dist = Normal(mu, std)
        if self.use_tanh:
            dist = TransformedDistribution(dist, [TanhTransform(cache_size=1)])
        return dist

    def forward_log_det_jacobian(self, x):
        # 2 * (log(2) - x - softplus(-2x))
        return 2.0 * (torch.log(torch.tensor(2.0)) - x - F.softplus(-2.0 * x))

    # @profile()
    def actor_forward(self, observations, z):
        h = observations
        layer_idx = 0
        film_idx = 0

        # Process layers with FiLM
        while layer_idx < len(self.mlp.layers) - 1:
            # Linear layer
            h = self.mlp.layers[layer_idx](h)
            layer_idx += 1

            # Activation
            if isinstance(self.mlp.activation, nn.SiLU):
                h = self.mlp.activation(h)
            # layer_idx += 1

            # Apply FiLM if available
            if film_idx < len(self.film_layers):
                h = self.film_layers[film_idx](h, z)
                film_idx += 1

        # Final output
        return self.mlp.layers[-1](h)


class ValueNetwork(nn.Module):
    def __init__(
        self,
        observation_size,
        preprocess_observations_fn,
        hidden_layers,
        activation_fn=torch.nn.SiLU,
    ):
        super().__init__()
        self.preprocess_observations_fn = preprocess_observations_fn
        self.mlp = MLP(
            [observation_size] + list(hidden_layers) + [1],
            activation_fn=activation_fn,
            layer_norm=False,
            activate_final=False,
        )

    def forward(self, obs, processer_params=None):
        obs = self.preprocess_observations_fn(obs, processer_params)
        return self.mlp(obs).squeeze(-1)


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_size,
        action_size,
        preprocess_observations_fn,
        hidden_layers,
        activation_fn=torch.nn.SiLU,
    ):
        super().__init__()
        self.preprocess_observations_fn = preprocess_observations_fn
        self.mlp = MLP(
            [observation_size + action_size] + list(hidden_layers) + [1],
            activation_fn=activation_fn,
            layer_norm=False,
            activate_final=False,
        )

    def forward(self, obs, action, processer_params=None):
        obs = self.preprocess_observations_fn(obs, processer_params)
        x = torch.cat([obs, action], dim=-1)
        return self.mlp(x).squeeze(-1)


class DoubleQNetwork(nn.Module):
    def __init__(
        self,
        observation_size,
        action_size,
        preprocess_observations_fn,
        hidden_layers,
        activation_fn=torch.nn.SiLU,
    ):
        super().__init__()
        self.q1 = QNetwork(
            observation_size,
            action_size,
            preprocess_observations_fn,
            hidden_layers,
            activation_fn,
        )
        self.q2 = QNetwork(
            observation_size,
            action_size,
            preprocess_observations_fn,
            hidden_layers,
            activation_fn,
        )

    def forward(self, obs, action, processer_params=None, return_min=True):
        if return_min:
            return torch.min(
                self.q1(obs, action, processer_params),
                self.q2(obs, action, processer_params),
            )
        else:
            return self.q1(obs, action, processer_params), self.q2(
                obs, action, processer_params
            )


class DynamicsNetwork(nn.Module):
    def __init__(
        self,
        observation_size,
        action_size,
        preprocess_observations_fn,
        hidden_layers,
        activation_fn=torch.nn.SiLU,
    ):
        super().__init__()
        self.preprocess_observations_fn = preprocess_observations_fn
        self.mlp = MLP(
            [observation_size + action_size]
            + list(hidden_layers)
            + [observation_size + 1],  # +1 for reward
            activation_fn=activation_fn,
            layer_norm=False,
            activate_final=False,
        )

    def forward(self, obs, action, processer_params=None) -> torch.Tensor:
        obs = self.preprocess_observations_fn(obs, processer_params)
        x = torch.cat([obs, action], dim=-1)
        return self.mlp(x)
