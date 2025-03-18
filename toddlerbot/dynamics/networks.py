import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, layer_sizes, activation_fn, activate_final=False, layer_norm=False
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        # Build the sequence of layers
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            linear = nn.Linear(in_dim, out_dim, bias=True)
            self.layers.append(linear)

            # Add activation and LayerNorm if not the last layer or if activate_final is True
            if i < len(layer_sizes) - 2 or activate_final:
                self.layers.append(activation_fn())
                if layer_norm:
                    self.layers.append(nn.LayerNorm(out_dim))

    def forward(self, x):
        # Apply each layer in sequence
        for layer in self.layers:
            x = layer(x)
        return x


class DynamicsNetwork(nn.Module):
    def __init__(
        self,
        observation_size,
        action_size,
        hidden_layers,
        history_length,
        num_envs,
        step_batch_size,
        env_batch_size,
        preprocess_observations_fn=None,
        activation_fn=torch.nn.SiLU,
    ):
        super().__init__()
        if preprocess_observations_fn is None:
            self.preprocess_observations_fn = lambda x, y: x
        else:
            self.preprocess_observations_fn = preprocess_observations_fn

        self.num_envs = num_envs
        self.step_batch_size = step_batch_size
        self.env_batch_size = env_batch_size

        self.ensemble_mlp = nn.ModuleList(
            [
                MLP(
                    [(observation_size + action_size) * history_length]
                    + list(hidden_layers)
                    + [observation_size],
                    activation_fn=activation_fn,
                    layer_norm=False,
                    activate_final=False,
                )
                for _ in range(num_envs)
            ]
        )

    def forward(
        self, obs, action, batch_idx, processer_params=None, output_limit=1e2
    ) -> torch.Tensor:
        obs_processed = self.preprocess_observations_fn(obs, processer_params)
        input = (
            torch.cat([obs_processed, action], dim=-1)
            .reshape(self.step_batch_size, self.env_batch_size, -1)
            .transpose(0, 1)
        )
        env_start = (
            batch_idx
            // (self.env_batch_size * self.step_batch_size)
            * self.env_batch_size
        )
        env_end = min(env_start + self.env_batch_size, self.num_envs)

        output = input
        mlps = self.ensemble_mlp[env_start:env_end]
        for i, layer in enumerate(mlps[0].layers):
            if isinstance(layer, nn.Linear):
                # Stack weights and biases for this layer across all MLPs
                weights = torch.stack([mlp.layers[i].weight for mlp in mlps], dim=0)
                biases = torch.stack([mlp.layers[i].bias for mlp in mlps], dim=0)
                # Apply batched linear transformation
                output = torch.einsum(
                    "eoi,ebi->ebo", weights, output
                ) + biases.unsqueeze(1)
            elif callable(layer):
                # Apply activation function element-wise
                output = layer(output)

        output = torch.clip(output, -output_limit, output_limit)
        return output.transpose(0, 1).flatten(0, 1)
