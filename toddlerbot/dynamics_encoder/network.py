import torch
from torch import nn
from toddlerbot.dynamics.networks import DynamicsNetwork
from toddlerbot.autoencoder.network import EncoderDecoder


# TODO: Debug this
class AEDynamicsNetwork(nn.Module):
    def __init__(
        self,
        observation_size,
        action_size,
        hidden_layers,
        encoder_decoder_args,
    ):
        super().__init__()
        self.autoencoder = EncoderDecoder(**encoder_decoder_args)
        self.dynamics_network = DynamicsNetwork(
            observation_size, action_size, hidden_layers, **encoder_decoder_args
        )

        # Keep a flat list of parameter shapes/sizes for easy assignment
        self.param_shapes = [
            param.shape for param in self.dynamics_network.parameters()
        ]
        self.param_numels = [
            param.numel() for param in self.dynamics_network.parameters()
        ]

    def assign_parameters(self, decoded_flat):
        pointer = 0
        with torch.no_grad():
            for param in self.dynamics_network.parameters():
                numel = param.numel()
                param.copy_(decoded_flat[pointer : pointer + numel].view_as(param))
                pointer += numel

    def forward(self, batch, batch_idx, processer_params=None, output_limit=1e2):
        params, obs, action = batch

        # Decode parameters from autoencoder
        x_recon, mean, logvar = self.autoencoder(params)
        decoded_flat = x_recon.flatten()

        # Efficient parameter assignment (no dict construction)
        self.assign_parameters(decoded_flat)

        # Use the updated dynamics network to predict
        obs_pred = self.dynamics_network(
            obs, action, batch_idx, processer_params, output_limit
        )

        return obs_pred
