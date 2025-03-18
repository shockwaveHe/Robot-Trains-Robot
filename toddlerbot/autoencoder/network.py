import random

import torch
from torch import nn


def build_encoder(n_embd, encoder_depth, input_splits):
    # Create a unique MLP encoder for each token
    input_parameter_projections = nn.ModuleList()
    for param_chunk_size in input_splits:
        in_proj = [nn.Linear(param_chunk_size, n_embd, bias=False)]
        for _ in range(encoder_depth - 1):
            in_proj.append(nn.GELU())
            in_proj.append(nn.Linear(n_embd, n_embd, bias=False))
        in_proj = nn.Sequential(*in_proj)
        input_parameter_projections.append(in_proj)
    return input_parameter_projections


def build_decoder(n_embd, decoder_depth, output_splits):
    # Create a unique MLP decoder for each noised token
    output_parameter_projections = nn.ModuleList()
    for output_chunk_size in output_splits:
        out_proj = []
        for _ in range(decoder_depth - 1):
            out_proj.append(nn.Linear(n_embd, n_embd, bias=False))
            out_proj.append(nn.GELU())
        out_proj.append(nn.Linear(n_embd, output_chunk_size, bias=False))
        out_proj = nn.Sequential(*out_proj)
        output_parameter_projections.append(out_proj)
    return output_parameter_projections


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        input_splits,
        n_embd,
        encoder_depth=1,
        decoder_depth=1,
        input_noise_factor=0.0001,
        latent_noise_factor=0.001,
        is_vae=False,  # New flag to toggle between AE and VAE
    ):
        super().__init__()
        self.input_splits = input_splits
        self.is_vae = is_vae
        self.input_parameter_projections = build_encoder(
            n_embd, encoder_depth, self.input_splits
        )
        self.output_parameter_projections = build_decoder(
            n_embd, decoder_depth, self.input_splits
        )
        self.num_output_heads = len(self.input_splits)
        self.input_noise_factor = input_noise_factor
        self.latent_noise_factor = latent_noise_factor
        self.ln_in = nn.LayerNorm(n_embd)

        # VAE-specific layers: only initialize if is_vae is True
        if self.is_vae:
            self.mean_layers = nn.ModuleList()
            self.logvar_layers = nn.ModuleList()
            for _ in range(len(self.input_splits)):
                self.mean_layers.append(nn.Linear(n_embd, n_embd))
                self.logvar_layers.append(nn.Linear(n_embd, n_embd))

    def encode(self, parameters, logvar_limit=20):
        """
        Chunk input parameter vector, apply per-chunk encoding, and
        stack projected chunks along the sequence (token) dimension.
        For VAE, compute mean and logvar, then sample latent z.
        """
        assert parameters.dim() == 2
        split_parameters = torch.split(parameters, self.input_splits, dim=1)

        if self.is_vae:
            representations, means, logvars = [], [], []
            for parameter, in_proj, mean_layer, logvar_layer in zip(
                split_parameters,
                self.input_parameter_projections,
                self.mean_layers,
                self.logvar_layers,
            ):
                repr = in_proj(parameter)
                # Compute mean and log variance for each token
                mean = mean_layer(repr)
                logvar = logvar_layer(repr)
                logvar = torch.clip(logvar, -logvar_limit, logvar_limit)

                # Reparameterization trick for sampling
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mean + eps * std

                representations.append(z)
                means.append(mean)
                logvars.append(logvar)

            representations = torch.stack(representations, dim=1)
            representations = self.ln_in(representations)
            assert representations.dim() == 3
            mean = torch.stack(means, dim=1)
            logvar = torch.stack(logvars, dim=1)
        else:
            representations = []
            for parameter, in_proj in zip(
                split_parameters, self.input_parameter_projections
            ):
                representations.append(in_proj(parameter))

            representations = torch.stack(representations, dim=1)
            representations = self.ln_in(representations)
            assert representations.dim() == 3
            # For AE, use representations directly
            mean, logvar = None, None

        return representations, mean, logvar

    def decode(self, features):
        """
        Apply a per-chunk decoding (only to the tokens corresponding to the noised/updated parameter vector),
        and concatenate them into a flattened parameter vector.
        """
        assert features.dim() == 3  # (b, t, d)
        x_recon = []
        for t in range(self.num_output_heads):
            out_proj = self.output_parameter_projections[t]
            x_recon.append(out_proj(features[:, t, :]))
        x_recon = torch.cat(x_recon, 1)  # (b, c)
        assert x_recon.dim() == 2
        return x_recon

    def add_noise(self, x, noise_factor):
        if not isinstance(noise_factor, float):
            assert len(noise_factor) == 2
            noise_factor = random.uniform(noise_factor[0], noise_factor[1])

        return torch.randn_like(x) * noise_factor + x * (1 - noise_factor)

    def forward(self, x):
        x = self.add_noise(x, self.input_noise_factor)
        z, mean, logvar = self.encode(x)
        z = self.add_noise(z, self.latent_noise_factor)
        z = torch.clamp(z, -1, 1)
        x_recon = self.decode(z)
        return x_recon, mean, logvar
