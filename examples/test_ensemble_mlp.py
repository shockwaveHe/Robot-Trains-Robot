import torch
from functorch import make_functional

from toddlerbot.dynamics.networks import DynamicsNetwork

obs_dim = 1245
act_dim = 12

model = DynamicsNetwork(obs_dim, act_dim, [128, 128, 128], 1)

# self.fmodel, self.ensemble_params = self.model, self.model
# Convert the model to its functional form.
fmodel, params = make_functional(model)
ensemble_size = 4
ensemble_params = tuple(torch.stack([p] * ensemble_size, dim=0) for p in params)

# Create an input tensor for each ensemble member.
# For example, if each ensemble member processes a batch of data, X should have shape:
# (ensemble_size, batch_size, input_dim)
batch_size = 32
obs_ensemble = torch.randn(ensemble_size, batch_size, obs_dim)
act_ensemble = torch.randn(ensemble_size, batch_size, act_dim)

# Use vmap to apply each ensemble member to its corresponding input.
# vmap maps over the first dimension of both ensemble_params and X.
ensemble_outputs = torch.vmap(lambda p, obs, act: fmodel(p, obs, act))(
    ensemble_params, obs_ensemble, act_ensemble
)

print("Ensemble outputs device:", ensemble_outputs.device)
print("Ensemble outputs shape:", ensemble_outputs.shape)
# Expected output shape: (ensemble_size, batch_size, output_dim)
print("Ensemble outputs:", ensemble_outputs)
