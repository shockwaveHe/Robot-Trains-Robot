import functools
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from brax.io import model  # type: ignore
from brax.training.agents.ppo import networks as ppo_networks  # type: ignore

from toddlerbot.envs.mjx_config import MuJoCoConfig
from toddlerbot.envs.mjx_env import MuJoCoEnv
from toddlerbot.policies import BasePolicy
from toddlerbot.sim.robot import Robot


class WalkFixedPolicy(BasePolicy):
    def __init__(self, robot: Robot):
        super().__init__(robot)
        self.name = "walk_fixed"

        cfg = MuJoCoConfig()
        cfg.rewards.healthy_z_range = [-0.2, 0.2]
        cfg.action.cycle_time = 1.2
        env = MuJoCoEnv(self.name, cfg, robot)

        make_networks_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=(128,) * 4,
            value_hidden_layer_sizes=(128,) * 4,
        )

        ppo_network = make_networks_factory(
            env.obs_size, env.privileged_obs_size, env.action_size
        )
        make_policy = ppo_networks.make_inference_fn(ppo_network)  # type: ignore

        policy_path = "tests/policy"
        params = model.load_params(policy_path)
        inference_fn = make_policy(params)

        # jit_inference_fn = inference_fn
        self.jit_inference_fn = jax.jit(inference_fn)  # type: ignore

        self.rng = jax.random.PRNGKey(0)  # type: ignore

    def run(
        self, obs_dict: Dict[str, npt.NDArray[np.float32]]
    ) -> npt.NDArray[np.float32]:
        obs = {}
        ctrl, _ = self.jit_inference_fn(obs, self.rng)  # type: ignore
        action = np.asarray(ctrl, dtype=np.float32)
        return action
