from dataclasses import dataclass, field
from typing import List

import jax
import jax.numpy as jnp

from toddlerbot.envs.walk_env import WalkCfg, WalkEnv


@dataclass
class TurnCfg(WalkCfg, env_name="turn"):
    @dataclass
    class CommandsConfig(WalkCfg.CommandsConfig):
        command_range: List[List[float]] = field(
            default_factory=lambda: [[-0.1, 0.2, -0.5], [-0.1, 0.1, 0.5]]
        )

    @dataclass
    class RewardScales(WalkCfg.RewardsConfig.RewardScales):
        # Walk specific rewards
        lin_vel_xy: float = 3.0
        ang_vel_z: float = 3.0

    def __init__(self):
        super().__init__()
        self.commands = self.CommandsConfig()
        self.rewards.scales = self.RewardScales()


class TurnEnv(WalkEnv, env_name="turn"):
    def _sample_command(self, rng: jax.Array) -> jax.Array:
        # Split the RNG for method selection and command sampling
        rng, rng_method, rng_1, rng_2, rng_3 = jax.random.split(rng, 5)

        # Decide which sampling method to use (50% chance for each)
        method_choice = jax.random.bernoulli(rng_method, p=0.5)

        # Method 1: Sampling with a fixed z value and checking deadzone on z
        def sample_method_1():
            x = jnp.zeros(1)
            y = jnp.zeros(1)
            z = jax.random.uniform(
                rng_1,
                (1,),
                minval=self.command_range[0][0],
                maxval=self.command_range[0][1],
            )
            command = jnp.concatenate([x, y, z])

            # Set small commands to zero based on norm condition
            mask = (jnp.abs(command[2]) > self.deadzone).astype(jnp.float32)
            return command.at[2].set(command[2] * mask)

        # Method 2: Sampling on the elliptical xy-plane, checking deadzone on xy
        def sample_method_2():
            theta = jax.random.uniform(rng_2, (1,), minval=0, maxval=2 * jnp.pi)
            r = jax.random.uniform(rng_3, (1,), minval=0, maxval=1)

            x = jnp.where(
                jnp.sin(theta) > 0,
                self.command_range[0][1] * r * jnp.sin(theta),
                -self.command_range[0][0] * r * jnp.sin(theta),
            )
            y = jnp.where(
                jnp.cos(theta) > 0,
                self.command_range[1][1] * r * jnp.cos(theta),
                -self.command_range[1][0] * r * jnp.cos(theta),
            )
            z = jnp.zeros(1)
            command = jnp.concatenate([x, y, z])

            # Set small commands to zero based on norm condition
            mask = (jnp.linalg.norm(command[:2]) > self.deadzone).astype(jnp.float32)
            return command.at[:2].set(command[:2] * mask)

        # Choose the sampling method based on method_choice
        command = jax.lax.cond(method_choice, sample_method_1, sample_method_2)

        return command
