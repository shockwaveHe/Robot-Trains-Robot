from dataclasses import dataclass, field
from typing import List

import jax
import jax.numpy as jnp

from toddlerbot.envs.walk_env import WalkCfg, WalkEnv


@dataclass
class TurnCfg(WalkCfg, env_name="turn"):
    @dataclass
    class CommandsConfig(WalkCfg.CommandsConfig):
        command_range: List[List[float]] = field(default_factory=lambda: [[-0.5, 0.5]])
        deadzone: float = 0.2

    @dataclass
    class RewardScales(WalkCfg.RewardsConfig.RewardScales):
        # Walk specific rewards
        torso_quat: float = 0.0
        torso_pitch: float = 0.1
        lin_vel_xy: float = 0.5
        ang_vel_z: float = 5.0
        feet_air_time: float = 50.0
        feet_distance: float = 0.5
        feet_slip: float = 0.1
        stand_still: float = 1.0

    def __init__(self):
        super().__init__()
        self.commands = self.CommandsConfig()
        self.rewards.scales = self.RewardScales()


class TurnEnv(WalkEnv, env_name="turn"):
    def _sample_command(self, rng: jax.Array) -> jax.Array:
        # Randomly sample an index from the command list
        rng, rng_1 = jax.random.split(rng, 2)

        # Parametric equation of ellipse
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
        command = command.at[2].set(command[2] * mask)

        return command
