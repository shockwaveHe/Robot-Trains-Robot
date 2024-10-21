from dataclasses import dataclass, field
from typing import List, Optional

import jax
import jax.numpy as jnp

from toddlerbot.locomotion.walk_env import WalkCfg, WalkEnv


@dataclass
class TurnCfg(WalkCfg, env_name="turn"):
    @dataclass
    class CommandsConfig(WalkCfg.CommandsConfig):
        reset_time: float = 5.0
        command_range: List[List[float]] = field(
            default_factory=lambda: [
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [-0.5, 0.5],
            ]
        )
        deadzone: List[float] = field(default_factory=lambda: [0.1])
        command_obs_indices: List[int] = field(default_factory=lambda: [5, 6, 7])

    @dataclass
    class RewardScales(WalkCfg.RewardsConfig.RewardScales):
        # Walk specific rewards
        torso_pitch: float = 0.2
        lin_vel_xy: float = 1.0
        ang_vel_z: float = 5.0
        feet_air_time: float = 50.0
        feet_distance: float = 0.5
        feet_slip: float = 0.1
        feet_clearance: float = 1.0
        stand_still: float = 1.0

    def __init__(self):
        super().__init__()
        self.commands = self.CommandsConfig()
        self.rewards.scales = self.RewardScales()


class TurnEnv(WalkEnv, env_name="turn"):
    def _sample_command(
        self, rng: jax.Array, last_command: Optional[jax.Array] = None
    ) -> jax.Array:
        rng, rng_1, rng_2 = jax.random.split(rng, 3)
        if last_command is not None:
            pose_command = last_command[:5]
        else:
            pose_command = self._sample_command_uniform(rng_1, self.command_range[:5])
            pose_command = pose_command.at[3].set(0.0)
            pose_command = pose_command.at[4].set(0.0)

        # Parametric equation of ellipse
        x = jnp.zeros(1)
        y = jnp.zeros(1)
        z = jax.random.uniform(
            rng_2,
            (1,),
            minval=self.command_range[7][0],
            maxval=self.command_range[7][1],
        )
        command = jnp.concatenate([pose_command, x, y, z])

        # Set small commands to zero based on norm condition
        mask = (jnp.linalg.norm(command[5:]) > self.deadzone).astype(jnp.float32)
        command = command.at[5:].set(command[5:] * mask)

        return command
