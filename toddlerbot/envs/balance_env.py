from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp

from toddlerbot.envs.mjx_config import MJXConfig
from toddlerbot.envs.mjx_env import MJXEnv
from toddlerbot.ref_motion.balance_ref import BalanceReference
from toddlerbot.sim.robot import Robot


@dataclass
class BalanceCfg(MJXConfig):
    @dataclass
    class ObsConfig(MJXConfig.ObsConfig):
        num_single_obs: int = 98
        num_single_privileged_obs: int = 137

    @dataclass
    class CommandsConfig(MJXConfig.CommandsConfig):
        resample_time: float = 100.0  # No resampling
        num_commands: int = 1
        sample_range: List[float] = field(default_factory=lambda: [0.0, 1.0])

    @dataclass
    class RewardScales(MJXConfig.RewardsConfig.RewardScales):
        # Balance specific rewards
        torso_pitch = 1.0

    def __init__(self):
        super().__init__()
        self.obs = self.ObsConfig()
        self.action = self.ActionConfig()
        self.commands = self.CommandsConfig()
        self.rewards.scales = self.RewardScales()


class BalanceEnv(MJXEnv):
    def __init__(
        self,
        name: str,
        robot: Robot,
        cfg: BalanceCfg,
        fixed_base: bool = False,
        fixed_command: Optional[jax.Array] = None,
        add_noise: bool = True,
        **kwargs: Any,
    ):
        motion_ref = BalanceReference(robot)

        self.num_commands = cfg.commands.num_commands
        self.sample_range = cfg.commands.sample_range

        super().__init__(
            name,
            robot,
            cfg,
            motion_ref,
            fixed_base=fixed_base,
            fixed_command=fixed_command,
            add_noise=add_noise,
            **kwargs,
        )

    def _sample_command(self, rng: jax.Array) -> jax.Array:
        if self.fixed_command is not None:
            assert self.fixed_command.shape[0] == self.num_commands
            return self.fixed_command

        rng, rng_1 = jax.random.split(rng)
        commands = jax.random.uniform(
            rng_1,
            (1,),
            minval=self.sample_range[0],
            maxval=self.sample_range[1],
        )
        return commands

    def _extract_command(self, command: jax.Array) -> Tuple[jax.Array, jax.Array]:
        lin_vel = jnp.array([0.0, 0.0, 0.0])
        ang_vel = jnp.array([0.0, 0.0, 0.0])

        return lin_vel, ang_vel
