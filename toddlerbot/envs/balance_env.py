from dataclasses import dataclass
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

from toddlerbot.envs.mjx_config import MJXConfig
from toddlerbot.envs.mjx_env import MJXEnv
from toddlerbot.ref_motion.squat_ref import SquatReference
from toddlerbot.sim.robot import Robot


@dataclass
class BalanceCfg(MJXConfig):
    @dataclass
    class ObsConfig(MJXConfig.ObsConfig):
        num_single_obs: int = 98
        num_single_privileged_obs: int = 137

    @dataclass
    class CommandsConfig(MJXConfig.CommandsConfig):
        resample_time: float = 5.0
        num_commands: int = 1

    @dataclass
    class RewardScales(MJXConfig.RewardsConfig.RewardScales):
        # Balance specific rewards
        pass

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
        motion_ref = SquatReference(robot)

        self.num_commands = cfg.commands.num_commands

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
        rng, rng_1 = jax.random.split(rng)  # type:ignore
        commands = jax.random.uniform(  # type:ignore
            rng_1,
            (1,),
            minval=0.0,
            maxval=1.0,
        )
        return commands

    def _extract_command(self, command: jax.Array) -> Tuple[jax.Array, jax.Array]:
        lin_vel = jnp.array([0.0, 0.0, 0.0])  # type:ignore
        ang_vel = jnp.array([0.0, 0.0, 0.0])  # type:ignore

        return lin_vel, ang_vel
