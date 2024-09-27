from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp

from toddlerbot.envs.mjx_config import MJXConfig
from toddlerbot.envs.mjx_env import MJXEnv
from toddlerbot.ref_motion.squat_ref import SquatReference
from toddlerbot.sim.robot import Robot


@dataclass
class SquatCfg(MJXConfig):
    @dataclass
    class ObsConfig(MJXConfig.ObsConfig):
        num_single_obs: int = 153
        num_single_privileged_obs: int = 192

    @dataclass
    class ActionConfig(MJXConfig.ActionConfig):
        action_scale: float = 0.25

    @dataclass
    class CommandsConfig(MJXConfig.CommandsConfig):
        num_commands: int = 1
        lin_vel_z_range: List[float] = field(default_factory=lambda: [-0.05, 0.05])

    @dataclass
    class RewardScales(MJXConfig.RewardsConfig.RewardScales):
        # Squat specific rewards
        lin_vel_xy: float = 0.5
        lin_vel_z: float = 1.5
        leg_joint_pos: float = 5.0
        waist_joint_pos: float = 5.0

    def __init__(self):
        super().__init__()
        self.obs = self.ObsConfig()
        self.action = self.ActionConfig()
        self.commands = self.CommandsConfig()
        self.rewards.scales = self.RewardScales()


class SquatEnv(MJXEnv):
    def __init__(
        self,
        name: str,
        robot: Robot,
        cfg: SquatCfg,
        fixed_base: bool = False,
        fixed_command: Optional[jax.Array] = None,
        add_noise: bool = True,
        add_domain_rand: bool = True,
        **kwargs: Any,
    ):
        motion_ref = SquatReference(robot)

        self.num_commands = cfg.commands.num_commands
        self.lin_vel_z_range = cfg.commands.lin_vel_z_range

        super().__init__(
            name,
            robot,
            cfg,
            motion_ref,
            fixed_base=fixed_base,
            fixed_command=fixed_command,
            add_noise=add_noise,
            add_domain_rand=add_domain_rand,
            **kwargs,
        )

    def _sample_command(self, rng: jax.Array) -> jax.Array:
        if self.fixed_command is not None:
            assert self.fixed_command.shape[0] == self.num_commands
            return self.fixed_command

        rng, rng_1 = jax.random.split(rng)
        lin_vel_z = jax.random.uniform(
            rng_1,
            (1,),
            minval=self.lin_vel_z_range[0],
            maxval=self.lin_vel_z_range[1],
        )
        commands = lin_vel_z

        return commands

    def _extract_command(self, command: jax.Array) -> Tuple[jax.Array, jax.Array]:
        z_vel = command[0]

        lin_vel = jnp.array([0.0, 0.0, z_vel])
        ang_vel = jnp.array([0.0, 0.0, 0.0])

        return lin_vel, ang_vel
