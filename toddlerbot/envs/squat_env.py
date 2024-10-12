from dataclasses import dataclass, field
from typing import Any, List, Optional

import jax
import jax.numpy as jnp

from toddlerbot.envs.mjx_config import MJXConfig
from toddlerbot.envs.mjx_env import MJXEnv
from toddlerbot.ref_motion.squat_ref import SquatReference
from toddlerbot.sim.robot import Robot


@dataclass
class SquatCfg(MJXConfig, env_name="squat"):
    @dataclass
    class ObsConfig(MJXConfig.ObsConfig):
        num_single_obs: int = 103
        num_single_privileged_obs: int = 142

    @dataclass
    class CommandsConfig(MJXConfig.CommandsConfig):
        resample_time: float = 1.0
        command_range: List[List[float]] = field(
            default_factory=lambda: [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [-0.03, 0.03],
            ]
        )
        deadzone: float = 0.005

    @dataclass
    class RewardScales(MJXConfig.RewardsConfig.RewardScales):
        # Balance specific rewards
        torso_quat: float = 0.0
        lin_vel_z: float = 5.0

    def __init__(self):
        super().__init__()
        self.obs = self.ObsConfig()
        self.action = self.ActionConfig()
        self.commands = self.CommandsConfig()
        self.rewards.scales = self.RewardScales()


class SquatEnv(MJXEnv, env_name="squat"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        cfg: SquatCfg,
        fixed_base: bool = False,
        add_noise: bool = True,
        add_domain_rand: bool = True,
        **kwargs: Any,
    ):
        motion_ref = SquatReference(robot, cfg.sim.timestep * cfg.action.n_frames)

        self.command_range = jnp.array(cfg.commands.command_range)
        self.deadzone = jnp.array(cfg.commands.deadzone)

        super().__init__(
            name,
            robot,
            cfg,
            motion_ref,
            fixed_base=fixed_base,
            add_noise=add_noise,
            add_domain_rand=add_domain_rand,
            **kwargs,
        )

    def _sample_command(
        self, rng: jax.Array, last_command: Optional[jax.Array] = None
    ) -> jax.Array:
        rng, rng_1, rng_2 = jax.random.split(rng, 3)
        pose_command = jax.random.uniform(
            rng_1,
            (5,),
            minval=self.command_range[:5, 0],
            maxval=self.command_range[:5, 1],
        )
        squat_command = jax.random.uniform(
            rng_2,
            (1,),
            minval=self.command_range[5][0],
            maxval=self.command_range[5][1],
        )
        command = jnp.concatenate([pose_command, squat_command])

        # Set small commands to zero based on norm condition
        mask = (jnp.abs(command[5:]) > self.deadzone).astype(jnp.float32)
        command = command.at[5:].set(command[5:] * mask)

        return command
