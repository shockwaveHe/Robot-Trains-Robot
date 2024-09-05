from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
from brax import base  # type: ignore  # type: ignore

from toddlerbot.envs.mjx_config import MJXConfig
from toddlerbot.envs.mjx_env import MJXEnv
from toddlerbot.ref_motion.rotate_torso_ref import RotateTorsoReference
from toddlerbot.sim.robot import Robot


@dataclass
class RotateTorsoCfg(MJXConfig):
    @dataclass
    class ObsConfig(MJXConfig.ObsConfig):
        num_single_obs: int = 148
        num_single_privileged_obs: int = 187

    @dataclass
    class ActionConfig(MJXConfig.ActionConfig):
        resample_time: float = 2.0

    @dataclass
    class CommandsConfig(MJXConfig.CommandsConfig):
        num_commands: int = 2
        ang_vel_x_range: List[float] = field(default_factory=lambda: [-0.2, 0.2])
        ang_vel_z_range: List[float] = field(default_factory=lambda: [-1.0, 1.0])

    @dataclass
    class RewardScales(MJXConfig.RewardsConfig.RewardScales):
        # Squat specific rewards
        lin_vel_xy: float = 0.5
        lin_vel_z: float = 0.5
        ang_vel_xy: float = 1.0
        ang_vel_z: float = 1.0
        leg_joint_pos: float = 0.0
        waist_joint_pos: float = 5.0

    def __init__(self):
        super().__init__()
        self.obs = self.ObsConfig()
        self.action = self.ActionConfig()
        self.commands = self.CommandsConfig()
        self.rewards.scales = self.RewardScales()


class RotateTorsoEnv(MJXEnv):
    def __init__(
        self,
        name: str,
        robot: Robot,
        cfg: RotateTorsoCfg,
        fixed_base: bool = False,
        fixed_command: Optional[jax.Array] = None,
        add_noise: bool = True,
        **kwargs: Any,
    ):
        motion_ref = RotateTorsoReference(
            robot,
            default_motor_pos=jnp.array(list(robot.default_motor_angles.values())),  # type:ignore
        )

        self.num_commands = cfg.commands.num_commands
        self.ang_vel_x_range = cfg.commands.ang_vel_x_range
        self.ang_vel_z_range = cfg.commands.ang_vel_z_range

        self.waist_roll_limits = jnp.array(robot.joint_limits["waist_roll"])  # type:ignore
        self.waist_yaw_limits = jnp.array(robot.joint_limits["waist_yaw"])  # type:ignore

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

    def _sample_command(self, pipeline_state: base.State, rng: jax.Array) -> jax.Array:
        if self.fixed_command is not None:
            assert self.fixed_command.shape[0] == self.num_commands
            return self.fixed_command

        rng, rng_1, rng_2 = jax.random.split(rng, 3)  # type:ignore
        ang_vel_x = jax.random.uniform(  # type:ignore
            rng_1,
            (1,),
            minval=self.ang_vel_x_range[0],
            maxval=self.ang_vel_x_range[1],
        )
        ang_vel_z = jax.random.uniform(  # type:ignore
            rng_2,
            (1,),
            minval=self.ang_vel_z_range[0],
            maxval=self.ang_vel_z_range[1],
        )
        commands = jnp.concatenate([ang_vel_x, ang_vel_z])  # type:ignore

        return commands

    def _get_total_time(self, info: dict[str, Any]) -> jax.Array:
        time_total = jnp.max(  # type:ignore
            jnp.concatenate(  # type:ignore
                [
                    self.waist_roll_limits / info["command"][0],
                    self.waist_yaw_limits / info["command"][1],
                ]
            )
        )
        return time_total

    def _extract_command(self, info: dict[str, Any]) -> Tuple[jax.Array, jax.Array]:
        ang_vel_x = info["command"][0]
        ang_vel_z = info["command"][1]

        lin_vel = jnp.array([0.0, 0.0, 0.0])  # type:ignore
        ang_vel = jnp.array([ang_vel_x, 0.0, ang_vel_z])  # type:ignore

        return lin_vel, ang_vel
