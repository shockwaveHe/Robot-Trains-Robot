from dataclasses import dataclass, field
from typing import Any, List, Tuple

import jax
import jax.numpy as jnp
from brax import base, math

from toddlerbot.envs.mjx_config import MJXConfig
from toddlerbot.envs.mjx_env import MJXEnv
from toddlerbot.ref_motion.balance_ref import BalanceReference
from toddlerbot.sim.robot import Robot


@dataclass
class BalanceCfg(MJXConfig, env_name="balance"):
    @dataclass
    class ObsConfig(MJXConfig.ObsConfig):
        num_single_obs: int = 103
        num_single_privileged_obs: int = 142

    @dataclass
    class CommandsConfig(MJXConfig.CommandsConfig):
        num_commands: int = 4
        command_range: List[List[float]] = field(
            default_factory=lambda: [
                [-1.0, 1.0],
                [-1.0, 1.0],
                [0.0, 0.8],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-0.03, 0.03],
            ]
        )
        deadzone: List[float] = field(
            default_factory=lambda: [0.05, 0.05, 0.0, 0.05, 0.05, 0.005]
        )

    @dataclass
    class RewardScales(MJXConfig.RewardsConfig.RewardScales):
        # Balance specific rewards
        torso_pitch: float = 0.1

    def __init__(self):
        super().__init__()
        self.obs = self.ObsConfig()
        self.action = self.ActionConfig()
        self.commands = self.CommandsConfig()
        self.rewards.scales = self.RewardScales()


class BalanceEnv(MJXEnv, env_name="balance"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        cfg: BalanceCfg,
        fixed_base: bool = False,
        add_noise: bool = True,
        add_domain_rand: bool = True,
        **kwargs: Any,
    ):
        motion_ref = BalanceReference(robot)

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

    def _sample_command(self, rng: jax.Array) -> jax.Array:
        rng, rng_1, rng_2, rng_3, rng_4, rng_5, rng_6 = jax.random.split(rng, 5)
        neck_yaw_command = jax.random.uniform(
            rng_1,
            (1,),
            minval=self.command_range[0][0],
            maxval=self.command_range[0][1],
        )
        neck_pitch_command = jax.random.uniform(
            rng_2,
            (1,),
            minval=self.command_range[1][0],
            maxval=self.command_range[1][1],
        )
        arm_command = jax.random.uniform(
            rng_3,
            (1,),
            minval=self.command_range[2][0],
            maxval=self.command_range[2][1],
        )
        waist_roll_command = jax.random.uniform(
            rng_4,
            (1,),
            minval=self.command_range[3][0],
            maxval=self.command_range[3][1],
        )
        waist_yaw_command = jax.random.uniform(
            rng_5,
            (1,),
            minval=self.command_range[4][0],
            maxval=self.command_range[4][1],
        )
        squat_command = jax.random.uniform(
            rng_6,
            (1,),
            minval=self.command_range[5][0],
            maxval=self.command_range[5][1],
        )
        command = jnp.concatenate(
            [
                neck_yaw_command,
                neck_pitch_command,
                arm_command,
                waist_roll_command,
                waist_yaw_command,
                squat_command,
            ]
        )

        # Set small commands to zero based on norm condition
        mask = (jnp.abs(command) > self.deadzone).astype(jnp.float32)
        command = command.at[:].set(command * mask)

        return command

    def _extract_command(self, command: jax.Array) -> Tuple[jax.Array, jax.Array]:
        lin_vel = jnp.array([0.0, 0.0, command[5]])
        ang_vel = jnp.array([command[3], 0.0, command[4]])

        return lin_vel, ang_vel

    def _reward_torso_pitch(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Reward for torso pitch"""
        torso_quat = pipeline_state.x.rot[0]
        torso_pitch = math.quat_to_euler(torso_quat)[1]
        reward = -(torso_pitch**2)
        return reward
