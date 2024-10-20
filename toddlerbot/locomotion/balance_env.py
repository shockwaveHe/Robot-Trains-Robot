from dataclasses import dataclass, field
from typing import Any, List, Optional

import jax
import jax.numpy as jnp

from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.mjx_env import MJXEnv
from toddlerbot.motion.balance_pd_ref import BalancePDReference
from toddlerbot.sim.robot import Robot


@dataclass
class BalanceCfg(MJXConfig, env_name="balance"):
    @dataclass
    class ObsConfig(MJXConfig.ObsConfig):
        num_single_obs: int = 101
        num_single_privileged_obs: int = 140

    @dataclass
    class CommandsConfig(MJXConfig.CommandsConfig):
        resample_time: float = 2.0
        command_range: List[List[float]] = field(
            default_factory=lambda: [
                [-1.5, 1.5],
                [-1.5, 1.5],
                [0.0, 0.5],
                [-0.3, 0.3],
                [-0.6, 0.6],
                [-1.0, 0.0],
            ]
        )
        deadzone: List[float] = field(
            default_factory=lambda: [0.05, 0.05, 0.0, 0.05, 0.05, 0.0]
        )
        command_obs_indices: List[int] = field(default_factory=lambda: [0, 1, 3, 4])

    @dataclass
    class RewardScales(MJXConfig.RewardsConfig.RewardScales):
        # Balance specific rewards
        torso_quat: float = 0.0

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
        motion_ref = BalancePDReference(robot, cfg.sim.timestep * cfg.action.n_frames)

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
        rng, rng_1, rng_2, rng_3, rng_4, rng_5, rng_6 = jax.random.split(rng, 7)
        if last_command is None:
            neck_yaw_command = self._sample_command_uniform(
                rng_1, self.command_range[0:1]
            )
            neck_pitch_command = self._sample_command_uniform(
                rng_2, self.command_range[1:2]
            )
            arm_command = self._sample_command_uniform(rng_3, self.command_range[2:3])
            waist_roll_command = self._sample_command_uniform(
                rng_4, self.command_range[3:4]
            )
            waist_yaw_command = self._sample_command_uniform(
                rng_5, self.command_range[4:5]
            )
            squat_command = self._sample_command_uniform(rng_6, self.command_range[5:6])
        else:
            # Sample neck and waist commands using Gaussian distribution with mean reversion towards last command
            neck_yaw_command = self._sample_command_normal_reversion(
                rng_1, self.command_range[0:1], last_command[0:1]
            )
            neck_pitch_command = self._sample_command_normal_reversion(
                rng_2, self.command_range[1:2], last_command[1:2]
            )
            arm_command = last_command[2:3]
            waist_roll_command = self._sample_command_normal_reversion(
                rng_4, self.command_range[3:4], last_command[3:4]
            )
            waist_yaw_command = self._sample_command_normal_reversion(
                rng_5, self.command_range[4:5], last_command[4:5]
            )
            squat_command = last_command[5:6]

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
