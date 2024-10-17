from dataclasses import dataclass, field
from typing import Any, List, Optional

import jax
import jax.numpy as jnp

from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.mjx_env import MJXEnv
from toddlerbot.motion.squat_ref import SquatReference
from toddlerbot.sim.robot import Robot


@dataclass
class SquatCfg(MJXConfig, env_name="squat"):
    @dataclass
    class ObsConfig(MJXConfig.ObsConfig):
        num_single_obs: int = 103
        num_single_privileged_obs: int = 142

    @dataclass
    class CommandsConfig(MJXConfig.CommandsConfig):
        resample_time: float = 2.0
        command_range: List[List[float]] = field(
            default_factory=lambda: [
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-0.03, 0.03],
            ]
        )
        deadzone: float = 0.005

    @dataclass
    class RewardScales(MJXConfig.RewardsConfig.RewardScales):
        # Balance specific rewards
        arm_joint_pos: float = 1.0
        arm_action_rate: float = 1e-2
        arm_action_acc: float = 1e-2
        leg_joint_pos: float = 0.0
        leg_action_rate: float = 0.0
        leg_action_acc: float = 0.0

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
        if last_command is not None:
            pose_command = last_command[:5]
        else:
            pose_command = self._sample_command_uniform(rng_1, self.command_range[:5])

        squat_command = self._sample_command_uniform(rng_2, self.command_range[5:6])

        command = jnp.concatenate([pose_command, squat_command])

        # Set small commands to zero based on norm condition
        mask = (jnp.abs(command[5:]) > self.deadzone).astype(jnp.float32)
        command = command.at[5:].set(command[5:] * mask)

        return command
