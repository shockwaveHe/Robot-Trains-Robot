from dataclasses import dataclass, field
from typing import Any, List, Optional

import jax
import jax.numpy as jnp
from brax import base  # type: ignore  # type: ignore

from toddlerbot.envs.mjx_config import MJXConfig
from toddlerbot.envs.mjx_env import MJXEnv
from toddlerbot.ref_motion.squat_ref import SquatReference
from toddlerbot.sim.robot import Robot


@dataclass
class SquatCfg(MJXConfig):
    @dataclass
    class ActionConfig(MJXConfig.ActionConfig):
        resample_time: float = 1.5
        episode_time: float = 2.0

    @dataclass
    class CommandsConfig(MJXConfig.CommandsConfig):
        squat_depth_range: List[float] = field(default_factory=lambda: [-1, 1])

    @dataclass
    class RewardsConfig(MJXConfig.RewardsConfig):
        @dataclass
        class RewardScales(MJXConfig.RewardsConfig.RewardScales):
            # Walk specific rewards
            pass

    def __init__(self):
        super().__init__()
        self.action = self.ActionConfig()
        self.commands = self.CommandsConfig()
        self.rewards = self.RewardsConfig()


class SquatEnv(MJXEnv):
    def __init__(
        self,
        name: str,
        robot: Robot,
        cfg: SquatCfg,
        fixed_base: bool = False,
        fixed_command: Optional[jax.Array] = None,
        add_noise: bool = True,
        **kwargs: Any,
    ):
        motion_ref = SquatReference(
            robot,
            episode_time=cfg.action.episode_time,
            default_joint_pos=jnp.array(  # type:ignore
                list(self.robot.default_joint_angles.values())
            ),
        )

        self.squat_depth_range = cfg.commands.squat_depth_range

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
            return self.fixed_command

        rng, rng_1 = jax.random.split(rng)  # type:ignore
        squat_depth = jax.random.uniform(  # type:ignore
            rng_1,
            (1,),
            minval=self.squat_depth_range[0],
            maxval=self.squat_depth_range[1],
        )
        commands = squat_depth  # type:ignore

        return commands
