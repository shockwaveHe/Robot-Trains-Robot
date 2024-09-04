from dataclasses import dataclass, field
from typing import Any, List, Optional

import jax
import jax.numpy as jnp
from brax import base  # type: ignore

from toddlerbot.envs.mjx_config import MJXConfig
from toddlerbot.envs.mjx_env import MJXEnv
from toddlerbot.sim.robot import Robot


@dataclass
class WalkCfg(MJXConfig):
    @dataclass
    class ActionConfig(MJXConfig.ActionConfig):
        cycle_time: float = 0.72

    @dataclass
    class CommandsConfig(MJXConfig.CommandsConfig):
        lin_vel_x_range: List[float] = field(default_factory=lambda: [-0.1, 0.3])
        lin_vel_y_range: List[float] = field(default_factory=lambda: [-0.1, 0.1])
        ang_vel_yaw_range: List[float] = field(default_factory=lambda: [-0.2, 0.2])

    @dataclass
    class RewardsConfig(MJXConfig.RewardsConfig):
        @dataclass
        class RewardScales(MJXConfig.RewardsConfig.RewardScales):
            # Walk specific rewards
            feet_air_time: float = 50.0
            feet_clearance: float = 0.0  # 1.0 # Doesn't help
            feet_contact: float = 0.5
            feet_distance: float = 1.0
            feet_slip: float = 0.1
            stand_still: float = 0.0  # 1.0

    def __init__(self):
        super().__init__()
        self.action = self.ActionConfig()
        self.commands = self.CommandsConfig()
        self.rewards = self.RewardsConfig()


class WalkEnv(MJXEnv):
    def __init__(
        self,
        name: str,
        robot: Robot,
        cfg: WalkCfg,
        ref_motion_type: str = "simple",
        fixed_base: bool = False,
        fixed_command: Optional[jax.Array] = None,
        add_noise: bool = True,
        **kwargs: Any,
    ):
        if ref_motion_type == "simple":
            from toddlerbot.ref_motion.walk_simple_ref import WalkSimpleReference

            motion_ref = WalkSimpleReference(
                robot,
                default_joint_pos=jnp.array(  # type:ignore
                    list(robot.default_joint_angles.values())
                ),
            )
        elif ref_motion_type == "zmp":
            from toddlerbot.ref_motion.walk_zmp_ref import WalkZMPReference

            motion_ref = WalkZMPReference(
                robot,
                cfg.action.cycle_time,
                [
                    cfg.commands.lin_vel_x_range,
                    cfg.commands.lin_vel_y_range,
                    cfg.commands.ang_vel_yaw_range,
                ],
                default_joint_pos=jnp.array(  # type:ignore
                    list(robot.default_joint_angles.values())
                ),
                control_dt=float(self.dt),
            )
        else:
            raise ValueError(f"Unknown ref_motion_type: {ref_motion_type}")

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

        # rng, rng_1, rng_2, rng_3 = jax.random.split(rng, 4)  # type:ignore
        # lin_vel_x = jax.random.uniform(  # type:ignore
        #     rng_1,
        #     (1,),
        #     minval=self.command_ranges["lin_vel_x"][0],
        #     maxval=self.command_ranges["lin_vel_x"][1],
        # )
        # lin_vel_y = jax.random.uniform(  # type:ignore
        #     rng_2,
        #     (1,),
        #     minval=self.command_ranges["lin_vel_y"][0],
        #     maxval=self.command_ranges["lin_vel_y"][1],
        # )
        # ang_vel_yaw = jax.random.uniform(  # type:ignore
        #     rng_3,
        #     (1,),
        #     minval=self.command_ranges["ang_vel_yaw"][0],
        #     maxval=self.command_ranges["ang_vel_yaw"][1],
        # )

        # TODO: Add command back
        commands = jnp.concatenate([jnp.array([0.3]), jnp.zeros(1), jnp.zeros(1)])  # type:ignore

        # Set small commands to zero based on norm condition
        mask = (jnp.linalg.norm(commands[:2]) > 0.05).astype(jnp.float32)  # type:ignore
        commands = commands.at[:2].set(commands[:2] * mask)  # type:ignore

        return commands

    def _reward_feet_air_time(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        # Reward air time.
        contact_filter = jnp.logical_or(info["stance_mask"], info["last_stance_mask"])  # type:ignore
        first_contact = (info["feet_air_time"] > 0) * contact_filter
        reward = jnp.sum(info["feet_air_time"] * first_contact)  # type:ignore
        # no reward for zero command
        reward *= jnp.linalg.norm(info["command"][:2]) > 0.05  # type:ignore
        return reward  # type:ignore

    def _reward_feet_clearance(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        # TODO: Fix this
        feet_height = pipeline_state.x.pos[self.feet_link_ids, 2]
        feet_z_delta = feet_height - info["init_feet_height"]
        is_close = jnp.abs(feet_z_delta - self.target_feet_z_delta) < 0.01  # type:ignore
        reward = jnp.sum(is_close * (1 - info["stance_mask"]))  # type:ignore
        return reward  # type:ignore

    def _reward_feet_contact(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Reward for contact"""
        reward = jnp.sum(info["stance_mask"] == info["state_ref"][-2:]).astype(  # type:ignore
            jnp.float32
        )
        return reward

    def _reward_feet_distance(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        # Calculates the reward based on the distance between the feet.
        # Penalize feet get close to each other or too far away on the y axis
        feet_pos = pipeline_state.x.pos[self.feet_link_ids, 1]
        feet_dist = jnp.abs(feet_pos[0] - feet_pos[1])  # type:ignore
        d_min = jnp.clip(feet_dist - self.min_feet_distance, max=0.0)  # type:ignore
        d_max = jnp.clip(feet_dist - self.max_feet_distance, min=0.0)  # type:ignore
        reward = (jnp.exp(-jnp.abs(d_min) * 100) + jnp.exp(-jnp.abs(d_max) * 100)) / 2  # type:ignore
        return reward

    def _reward_feet_slip(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        feet_speed = pipeline_state.xd.vel[self.feet_link_ids]  # type:ignore
        feet_speed_square = jnp.square(feet_speed[:, :2])  # type:ignore
        reward = -jnp.sum(feet_speed_square * info["stance_mask"])  # type:ignore
        # Penalize large feet velocity for feet that are in contact with the ground.
        return reward  # type:ignore

    def _reward_stand_still(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        # TODO: Fix this
        # Penalize motion at zero commands
        qpos_diff = jnp.sum(  # type:ignore
            jnp.abs(  # type:ignore
                pipeline_state.q[self.q_start_idx + self.leg_joint_indices]
                - self.default_qpos[self.q_start_idx + self.leg_joint_indices]
            )
        )
        reward = -(qpos_diff**2)  # type:ignore
        reward *= jnp.linalg.norm(info["command"][:2]) < 0.1  # type:ignore
        return reward  # type:ignore
