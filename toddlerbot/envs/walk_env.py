from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
from brax import base, math

from toddlerbot.envs.mjx_config import MJXConfig
from toddlerbot.envs.mjx_env import MJXEnv
from toddlerbot.ref_motion.walk_simple_ref import WalkSimpleReference
from toddlerbot.ref_motion.walk_zmp_ref import WalkZMPReference
from toddlerbot.sim.robot import Robot


@dataclass
class WalkCfg(MJXConfig):
    @dataclass
    class ActionConfig(MJXConfig.ActionConfig):
        cycle_time: float = 0.96

    @dataclass
    class CommandsConfig(MJXConfig.CommandsConfig):
        command_list: List[List[float]] = field(
            default_factory=lambda: [
                [0.0, 0.0, 0.0],
                [-0.1, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, -0.1, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.2],
                [0.0, 0.0, -0.2],
            ]
        )

    @dataclass
    class RewardScales(MJXConfig.RewardsConfig.RewardScales):
        # Walk specific rewards
        torso_pitch: float = 0.1
        lin_vel_xy: float = 2.0
        feet_air_time: float = 50.0
        feet_clearance: float = 0.0  # Doesn't help
        feet_distance: float = 0.5
        feet_slip: float = 0.1
        stand_still: float = 0.0  # 1.0

    def __init__(self):
        super().__init__()
        self.action = self.ActionConfig()
        self.commands = self.CommandsConfig()
        self.rewards.scales = self.RewardScales()


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
        add_domain_rand: bool = True,
        **kwargs: Any,
    ):
        motion_ref: WalkSimpleReference | WalkZMPReference | None = None

        if ref_motion_type == "simple":
            motion_ref = WalkSimpleReference(robot, cfg.action.cycle_time)

        elif ref_motion_type == "zmp":
            motion_ref = WalkZMPReference(
                robot,
                cfg.commands.command_list,
                cfg.action.cycle_time,
                cfg.sim.timestep * cfg.action.n_frames,
            )
        else:
            raise ValueError(f"Unknown ref_motion_type: {ref_motion_type}")

        self.cycle_time = jnp.array(cfg.action.cycle_time)
        self.command_list = jnp.array(cfg.commands.command_list)
        self.torso_pitch_range = cfg.rewards.torso_pitch_range
        self.min_feet_y_dist = cfg.rewards.min_feet_y_dist
        self.max_feet_y_dist = cfg.rewards.max_feet_y_dist
        self.target_feet_z_delta = cfg.rewards.target_feet_z_delta

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
            return self.fixed_command

        # Randomly sample an index from the command list
        num_commands = self.command_list.shape[0]
        rng, rng1 = jax.random.split(rng)
        command_idx = jax.random.randint(rng1, (), 0, num_commands)

        # Select the corresponding command
        command = self.command_list[command_idx]

        # Set small commands to zero based on norm condition
        mask = (jnp.linalg.norm(command[:2]) > 0.01).astype(jnp.float32)
        command = command.at[:2].set(command[:2] * mask)

        return command

    def _extract_command(self, command: jax.Array) -> Tuple[jax.Array, jax.Array]:
        x_vel = command[0]
        y_vel = command[1]
        yaw_vel = command[2]

        lin_vel = jnp.array([x_vel, y_vel, 0.0])
        ang_vel = jnp.array([0.0, 0.0, yaw_vel])
        return lin_vel, ang_vel

    def _reward_torso_pitch(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Reward for torso pitch"""
        torso_quat = pipeline_state.x.rot[0]
        torso_pitch = math.quat_to_euler(torso_quat)[1]

        pitch_min = jnp.clip(torso_pitch - self.torso_pitch_range[0], max=0.0)
        pitch_max = jnp.clip(torso_pitch - self.torso_pitch_range[1], min=0.0)
        reward = (
            jnp.exp(-jnp.abs(pitch_min) * 100) + jnp.exp(-jnp.abs(pitch_max) * 100)
        ) / 2
        return reward

    def _reward_feet_air_time(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        # Reward air time.
        contact_filter = jnp.logical_or(info["stance_mask"], info["last_stance_mask"])
        first_contact = (info["feet_air_time"] > 0) * contact_filter
        reward = jnp.sum(info["feet_air_time"] * first_contact)
        # no reward for zero command
        reward *= jnp.linalg.norm(info["command"][:2]) > 0.01
        return reward

    def _reward_feet_clearance(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        feet_height = pipeline_state.x.pos[self.feet_link_ids, 2]
        feet_z_delta = feet_height - info["init_feet_height"]
        is_close = jnp.abs(feet_z_delta - self.target_feet_z_delta) < 0.01
        reward = jnp.sum(is_close * (1 - info["stance_mask"]))
        return reward

    def _reward_feet_distance(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        # Calculates the reward based on the distance between the feet.
        # Penalize feet get close to each other or too far away on the y axis
        feet_pos = pipeline_state.x.pos[self.feet_link_ids, 1]
        feet_dist = jnp.abs(feet_pos[0] - feet_pos[1])
        d_min = jnp.clip(feet_dist - self.min_feet_y_dist, max=0.0)
        d_max = jnp.clip(feet_dist - self.max_feet_y_dist, min=0.0)
        reward = (jnp.exp(-jnp.abs(d_min) * 100) + jnp.exp(-jnp.abs(d_max) * 100)) / 2
        return reward

    def _reward_feet_slip(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        feet_speed = pipeline_state.xd.vel[self.feet_link_ids]
        feet_speed_square = jnp.square(feet_speed[:, :2])
        reward = -jnp.sum(feet_speed_square * info["stance_mask"])
        # Penalize large feet velocity for feet that are in contact with the ground.
        return reward

    def _reward_stand_still(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        # Penalize motion at zero commands
        qpos_diff = jnp.sum(
            jnp.abs(
                pipeline_state.q[self.q_start_idx + self.leg_joint_indices]
                - self.default_qpos[self.q_start_idx + self.leg_joint_indices]
            )
        )
        reward = -(qpos_diff**2)
        reward *= jnp.linalg.norm(info["command"][:2]) < 0.01
        return reward
