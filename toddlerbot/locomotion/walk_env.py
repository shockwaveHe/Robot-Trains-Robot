from typing import Any, Optional

import jax
import jax.numpy as jnp
from brax import base, math

from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.mjx_env import MJXEnv
from toddlerbot.motion.walk_simple_ref import WalkSimpleReference
from toddlerbot.motion.walk_zmp_ref import WalkZMPReference
from toddlerbot.sim.robot import Robot


class WalkEnv(MJXEnv, env_name="walk"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        cfg: MJXConfig,
        ref_motion_type: str = "zmp",
        fixed_base: bool = False,
        add_noise: bool = True,
        add_domain_rand: bool = True,
        **kwargs: Any,
    ):
        motion_ref: WalkSimpleReference | WalkZMPReference | None = None

        if ref_motion_type == "simple":
            motion_ref = WalkSimpleReference(
                robot, cfg.sim.timestep * cfg.action.n_frames, cfg.action.cycle_time
            )

        elif ref_motion_type == "zmp":
            motion_ref = WalkZMPReference(
                robot,
                cfg.sim.timestep * cfg.action.n_frames,
                cfg.action.cycle_time,
                cfg.action.waist_roll_max,
            )
        else:
            raise ValueError(f"Unknown ref_motion_type: {ref_motion_type}")

        self.cycle_time = jnp.array(cfg.action.cycle_time)
        self.torso_roll_range = cfg.rewards.torso_roll_range
        self.torso_pitch_range = cfg.rewards.torso_pitch_range

        self.max_feet_air_time = self.cycle_time / 2.0
        self.min_feet_y_dist = cfg.rewards.min_feet_y_dist
        self.max_feet_y_dist = cfg.rewards.max_feet_y_dist

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
        # Randomly sample an index from the command list
        rng, rng_1, rng_2, rng_3, rng_4, rng_5, rng_6 = jax.random.split(rng, 7)
        if last_command is not None:
            pose_command = last_command[:5]
        else:
            pose_command = self._sample_command_uniform(rng_1, self.command_range[:5])
            # TODO: Bring the random pose sampling back
            pose_command = pose_command.at[:5].set(0.0)

        def sample_walk_command():
            # Sample random angles uniformly between 0 and 2*pi
            theta = jax.random.uniform(rng_3, (1,), minval=0, maxval=2 * jnp.pi)
            # Parametric equation of ellipse
            x_max = jnp.where(
                jnp.sin(theta) > 0, self.command_range[5][1], -self.command_range[5][0]
            )
            x = jax.random.uniform(
                rng_4, (1,), minval=self.deadzone, maxval=x_max
            ) * jnp.sin(theta)
            y_max = jnp.where(
                jnp.cos(theta) > 0, self.command_range[6][1], -self.command_range[6][0]
            )
            y = jax.random.uniform(
                rng_4, (1,), minval=self.deadzone, maxval=y_max
            ) * jnp.cos(theta)
            z = jnp.zeros(1)
            return jnp.concatenate([x, y, z])

        def sample_turn_command():
            x = jnp.zeros(1)
            y = jnp.zeros(1)
            z = jnp.where(
                jax.random.uniform(rng_5, (1,)) < 0.5,
                jax.random.uniform(
                    rng_6,
                    (1,),
                    minval=self.deadzone,
                    maxval=self.command_range[7][1],
                ),
                -jax.random.uniform(
                    rng_6,
                    (1,),
                    minval=self.deadzone,
                    maxval=-self.command_range[7][0],
                ),
            )
            return jnp.concatenate([x, y, z])

        random_number = jax.random.uniform(rng_2, (1,))
        walk_command = jnp.where(
            random_number < self.zero_chance,
            jnp.zeros(3),
            jnp.where(
                random_number < self.zero_chance + self.turn_chance,
                sample_turn_command(),
                sample_walk_command(),
            ),
        )
        command = jnp.concatenate([pose_command, walk_command])

        # jax.debug.print("command: {}", command)

        return command

    def _reward_torso_roll(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Reward for torso pitch"""
        torso_quat = pipeline_state.x.rot[0]
        torso_roll = math.quat_to_euler(torso_quat)[0]

        roll_min = jnp.clip(torso_roll - self.torso_roll_range[0], max=0.0)
        roll_max = jnp.clip(torso_roll - self.torso_roll_range[1], min=0.0)
        reward = (
            jnp.exp(-jnp.abs(roll_min) * 100) + jnp.exp(-jnp.abs(roll_max) * 100)
        ) / 2
        return reward

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
        reward *= jnp.linalg.norm(info["command_obs"]) > self.deadzone
        return reward

    def _reward_feet_clearance(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        contact_filter = jnp.logical_or(info["stance_mask"], info["last_stance_mask"])
        first_contact = (info["feet_air_dist"] > 0) * contact_filter
        reward = jnp.sum(info["feet_air_dist"] * first_contact)
        # no reward for zero command
        reward *= jnp.linalg.norm(info["command_obs"]) > self.deadzone
        return reward

    def _reward_feet_distance(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        # Calculates the reward based on the distance between the feet.
        # Penalize feet get close to each other or too far away on the y axis
        feet_vec = math.rotate(
            pipeline_state.x.pos[self.feet_link_ids[0]]
            - pipeline_state.x.pos[self.feet_link_ids[1]],
            math.quat_inv(pipeline_state.x.rot[0]),
        )
        feet_dist = jnp.abs(feet_vec[1])
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
        reward *= jnp.linalg.norm(info["command_obs"]) < self.deadzone
        return reward

    def _reward_align_ground(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        hip_pitch_joint_pos = jnp.abs(
            pipeline_state.q[self.q_start_idx + self.hip_pitch_joint_indices]
        )
        knee_joint_pos = jnp.abs(
            pipeline_state.q[self.q_start_idx + self.knee_joint_indices]
        )
        ank_pitch_joint_pos = jnp.abs(
            pipeline_state.q[self.q_start_idx + self.ank_pitch_joint_indices]
        )
        error = hip_pitch_joint_pos + ank_pitch_joint_pos - knee_joint_pos
        reward = -jnp.mean(error**2)
        return reward
