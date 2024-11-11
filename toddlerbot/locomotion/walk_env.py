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
        self.torso_pitch_range = cfg.rewards.torso_pitch_range
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
        rng, rng_1, rng_2, rng_3 = jax.random.split(rng, 4)
        if last_command is not None:
            pose_command = last_command[:5]
        else:
            pose_command = self._sample_command_uniform(rng_1, self.command_range[:5])
            # TODO: Bring the random pose sampling back
            pose_command = pose_command.at[:5].set(0.0)

        # Sample random angles uniformly between 0 and 2*pi
        theta = jax.random.uniform(rng_2, (1,), minval=0, maxval=2 * jnp.pi)
        r = jax.random.uniform(rng_3, (1,), minval=0, maxval=1)

        # Parametric equation of ellipse
        x = jnp.where(
            jnp.sin(theta) > 0,
            self.command_range[5][1] * r * jnp.sin(theta),
            -self.command_range[5][0] * r * jnp.sin(theta),
        )
        y = jnp.where(
            jnp.cos(theta) > 0,
            self.command_range[6][1] * r * jnp.cos(theta),
            -self.command_range[6][0] * r * jnp.cos(theta),
        )
        z = jnp.zeros(1)
        command = jnp.concatenate([pose_command, x, y, z])

        # Set small commands to zero based on norm condition
        mask = (jnp.linalg.norm(command[5:]) > self.deadzone).astype(jnp.float32)
        command = command.at[5:].set(command[5:] * mask)

        return command

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

    def _reward_hip_motor_torque(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for minimizing joint torques"""
        torque = pipeline_state.qfrc_actuator[
            self.qd_start_idx + self.hip_motor_indices
        ]
        error = jnp.square(torque)
        reward = -jnp.mean(error)
        return reward
