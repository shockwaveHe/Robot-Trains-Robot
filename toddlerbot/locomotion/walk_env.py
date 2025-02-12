from typing import Any, Optional

import jax
import jax.numpy as jnp
from brax import base

from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.locomotion.mjx_env import MJXEnv
from toddlerbot.reference.walk_simple_ref import WalkSimpleReference
from toddlerbot.reference.walk_zmp_ref import WalkZMPReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import quat2euler, quat_inv, rotate_vec


class WalkEnv(MJXEnv, env_name="walk"):
    """Walk environment with ToddlerBot."""

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
        """Initializes the walking controller with specified configuration and motion reference type.

        Args:
            name (str): The name of the controller.
            robot (Robot): The robot instance to be controlled.
            cfg (MJXConfig): Configuration settings for the controller.
            ref_motion_type (str, optional): Type of motion reference to use, either 'simple' or 'zmp'. Defaults to 'zmp'.
            fixed_base (bool, optional): Indicates if the robot has a fixed base. Defaults to False.
            add_noise (bool, optional): Whether to add noise to the simulation. Defaults to True.
            add_domain_rand (bool, optional): Whether to add domain randomization. Defaults to True.
            **kwargs (Any): Additional keyword arguments for the superclass initializer.

        Raises:
            ValueError: If an unknown `ref_motion_type` is provided.
        """
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
        """Generates a random command array based on the provided random number generator state and optionally the last command.

        Args:
            rng (jax.Array): A JAX random number generator state used for sampling.
            last_command (Optional[jax.Array]): The last command array to be used as a reference for generating the new command. If None, a new pose command is sampled.

        Returns:
            jax.Array: A command array consisting of pose and walk/turn components, sampled based on the given probabilities and constraints.
        """
        # Randomly sample an index from the command list
        rng, rng_1, rng_2, rng_3, rng_4, rng_5, rng_6 = jax.random.split(rng, 7)
        if last_command is not None:
            pose_command = last_command[:5]
        else:
            pose_command = self._sample_command_uniform(rng_1, self.command_range[:5])
            # If you want to sample a random pose command for the upper body, uncomment the line below
            pose_command = pose_command.at[:5].set(0.0)

        def sample_walk_command():
            """Generates a random walk command within specified elliptical bounds.

            This function samples random angles to compute a point on an ellipse, ensuring the point lies within defined command ranges. The resulting command is a 3D vector with the z-component set to zero.

            Returns:
                jnp.ndarray: A 3D vector representing the sampled walk command.
            """
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
            """Generates a sample turn command vector with randomized z-component.

            Returns:
                jnp.ndarray: A concatenated array representing the 3D vector [x, y, z].
            """
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
        """Calculates a reward based on the roll angle of the torso.

        This function computes a reward that penalizes deviations of the torso's roll angle from a specified range. The reward is calculated using an exponential function that decreases as the roll angle moves away from the desired range.

        Args:
            pipeline_state (base.State): The current state of the simulation, containing the orientation of the torso.
            info (dict[str, Any]): Additional information that might be used for reward calculation (not used in this function).
            action (jax.Array): The action taken by the agent (not used in this function).

        Returns:
            jax.Array: A scalar reward value based on the torso's roll angle.
        """
        torso_quat = pipeline_state.x.rot[0]
        torso_roll = quat2euler(torso_quat)[0]

        roll_min = jnp.clip(torso_roll - self.torso_roll_range[0], max=0.0)
        roll_max = jnp.clip(torso_roll - self.torso_roll_range[1], min=0.0)
        reward = (
            jnp.exp(-jnp.abs(roll_min) * 100) + jnp.exp(-jnp.abs(roll_max) * 100)
        ) / 2
        return reward

    def _reward_torso_pitch(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Calculates a reward based on the pitch of the torso.

        This function computes a reward that penalizes deviations of the torso's pitch from a specified range. The reward is calculated using the exponential of the absolute difference between the current pitch and the boundaries of the desired pitch range.

        Args:
            pipeline_state (base.State): The current state of the simulation, containing the rotation quaternion of the torso.
            info (dict[str, Any]): Additional information that might be used for reward computation (not used in this function).
            action (jax.Array): The action taken by the agent (not used in this function).

        Returns:
            jax.Array: A scalar reward value that decreases as the torso pitch deviates from the specified range.
        """
        torso_quat = pipeline_state.x.rot[0]
        torso_pitch = quat2euler(torso_quat)[1]

        pitch_min = jnp.clip(torso_pitch - self.torso_pitch_range[0], max=0.0)
        pitch_max = jnp.clip(torso_pitch - self.torso_pitch_range[1], min=0.0)
        reward = (
            jnp.exp(-jnp.abs(pitch_min) * 100) + jnp.exp(-jnp.abs(pitch_max) * 100)
        ) / 2
        return reward

    def _reward_feet_air_time(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates the reward based on the air time of feet during a movement cycle.

        This function computes a reward for a movement task by evaluating the air time of feet that have made contact with the ground. The reward is only given if the command observation norm exceeds a specified deadzone threshold.

        Args:
            pipeline_state (base.State): The current state of the pipeline, containing relevant state information.
            info (dict[str, Any]): A dictionary containing information about the current step, including 'stance_mask', 'last_stance_mask', 'feet_air_time', and 'command_obs'.
            action (jax.Array): The action taken at the current step.

        Returns:
            jax.Array: The computed reward based on the air time of feet that have made contact.
        """
        contact_filter = jnp.logical_or(info["stance_mask"], info["last_stance_mask"])
        first_contact = (info["feet_air_time"] > 0) * contact_filter
        reward = jnp.sum(info["feet_air_time"] * first_contact)
        # no reward for zero command
        reward *= jnp.linalg.norm(info["command_obs"]) > self.deadzone
        return reward

    def _reward_feet_clearance(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculate the reward for feet clearance during a movement.

        This function computes a reward based on the clearance of feet from the ground,
        considering only the first contact instances and ignoring cases where the command
        magnitude is below a specified threshold.

        Args:
            pipeline_state (base.State): The current state of the pipeline.
            info (dict[str, Any]): A dictionary containing information about the current
                state, including 'stance_mask', 'last_stance_mask', 'feet_air_dist', and
                'command_obs'.
            action (jax.Array): The action taken, represented as a JAX array.

        Returns:
            jax.Array: The computed reward for feet clearance.
        """
        contact_filter = jnp.logical_or(info["stance_mask"], info["last_stance_mask"])
        first_contact = (info["feet_air_dist"] > 0) * contact_filter
        reward = jnp.sum(info["feet_air_dist"] * first_contact)
        # no reward for zero command
        reward *= jnp.linalg.norm(info["command_obs"]) > self.deadzone
        return reward

    def _reward_feet_distance(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Calculates a reward based on the distance between the feet, penalizing positions where the feet are too close or too far apart on the y-axis.

        Args:
            pipeline_state (base.State): The current state of the pipeline, containing positional and rotational data.
            info (dict[str, Any]): Additional information that may be used for reward calculation.
            action (jax.Array): The action taken, represented as a JAX array.

        Returns:
            jax.Array: A reward value that decreases as the feet distance deviates from the desired range.
        """
        # Calculates the reward based on the distance between the feet.
        # Penalize feet get close to each other or too far away on the y axis
        feet_vec = rotate_vec(
            pipeline_state.x.pos[self.feet_link_ids[0]]
            - pipeline_state.x.pos[self.feet_link_ids[1]],
            quat_inv(pipeline_state.x.rot[0]),
        )
        feet_dist = jnp.abs(feet_vec[1])
        d_min = jnp.clip(feet_dist - self.min_feet_y_dist, max=0.0)
        d_max = jnp.clip(feet_dist - self.max_feet_y_dist, min=0.0)
        reward = (jnp.exp(-jnp.abs(d_min) * 100) + jnp.exp(-jnp.abs(d_max) * 100)) / 2
        return reward

    def _reward_feet_slip(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates a penalty based on the velocity of feet in contact with the ground.

        This function computes a reward that penalizes high velocities of feet that are in contact with the ground. The penalty is calculated as the negative sum of the squared velocities of the feet in the horizontal plane, weighted by a stance mask indicating which feet are in contact.

        Args:
            pipeline_state (base.State): The current state of the simulation, containing velocity information.
            info (dict[str, Any]): Additional information, including a 'stance_mask' that indicates which feet are in contact with the ground.
            action (jax.Array): The action taken, though not used in this calculation.

        Returns:
            jax.Array: A scalar penalty value representing the negative sum of squared velocities for feet in contact with the ground.
        """
        feet_speed = pipeline_state.xd.vel[self.feet_link_ids]
        feet_speed_square = jnp.square(feet_speed[:, :2])
        reward = -jnp.sum(feet_speed_square * info["stance_mask"])
        # Penalize large feet velocity for feet that are in contact with the ground.
        return reward

    def _reward_stand_still(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Calculates a penalty for motion when the command is near zero.

        Args:
            pipeline_state (base.State): The current state of the pipeline, containing joint positions.
            info (dict[str, Any]): Additional information, including command observations.
            action (jax.Array): The action taken, represented as a JAX array.

        Returns:
            jax.Array: A negative reward proportional to the squared difference in joint positions, applied when the command is within a specified deadzone.
        """
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
        """Calculates the alignment reward for ground contact based on joint positions.

        Args:
            pipeline_state (base.State): The current state of the pipeline containing joint positions.
            info (dict[str, Any]): Additional information, not used in this function.
            action (jax.Array): The action taken, not used in this function.

        Returns:
            jax.Array: The calculated reward based on the alignment of hip, knee, and ankle joint positions.
        """
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
