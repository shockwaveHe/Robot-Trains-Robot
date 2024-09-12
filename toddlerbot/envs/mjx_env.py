from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import mujoco
import numpy as np
from brax import base, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import support

from toddlerbot.envs.mjx_config import MJXConfig
from toddlerbot.ref_motion import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.math_utils import exponential_moving_average


class MJXEnv(PipelineEnv):
    def __init__(
        self,
        name: str,
        robot: Robot,
        cfg: MJXConfig,
        motion_ref: MotionReference,
        fixed_base: bool = False,
        fixed_command: Optional[jax.Array] = None,
        add_noise: bool = True,
        add_push: bool = True,
        **kwargs: Any,
    ):
        self.name = name
        self.cfg = cfg
        self.robot = robot
        self.motion_ref = motion_ref
        self.fixed_base = fixed_base
        self.fixed_command = fixed_command
        self.add_noise = add_noise
        self.add_push = add_push

        if fixed_base:
            xml_path = find_robot_file_path(robot.name, suffix="_fixed_scene.xml")
        else:
            xml_path = find_robot_file_path(robot.name, suffix="_scene.xml")

        sys = mjcf.load(xml_path)
        sys = sys.tree_replace(
            {
                "opt.timestep": cfg.sim.timestep,
                "opt.solver": cfg.sim.solver,
                "opt.iterations": cfg.sim.iterations,
                "opt.ls_iterations": cfg.sim.ls_iterations,
            }
        )

        kwargs["n_frames"] = cfg.action.n_frames
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

        self._init_env()
        self._init_reward()

    def _init_env(self) -> None:
        self.nu = self.sys.nu
        self.nq = self.sys.nq
        self.nv = self.sys.nv

        # colliders
        pair_geom1 = self.sys.pair_geom1
        pair_geom2 = self.sys.pair_geom2
        self.collider_geom_ids = np.unique(np.concatenate([pair_geom1, pair_geom2]))
        self.num_colliders = self.collider_geom_ids.shape[0]
        left_foot_collider_indices: List[int] = []
        right_foot_collider_indices: List[int] = []
        for i, geom_id in enumerate(self.collider_geom_ids):
            geom_name = support.id2name(self.sys, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name is None:
                continue

            if f"{self.robot.foot_name}_2" in geom_name:
                right_foot_collider_indices.append(i)
            elif f"{self.robot.foot_name}" in geom_name:
                left_foot_collider_indices.append(i)

        self.left_foot_collider_indices = jnp.array(left_foot_collider_indices)
        self.right_foot_collider_indices = jnp.array(right_foot_collider_indices)

        foot_link_mask = jnp.array(
            np.char.find(self.sys.link_names, self.robot.foot_name) >= 0
        )
        self.feet_link_ids = jnp.arange(self.sys.num_links())[foot_link_mask]

        self.contact_force_threshold = self.cfg.action.contact_force_threshold

        # This leads to CPU memory leak
        # self.jit_contact_force = jax.jit(support.contact_force, static_argnums=(2, 3))
        self.jit_contact_force = support.contact_force

        self.joint_indices = jnp.array(
            [
                support.name2id(self.sys, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.joint_ordering
            ]
        )
        if not self.fixed_base:
            # Disregard the free joint
            self.joint_indices -= 1

        joint_groups = np.array(
            [self.robot.joint_groups[name] for name in self.robot.joint_ordering]
        )
        self.leg_joint_indices = self.joint_indices[joint_groups == "leg"]
        self.arm_joint_indices = self.joint_indices[joint_groups == "arm"]
        self.neck_joint_indices = self.joint_indices[joint_groups == "neck"]
        self.waist_joint_indices = self.joint_indices[joint_groups == "waist"]

        self.motor_indices = jnp.array(
            [
                support.name2id(self.sys, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.motor_ordering
            ]
        )
        if not self.fixed_base:
            # Disregard the free joint
            self.motor_indices -= 1

        motor_groups = np.array(
            [self.robot.joint_groups[name] for name in self.robot.motor_ordering]
        )
        self.leg_motor_indices = self.motor_indices[joint_groups == "leg"]
        self.arm_motor_indices = self.motor_indices[joint_groups == "arm"]
        self.neck_motor_indices = self.motor_indices[joint_groups == "neck"]
        self.waist_motor_indices = self.motor_indices[joint_groups == "waist"]

        self.actuator_indices = jnp.array(
            [
                support.name2id(self.sys, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in self.robot.motor_ordering
            ]
        )
        self.leg_actuator_indices = self.actuator_indices[motor_groups == "leg"]
        self.arm_actuator_indices = self.actuator_indices[motor_groups == "arm"]
        self.neck_actuator_indices = self.actuator_indices[motor_groups == "neck"]
        self.waist_actuator_indices = self.actuator_indices[motor_groups == "waist"]
        self.motor_limits = jnp.array(
            [
                self.sys.actuator_ctrlrange[motor_id]
                for motor_id in self.actuator_indices
            ]
        )

        arm_motor_names: List[str] = [
            self.robot.motor_ordering[i] for i in self.arm_actuator_indices
        ]
        self.arm_joint_coef = jnp.ones(len(arm_motor_names), dtype=np.float32)
        for i, motor_name in enumerate(arm_motor_names):
            motor_config = self.robot.config["joints"][motor_name]
            if motor_config["transmission"] == "gears":
                self.arm_joint_coef = self.arm_joint_coef.at[i].set(
                    -motor_config["gear_ratio"]
                )

        self.joint_ref_indices = jnp.arange(len(self.robot.joint_ordering))
        self.leg_ref_indices = self.joint_ref_indices[joint_groups == "leg"]
        self.arm_ref_indices = self.joint_ref_indices[joint_groups == "arm"]
        self.neck_ref_indices = self.joint_ref_indices[joint_groups == "neck"]
        self.waist_ref_indices = self.joint_ref_indices[joint_groups == "waist"]

        # default qpos
        self.default_qpos = jnp.array(self.sys.mj_model.keyframe("home").qpos)
        # default action
        self.default_motor_pos = jnp.array(
            list(self.robot.default_motor_angles.values())
        )
        self.action_scale = self.cfg.action.action_scale
        self.n_steps_delay = self.cfg.action.n_steps_delay
        self.action_smooth_alpha = float(
            self.cfg.action.action_smooth_rate
            / (self.cfg.action.action_smooth_rate + 1 / (self.dt * 2 * jnp.pi))
        )

        # commands
        # x vel, y vel, yaw vel, heading
        self.resample_time = self.cfg.commands.resample_time
        self.resample_steps = int(self.resample_time / self.dt)

        # observation
        self.ref_start_idx = 7 + 6
        self.num_obs_history = self.cfg.obs.frame_stack
        self.num_privileged_obs_history = self.cfg.obs.c_frame_stack
        self.obs_size = self.cfg.obs.num_single_obs
        self.privileged_obs_size = self.cfg.obs.num_single_privileged_obs
        self.obs_scales = self.cfg.obs.scales

        self.q_start_idx = 0 if self.fixed_base else 7
        self.qd_start_idx = 0 if self.fixed_base else 6

        # noise
        self.obs_noise_scale = self.cfg.noise.obs_noise_scale * jnp.concatenate(
            [
                jnp.zeros(self.obs_size - 3 * self.nu - 6),
                jnp.ones_like(self.actuator_indices) * self.cfg.noise.dof_pos,
                jnp.ones_like(self.actuator_indices) * self.cfg.noise.dof_vel,
                jnp.zeros_like(self.actuator_indices),
                # jnp.ones(3) * self.cfg.noise.lin_vel,
                jnp.ones(3) * self.cfg.noise.ang_vel,
                jnp.ones(3) * self.cfg.noise.euler,
            ]
        )
        self.reset_noise_pos = self.cfg.noise.reset_noise_pos

        self.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.push_vel = self.cfg.domain_rand.push_vel

    def _init_reward(self) -> None:
        """Prepares a list of reward functions, which will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """

        reward_scale_dict = asdict(self.cfg.rewards.scales)
        # Remove zero scales and multiply non-zero ones by dt
        for key in list(reward_scale_dict.keys()):
            if reward_scale_dict[key] == 0:
                reward_scale_dict.pop(key)

        # prepare list of functions
        self.reward_names = list(reward_scale_dict.keys())
        self.reward_functions: List[Callable[..., jax.Array]] = []
        self.reward_scales = jnp.zeros(len(reward_scale_dict))
        for i, (name, scale) in enumerate(reward_scale_dict.items()):
            self.reward_functions.append(getattr(self, "_reward_" + name))
            self.reward_scales = self.reward_scales.at[i].set(scale)

        self.healthy_z_range = self.cfg.rewards.healthy_z_range
        self.tracking_sigma = self.cfg.rewards.tracking_sigma
        self.min_feet_distance = self.cfg.rewards.min_feet_distance
        self.max_feet_distance = self.cfg.rewards.max_feet_distance
        self.target_feet_z_delta = self.cfg.rewards.target_feet_z_delta
        self.torso_pitch_range = self.cfg.rewards.torso_pitch_range

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        state_info = {
            "rng": rng,
            "contact_forces": jnp.zeros((self.num_colliders, self.num_colliders, 3)),
            "left_foot_contact_mask": jnp.zeros(len(self.left_foot_collider_indices)),
            "right_foot_contact_mask": jnp.zeros(len(self.right_foot_collider_indices)),
            "stance_mask": jnp.ones(2),
            "last_stance_mask": jnp.ones(2),
            "feet_air_time": jnp.zeros(2),
            "action_buffer": jnp.zeros((self.n_steps_delay + 1) * self.nu),
            "last_last_act": jnp.zeros(self.nu),
            "last_act": jnp.zeros(self.nu),
            "last_torso_euler": jnp.zeros(3),
            "rewards": {k: 0.0 for k in self.reward_names},
            "push": jnp.zeros(2),
            "done": False,
            "step": 0,
        }

        path_pos = jnp.zeros(3)
        path_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        command = self._sample_command(rng2)
        state_info["phase_signal"] = self.motion_ref.get_phase_signal(0.0, command)
        state_ref = self.motion_ref.get_state_ref(path_pos, path_quat, 0.0, command)
        state_info["path_pos"] = path_pos
        state_info["path_quat"] = path_quat
        state_info["command"] = command
        state_info["state_ref"] = jnp.asarray(state_ref)

        qpos = self.default_qpos
        arm_joint_pos = state_ref[self.ref_start_idx + self.arm_ref_indices]
        arm_motor_pos = arm_joint_pos * self.arm_joint_coef
        qpos = qpos.at[self.q_start_idx + self.arm_joint_indices].set(arm_joint_pos)
        qpos = qpos.at[self.q_start_idx + self.arm_motor_indices].set(arm_motor_pos)

        if self.add_noise:
            noise_pos = jax.random.uniform(
                rng1,
                (self.nq - self.q_start_idx,),
                minval=-self.reset_noise_pos,
                maxval=self.reset_noise_pos,
            )
            qpos = qpos.at[self.q_start_idx :].add(noise_pos)

        qvel = jnp.zeros(self.nv)

        pipeline_state = self.pipeline_init(qpos, qvel)

        state_info["last_motor_target"] = pipeline_state.qpos[
            self.q_start_idx + self.motor_indices
        ]
        state_info["init_feet_height"] = pipeline_state.x.pos[self.feet_link_ids, 2]

        obs_history = jnp.zeros(self.num_obs_history * self.obs_size)
        privileged_obs_history = jnp.zeros(
            self.num_privileged_obs_history * self.privileged_obs_size
        )
        obs, privileged_obs = self._get_obs(
            pipeline_state,
            state_info,
            obs_history,
            privileged_obs_history,
        )
        reward, done, zero = jnp.zeros(3)

        metrics: Dict[str, Any] = {}
        for k in self.reward_names:
            metrics[k] = zero

        return State(
            pipeline_state, obs, privileged_obs, reward, done, metrics, state_info
        )

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        rng, cmd_rng, push_rng = jax.random.split(state.info["rng"], 3)

        time_curr = state.info["step"] * self.dt
        path_pos, path_quat = self._integrate_path_frame(state.info)
        phase_signal = self.motion_ref.get_phase_signal(
            time_curr, state.info["command"]
        )
        state_ref = self.motion_ref.get_state_ref(
            path_pos,
            path_quat,
            time_curr,
            state.info["command"],
        )

        state.info["path_pos"] = path_pos
        state.info["path_quat"] = path_quat
        state.info["phase_signal"] = phase_signal
        state.info["state_ref"] = state_ref
        state.info["action_buffer"] = (
            jnp.roll(state.info["action_buffer"], self.nu).at[: self.nu].set(action)
        )

        action_delay: jax.Array = state.info["action_buffer"][-self.nu :]
        motor_target = jnp.where(
            action_delay < 0,
            self.default_motor_pos
            + self.action_scale
            * action_delay
            * (self.default_motor_pos - self.motor_limits[:, 0]),
            self.default_motor_pos
            + self.action_scale
            * action_delay
            * (self.motor_limits[:, 1] - self.default_motor_pos),
        )
        motor_target = self.motion_ref.override_motor_target(motor_target, state_ref)
        motor_target = jnp.clip(
            motor_target, self.motor_limits[:, 0], self.motor_limits[:, 1]
        )
        motor_target = exponential_moving_average(
            self.action_smooth_alpha, motor_target, state.info["last_motor_target"]
        )
        assert isinstance(motor_target, jax.Array)
        state.info["last_motor_target"] = motor_target.copy()

        if self.add_push:
            push_theta = jax.random.uniform(push_rng, maxval=2 * jnp.pi)
            push = jnp.array([jnp.cos(push_theta), jnp.sin(push_theta)])
            push *= jnp.mod(state.info["step"], self.push_interval) == 0
            qvel = state.pipeline_state.qd
            qvel = qvel.at[:2].set(push * self.push_vel + qvel[:2])
            state = state.tree_replace({"pipeline_state.qd": qvel})
            state.info["push"] = push

        # jax.debug.breakpoint()

        pipeline_state = self.pipeline_step(state.pipeline_state, motor_target)

        # jax.debug.print(
        #     "qfrc: {}",
        #     pipeline_state.qfrc_actuator[self.qd_start_idx + self.leg_motor_indices],
        # )
        # jax.debug.print("stance_mask: {}", state.info["stance_mask"])
        # jax.debug.print("feet_air_time: {}", state.info["feet_air_time"])

        if not self.fixed_base:
            contact_forces, left_foot_contact_mask, right_foot_contact_mask = (
                self._get_contact_forces(pipeline_state)
            )
            stance_mask = jnp.array(
                [jnp.any(left_foot_contact_mask), jnp.any(right_foot_contact_mask)]
            ).astype(jnp.float32)

            state.info["contact_forces"] = contact_forces
            state.info["left_foot_contact_mask"] = left_foot_contact_mask
            state.info["right_foot_contact_mask"] = right_foot_contact_mask
            state.info["stance_mask"] = stance_mask

        torso_height = pipeline_state.x.pos[0, 2]
        done = jnp.logical_or(
            torso_height < self.healthy_z_range[0],
            torso_height > self.healthy_z_range[1],
        )
        state.info["done"] = done

        obs, privileged_obs = self._get_obs(
            pipeline_state, state.info, state.obs, state.privileged_obs
        )

        torso_euler = math.quat_to_euler(pipeline_state.x.rot[0])
        torso_euler_delta = torso_euler - state.info["last_torso_euler"]
        torso_euler_delta = (torso_euler_delta + jnp.pi) % (2 * jnp.pi) - jnp.pi
        torso_euler = state.info["last_torso_euler"] + torso_euler_delta

        reward_dict = self._compute_reward(pipeline_state, state.info, action)
        reward = sum(reward_dict.values()) * self.dt
        # reward = jnp.clip(reward, 0.0)

        if not self.fixed_base:
            state.info["last_stance_mask"] = stance_mask.copy()
            state.info["feet_air_time"] += self.dt
            state.info["feet_air_time"] *= 1.0 - stance_mask

        state.info["push"] = push
        state.info["last_last_act"] = state.info["last_act"].copy()
        state.info["last_act"] = action_delay.copy()
        state.info["last_torso_euler"] = torso_euler
        state.info["rewards"] = reward_dict
        state.info["rng"] = rng
        state.info["step"] += 1

        # sample new command if more than 500 timesteps achieved
        state.info["command"] = jnp.where(
            state.info["step"] > self.resample_steps,
            self._sample_command(cmd_rng),
            state.info["command"],
        )

        # reset the step counter when done
        state.info["step"] = jnp.where(
            done | (state.info["step"] > self.resample_steps), 0, state.info["step"]
        )
        state.metrics.update(reward_dict)

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            privileged_obs=privileged_obs,
            reward=reward,
            done=done.astype(jnp.float32),
        )

    def _sample_command(self, rng: jax.Array) -> jax.Array:
        # placeholder
        return jnp.zeros(1)

    def _extract_command(self, command: jax.Array) -> Tuple[jax.Array, jax.Array]:
        # placeholder
        return jnp.zeros(3), jnp.zeros(3)

    def _integrate_path_frame(
        self, info: Dict[str, Any]
    ) -> Tuple[jax.Array, jax.Array]:
        pos = info["path_pos"]
        quat = info["path_quat"]

        lin_vel, ang_vel = self._extract_command(info["command"])

        # Update position
        pos += lin_vel * self.dt

        # Compute the angle of rotation for each axis
        theta_roll = ang_vel[0] * self.dt / 2.0
        theta_pitch = ang_vel[1] * self.dt / 2.0
        theta_yaw = ang_vel[2] * self.dt / 2.0

        # Compute the quaternion for each rotational axis
        roll_quat = jnp.array([jnp.cos(theta_roll), jnp.sin(theta_roll), 0.0, 0.0])
        pitch_quat = jnp.array([jnp.cos(theta_pitch), 0.0, jnp.sin(theta_pitch), 0.0])
        yaw_quat = jnp.array([jnp.cos(theta_yaw), 0.0, 0.0, jnp.sin(theta_yaw)])

        # Normalize each quaternion
        roll_quat /= jnp.linalg.norm(roll_quat)
        pitch_quat /= jnp.linalg.norm(pitch_quat)
        yaw_quat /= jnp.linalg.norm(yaw_quat)

        # Combine the quaternions to get the full rotation (roll * pitch * yaw)
        full_quat = math.quat_mul(math.quat_mul(roll_quat, pitch_quat), yaw_quat)

        # Update the current quaternion by applying the new rotation
        quat = math.quat_mul(quat, full_quat)
        quat /= jnp.linalg.norm(quat)

        return pos, quat

    def _get_contact_forces(self, data: mjx.Data):
        # Extract geom1 and geom2 directly
        geom1 = data.contact.geom1
        geom2 = data.contact.geom2

        def get_body_index(geom_id: jax.Array) -> jax.Array:
            return jnp.argmax(self.collider_geom_ids == geom_id)

        # Vectorized computation of body indices for geom1 and geom2
        body_indices_1 = jax.vmap(get_body_index)(geom1)
        body_indices_2 = jax.vmap(get_body_index)(geom2)

        contact_forces_global = jnp.zeros((self.num_colliders, self.num_colliders, 3))
        for i in range(data.ncon):
            contact_force = self.jit_contact_force(self.sys, data, i, True)[:3]
            # Update the contact forces for both body_indices_1 and body_indices_2
            # Add instead of set to accumulate forces from multiple contacts
            contact_forces_global = contact_forces_global.at[
                body_indices_1[i], body_indices_2[i]
            ].add(contact_force)
            contact_forces_global = contact_forces_global.at[
                body_indices_2[i], body_indices_1[i]
            ].add(contact_force)

        left_foot_contact_mask = (
            contact_forces_global[0, self.left_foot_collider_indices, 2]
            > self.contact_force_threshold
        ).astype(jnp.float32)
        right_foot_contact_mask = (
            contact_forces_global[0, self.right_foot_collider_indices, 2]
            > self.contact_force_threshold
        ).astype(jnp.float32)

        return contact_forces_global, left_foot_contact_mask, right_foot_contact_mask

    def _get_obs(
        self,
        pipeline_state: base.State,
        info: dict[str, Any],
        obs_history: jax.Array,
        privileged_obs_history: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Observes humanoid body position, velocities, and angles."""
        motor_pos = pipeline_state.q[self.q_start_idx + self.motor_indices]
        motor_pos_delta = (
            motor_pos - self.default_qpos[self.q_start_idx + self.motor_indices]
        )
        motor_vel = pipeline_state.qd[self.qd_start_idx + self.motor_indices]

        joint_pos = pipeline_state.q[self.q_start_idx + self.joint_indices]
        joint_pos_error = (
            joint_pos
            - info["state_ref"][self.ref_start_idx : self.ref_start_idx + self.nu]
        )

        torso_quat = pipeline_state.x.rot[0]
        torso_lin_vel = math.rotate(pipeline_state.xd.vel[0], math.quat_inv(torso_quat))
        torso_ang_vel = math.rotate(pipeline_state.xd.ang[0], math.quat_inv(torso_quat))

        torso_euler = math.quat_to_euler(torso_quat)
        torso_euler_delta = torso_euler - info["last_torso_euler"]
        torso_euler_delta = (torso_euler_delta + jnp.pi) % (2 * jnp.pi) - jnp.pi
        torso_euler = info["last_torso_euler"] + torso_euler_delta

        obs = jnp.concatenate(
            [
                info["phase_signal"],
                info["command"],
                motor_pos_delta * self.obs_scales.dof_pos,
                motor_vel * self.obs_scales.dof_vel,
                info["last_act"],
                # torso_lin_vel * self.obs_scales.lin_vel,
                torso_ang_vel * self.obs_scales.ang_vel,
                torso_euler * self.obs_scales.euler,
            ]
        )
        privileged_obs = jnp.concatenate(
            [
                info["phase_signal"],
                info["command"],
                motor_pos_delta * self.obs_scales.dof_pos,
                motor_vel * self.obs_scales.dof_vel,
                info["last_act"],
                torso_lin_vel * self.obs_scales.lin_vel,
                torso_ang_vel * self.obs_scales.ang_vel,
                torso_euler * self.obs_scales.euler,
                joint_pos_error,
                info["stance_mask"],
                info["state_ref"][-2:],
                info["push"],
            ]
        )

        if self.add_noise:
            obs += self.obs_noise_scale * jax.random.uniform(
                info["rng"], obs.shape, minval=-1, maxval=1
            )

        # jax.debug.breakpoint()

        # obs = jnp.clip(obs, -100.0, 100.0)
        # stack observations through time
        obs = jnp.roll(obs_history, obs.size).at[: obs.size].set(obs)

        privileged_obs = (
            jnp.roll(privileged_obs_history, privileged_obs.size)
            .at[: privileged_obs.size]
            .set(privileged_obs)
        )

        return obs, privileged_obs

    def _compute_reward(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        # Create an array of indices to map over
        indices = jnp.arange(len(self.reward_names))
        # Use jax.lax.map to compute rewards
        reward_arr = jax.lax.map(
            lambda i: jax.lax.switch(
                i,
                self.reward_functions,
                pipeline_state,
                info,
                action,
            )
            * self.reward_scales[i],
            indices,
        )

        reward_dict: Dict[str, jax.Array] = {}
        for i, name in enumerate(self.reward_names):
            reward_dict[name] = reward_arr[i]

        return reward_dict

    def _reward_torso_pos(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Reward for track torso position"""
        torso_pos = pipeline_state.x.pos[0, :2]  # Assuming [:2] extracts xy components
        torso_pos_ref = info["state_ref"][:2]
        error = jnp.linalg.norm(torso_pos - torso_pos_ref, axis=-1)
        reward = jnp.exp(-200.0 * error**2)
        return reward

    def _reward_torso_quat(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Reward for track torso orientation"""
        torso_quat = pipeline_state.x.rot[0]
        torso_quat_ref = info["state_ref"][3:7]
        # Quaternion dot product (cosine of the half-angle)
        dot_product = jnp.sum(torso_quat * torso_quat_ref, axis=-1)
        # Ensure the dot product is within the valid range
        dot_product = jnp.clip(dot_product, -1.0, 1.0)
        # Quaternion angle difference
        angle_diff = 2.0 * jnp.arccos(jnp.abs(dot_product))
        reward = jnp.exp(-20.0 * (angle_diff**2))
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

    def _reward_lin_vel_xy(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Reward for track linear velocity in xy"""
        lin_vel_local = math.rotate(
            pipeline_state.xd.vel[0], math.quat_inv(pipeline_state.x.rot[0])
        )
        lin_vel_xy = lin_vel_local[:2]
        lin_vel_xy_ref = info["state_ref"][7:9]
        error = jnp.linalg.norm(lin_vel_xy - lin_vel_xy_ref, axis=-1)
        reward = jnp.exp(-self.tracking_sigma * error**2)
        return reward

    def _reward_lin_vel_z(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Reward for track linear velocity in z"""
        lin_vel_local = math.rotate(
            pipeline_state.xd.vel[0], math.quat_inv(pipeline_state.x.rot[0])
        )
        lin_vel_z = lin_vel_local[2]
        lin_vel_z_ref = info["state_ref"][9]
        error = lin_vel_z - lin_vel_z_ref
        reward = jnp.exp(-self.tracking_sigma * error**2)
        return reward

    def _reward_ang_vel_xy(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Reward for track angular velocity in xy"""
        ang_vel_local = math.rotate(
            pipeline_state.xd.ang[0], math.quat_inv(pipeline_state.x.rot[0])
        )
        ang_vel_xy = ang_vel_local[:2]
        ang_vel_xy_ref = info["state_ref"][10:12]
        error = jnp.linalg.norm(ang_vel_xy - ang_vel_xy_ref, axis=-1)
        reward = jnp.exp(-self.tracking_sigma / 4 * error**2)
        return reward

    def _reward_ang_vel_z(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Reward for track angular velocity in z"""
        ang_vel_local = math.rotate(
            pipeline_state.xd.ang[0], math.quat_inv(pipeline_state.x.rot[0])
        )
        ang_vel_z = ang_vel_local[2]
        ang_vel_z_ref = info["state_ref"][12]
        error = ang_vel_z - ang_vel_z_ref
        reward = jnp.exp(-self.tracking_sigma / 4 * error**2)
        return reward

    def _reward_feet_contact(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Reward for contact"""
        reward = jnp.sum(info["stance_mask"] == info["state_ref"][-2:]).astype(
            jnp.float32
        )
        return reward

    def _reward_feet_contact_number(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Reward for contact"""
        left_contact_numbers = jnp.sum(info["left_foot_contact_mask"])
        right_contact_numbers = jnp.sum(info["right_foot_contact_mask"])
        contact_numbers = jnp.array([left_contact_numbers, right_contact_numbers])
        reward = jnp.sum(contact_numbers * info["state_ref"][-2:]).astype(jnp.float32)
        return reward

    def _reward_leg_joint_pos(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking leg joint positions"""
        joint_pos = pipeline_state.q[self.q_start_idx + self.leg_joint_indices]
        joint_pos_ref = info["state_ref"][self.ref_start_idx + self.leg_ref_indices]
        error = joint_pos - joint_pos_ref
        reward = -jnp.mean(error**2)
        return reward

    def _reward_leg_joint_vel(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking leg joint velocities"""
        joint_vel = pipeline_state.qd[self.qd_start_idx + self.leg_joint_indices]
        joint_vel_ref = info["state_ref"][
            self.ref_start_idx + self.nu + self.leg_ref_indices
        ]
        error = joint_vel - joint_vel_ref
        reward = -jnp.mean(error**2)
        return reward

    def _reward_arm_joint_pos(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking arm joint positions"""
        joint_pos = pipeline_state.q[self.q_start_idx + self.arm_joint_indices]
        joint_pos_ref = info["state_ref"][self.ref_start_idx + self.arm_ref_indices]
        error = joint_pos - joint_pos_ref
        reward = -jnp.mean(error**2)
        return reward

    def _reward_arm_joint_vel(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking arm joint velocities"""
        joint_vel = pipeline_state.qd[self.qd_start_idx + self.arm_joint_indices]
        joint_vel_ref = info["state_ref"][
            self.ref_start_idx + self.nu + self.arm_ref_indices
        ]
        error = joint_vel - joint_vel_ref
        reward = -jnp.mean(error**2)
        return reward

    def _reward_neck_joint_pos(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking neck joint positions"""
        joint_pos = pipeline_state.q[self.q_start_idx + self.neck_joint_indices]
        joint_pos_ref = info["state_ref"][self.ref_start_idx + self.neck_ref_indices]
        error = joint_pos - joint_pos_ref
        reward = -jnp.mean(error**2)
        return reward

    def _reward_neck_joint_vel(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking neck joint velocities"""
        joint_vel = pipeline_state.qd[self.qd_start_idx + self.neck_joint_indices]
        joint_vel_ref = info["state_ref"][
            self.ref_start_idx + self.nu + self.neck_ref_indices
        ]
        error = joint_vel - joint_vel_ref
        reward = -jnp.mean(error**2)
        return reward

    def _reward_waist_joint_pos(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking waist joint positions"""
        joint_pos = pipeline_state.q[self.q_start_idx + self.waist_joint_indices]
        joint_pos_ref = info["state_ref"][self.ref_start_idx + self.waist_ref_indices]
        error = joint_pos - joint_pos_ref
        reward = -jnp.mean(error**2)
        return reward

    def _reward_waist_joint_vel(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking waist joint velocities"""
        joint_vel = pipeline_state.qd[self.qd_start_idx + self.waist_joint_indices]
        joint_vel_ref = info["state_ref"][
            self.ref_start_idx + self.nu + self.waist_ref_indices
        ]
        error = joint_vel - joint_vel_ref
        reward = -jnp.mean(error**2)
        return reward

    def _reward_collision(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        collision_forces = jnp.linalg.norm(
            info["contact_forces"][1:, 1:],  # exclude the floor
            axis=-1,
        )
        collision_contact = collision_forces > 0.1
        reward = -jnp.sum(collision_contact.astype(jnp.float32))
        return reward

    def _reward_motor_torque(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for minimizing joint torques"""
        torque = pipeline_state.qfrc_actuator[self.motor_indices]
        error = jnp.square(torque)
        reward = -jnp.mean(error)
        return reward

    def _reward_joint_acc(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for minimizing joint accelerations"""
        joint_acc = pipeline_state.qacc[self.qd_start_idx + self.joint_indices]
        error = jnp.square(joint_acc)
        reward = -jnp.mean(error)
        return reward

    def _reward_leg_action_rate(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking leg action rates"""
        leg_action = action[self.leg_actuator_indices]
        last_leg_action = info["last_act"][self.leg_actuator_indices]
        error = jnp.square(leg_action - last_leg_action)
        reward = -jnp.mean(error)
        return reward

    def _reward_leg_action_acc(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking leg action accelerations"""
        leg_action = action[self.leg_actuator_indices]
        last_leg_action = info["last_act"][self.leg_actuator_indices]
        last_last_leg_action = info["last_last_act"][self.leg_actuator_indices]
        error = jnp.square(leg_action - 2 * last_leg_action + last_last_leg_action)
        reward = -jnp.mean(error)
        return reward

    def _reward_arm_action_rate(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking arm action rates"""
        arm_action = action[self.arm_actuator_indices]
        last_arm_action = info["last_act"][self.arm_actuator_indices]
        error = jnp.square(arm_action - last_arm_action)
        reward = -jnp.mean(error)
        return reward

    def _reward_arm_action_acc(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking arm action accelerations"""
        arm_action = action[self.arm_actuator_indices]
        last_arm_action = info["last_act"][self.arm_actuator_indices]
        last_last_arm_action = info["last_last_act"][self.arm_actuator_indices]
        error = jnp.square(arm_action - 2 * last_arm_action + last_last_arm_action)
        reward = -jnp.mean(error)
        return reward

    def _reward_neck_action_rate(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking neck action rates"""
        neck_action = action[self.neck_actuator_indices]
        last_neck_action = info["last_act"][self.neck_actuator_indices]
        error = jnp.square(neck_action - last_neck_action)
        reward = -jnp.mean(error)
        return reward

    def _reward_neck_action_acc(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking neck action accelerations"""
        neck_action = action[self.neck_actuator_indices]
        last_neck_action = info["last_act"][self.neck_actuator_indices]
        last_last_neck_action = info["last_last_act"][self.neck_actuator_indices]
        error = jnp.square(neck_action - 2 * last_neck_action + last_last_neck_action)
        reward = -jnp.mean(error)
        return reward

    def _reward_waist_action_rate(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking waist action rates"""
        waist_action = action[self.waist_actuator_indices]
        last_waist_action = info["last_act"][self.waist_actuator_indices]
        error = jnp.square(waist_action - last_waist_action)
        reward = -jnp.mean(error)
        return reward

    def _reward_waist_action_acc(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking waist action accelerations"""
        waist_action = action[self.waist_actuator_indices]
        last_waist_action = info["last_act"][self.waist_actuator_indices]
        last_last_waist_action = info["last_last_act"][self.waist_actuator_indices]
        error = jnp.square(
            waist_action - 2 * last_waist_action + last_last_waist_action
        )
        reward = -jnp.mean(error)
        return reward

    def _reward_survival(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        return -(info["done"] & (info["step"] < self.resample_steps)).astype(
            jnp.float32
        )
