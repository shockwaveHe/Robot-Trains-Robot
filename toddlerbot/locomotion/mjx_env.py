from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import jax
import mujoco
import numpy as np
import scipy
from brax import base, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import support  # type: ignore

from toddlerbot.actuation.mujoco.mujoco_control import MotorController
from toddlerbot.locomotion.mjx_config import MJXConfig
from toddlerbot.motion.motion_ref import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.math_utils import (
    butterworth,
    exponential_moving_average,
)

# Global registry to store env names and their corresponding classes
env_registry: Dict[str, Type["MJXEnv"]] = {}


def get_env_class(env_name: str) -> Type["MJXEnv"]:
    if env_name not in env_registry:
        raise ValueError(f"Unknown env: {env_name}")

    return env_registry[env_name]


def get_env_names() -> List[str]:
    return list(env_registry.keys())


class MJXEnv(PipelineEnv):
    def __init__(
        self,
        name: str,
        robot: Robot,
        cfg: MJXConfig,
        motion_ref: MotionReference,
        fixed_base: bool = False,
        add_noise: bool = True,
        add_domain_rand: bool = True,
        **kwargs: Any,
    ):
        self.name = name
        self.cfg = cfg
        self.robot = robot
        self.motion_ref = motion_ref
        self.fixed_base = fixed_base
        self.add_noise = add_noise
        self.add_domain_rand = add_domain_rand

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

    # Automatic registration of subclasses
    def __init_subclass__(cls, env_name: str = "", **kwargs):
        super().__init_subclass__(**kwargs)
        if len(env_name) > 0:
            env_registry[env_name] = cls

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

        feet_link_mask = jnp.array(
            np.char.find(self.sys.link_names, self.robot.foot_name) >= 0
        )
        self.feet_link_ids = jnp.arange(self.sys.num_links())[feet_link_mask]

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
            [self.robot.joint_limits[name] for name in self.robot.motor_ordering]
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

        self.controller = MotorController(self.robot)

        # Filter
        self.filter_type = self.cfg.action.filter_type
        self.filter_order = self.cfg.action.filter_order
        # EMA
        self.ema_alpha = float(
            self.cfg.action.filter_cutoff
            / (self.cfg.action.filter_cutoff + 1 / (self.dt * 2 * jnp.pi))
        )
        # Butterworth
        b, a = scipy.signal.butter(
            self.filter_order,
            self.cfg.action.filter_cutoff / (0.5 / self.dt),
            btype="low",
            analog=False,
        )
        self.butter_b_coef = jnp.array(b)[:, None]
        self.butter_a_coef = jnp.array(a)[:, None]

        # commands
        # x vel, y vel, yaw vel, heading
        self.resample_time = self.cfg.commands.resample_time
        self.resample_steps = int(self.resample_time / self.dt)
        self.reset_time = self.cfg.commands.reset_time
        self.reset_steps = int(self.reset_time / self.dt)
        self.mean_reversion = self.cfg.commands.mean_reversion
        self.command_range = jnp.array(self.cfg.commands.command_range)
        self.deadzone = (
            jnp.array(self.cfg.commands.deadzone)
            if len(self.cfg.commands.deadzone) > 1
            else self.cfg.commands.deadzone[0]
        )
        self.command_obs_indices = jnp.array(self.cfg.commands.command_obs_indices)

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
        self.reset_noise_joint_pos = self.cfg.noise.reset_noise_joint_pos
        self.reset_noise_torso_pitch = self.cfg.noise.reset_noise_torso_pitch
        self.backlash_scale = self.cfg.noise.backlash_scale
        self.backlash_activation = self.cfg.noise.backlash_activation

        self.kp_range = self.cfg.domain_rand.kp_range
        self.kd_range = self.cfg.domain_rand.kd_range
        self.tau_max_range = self.cfg.domain_rand.tau_max_range
        self.q_dot_tau_max_range = self.cfg.domain_rand.q_dot_tau_max_range
        self.q_dot_max_range = self.cfg.domain_rand.q_dot_max_range

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

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        (
            rng,
            rng_torso_pitch,
            rng_torso_yaw,
            rng_joint_pos,
            rng_command,
            rng_kp,
            rng_kd,
            rng_q_dot_tau_max,
            rng_q_dot_max,
        ) = jax.random.split(rng, 9)

        state_info = {
            "rng": rng,
            "contact_forces": jnp.zeros((self.num_colliders, self.num_colliders, 3)),
            "left_foot_contact_mask": jnp.zeros(len(self.left_foot_collider_indices)),
            "right_foot_contact_mask": jnp.zeros(len(self.right_foot_collider_indices)),
            "feet_air_time": jnp.zeros(2),
            "feet_air_dist": jnp.zeros(2),
            "action_buffer": jnp.zeros((self.n_steps_delay + 1) * self.nu),
            "last_last_act": jnp.zeros(self.nu),
            "last_act": jnp.zeros(self.nu),
            "last_torso_euler": jnp.zeros(3),
            "rewards": {k: 0.0 for k in self.reward_names},
            "push": jnp.zeros(2),
            "done": False,
            "step": 0,
        }

        qpos = self.default_qpos
        qvel = jnp.zeros(self.nv)

        torso_pos = qpos[:3]
        torso_yaw = jax.random.uniform(rng_torso_yaw, (1,), minval=0, maxval=2 * jnp.pi)
        torso_quat = math.euler_to_quat(
            jnp.array([0.0, 0.0, jnp.degrees(torso_yaw)[0]])
        )
        torso_lin_vel = jnp.zeros(3)
        torso_ang_vel = jnp.zeros(3)

        joint_pos = qpos[self.q_start_idx + self.joint_indices]
        joint_vel = qvel[self.qd_start_idx + self.joint_indices]
        stance_mask = jnp.ones(2)

        state_ref_init = jnp.concatenate(
            [
                torso_pos,
                torso_quat,
                torso_lin_vel,
                torso_ang_vel,
                joint_pos,
                joint_vel,
                stance_mask,
            ]
        )
        command = self._sample_command(rng_command)
        state_ref = jnp.asarray(
            self.motion_ref.get_state_ref(state_ref_init, 0.0, command)
        )

        neck_joint_pos = state_ref[self.ref_start_idx + self.neck_ref_indices]
        neck_motor_pos = self.motion_ref.neck_ik(neck_joint_pos)
        arm_joint_pos = state_ref[self.ref_start_idx + self.arm_ref_indices]
        arm_motor_pos = self.motion_ref.arm_ik(arm_joint_pos)
        waist_joint_pos = state_ref[self.ref_start_idx + self.waist_ref_indices]
        waist_motor_pos = self.motion_ref.waist_ik(waist_joint_pos)

        qpos = qpos.at[self.q_start_idx + self.neck_joint_indices].set(neck_joint_pos)
        qpos = qpos.at[self.q_start_idx + self.neck_motor_indices].set(neck_motor_pos)
        qpos = qpos.at[self.q_start_idx + self.arm_joint_indices].set(arm_joint_pos)
        qpos = qpos.at[self.q_start_idx + self.arm_motor_indices].set(arm_motor_pos)
        qpos = qpos.at[self.q_start_idx + self.waist_joint_indices].set(waist_joint_pos)
        qpos = qpos.at[self.q_start_idx + self.waist_motor_indices].set(waist_motor_pos)

        if self.add_noise:
            if not self.fixed_base:
                noise_torso_pitch = self.reset_noise_torso_pitch * jax.random.normal(
                    rng_torso_pitch, (1,)
                )
                torso_euler = jnp.array([0.0, noise_torso_pitch[0], torso_yaw[0]])
                torso_quat = math.euler_to_quat(jnp.degrees(torso_euler))

            noise_joint_pos = self.reset_noise_joint_pos * jax.random.normal(
                rng_joint_pos, (self.nq - self.q_start_idx,)
            )
            qpos = qpos.at[self.q_start_idx :].add(noise_joint_pos)

        waist_joint_pos = qpos[self.q_start_idx + self.waist_joint_indices]
        waist_euler = jnp.array([-waist_joint_pos[0], 0.0, -waist_joint_pos[1]])
        waist_quat = math.euler_to_quat(jnp.degrees(waist_euler))
        torso_quat = math.quat_mul(torso_quat, waist_quat)

        state_ref = state_ref.at[3:7].set(torso_quat)
        qpos = qpos.at[3:7].set(torso_quat)

        # jax.debug.print("euler: {}", math.quat_to_euler(torso_quat))
        # jax.debug.print("torso_euler: {}", torso_euler)
        # jax.debug.print("waist_euler: {}", waist_euler)

        pipeline_state = self.pipeline_init(qpos, qvel)

        state_info["command"] = command
        state_info["state_ref"] = state_ref
        state_info["stance_mask"] = state_ref[-2:]
        state_info["last_stance_mask"] = state_ref[-2:]
        state_info["phase_signal"] = self.motion_ref.get_phase_signal(0.0)
        state_info["feet_height_init"] = pipeline_state.x.pos[self.feet_link_ids, 2]
        last_motor_target = pipeline_state.qpos[self.q_start_idx + self.motor_indices]
        state_info["last_motor_target"] = last_motor_target
        state_info["butter_past_inputs"] = jnp.tile(
            last_motor_target, (self.filter_order, 1)
        )
        state_info["butter_past_outputs"] = jnp.tile(
            last_motor_target, (self.filter_order, 1)
        )
        state_info["controller_kp"] = self.controller.kp.copy()
        state_info["controller_kd"] = self.controller.kd.copy()
        state_info["controller_tau_max"] = self.controller.tau_max.copy()
        state_info["controller_q_dot_tau_max"] = self.controller.q_dot_tau_max.copy()
        state_info["controller_q_dot_max"] = self.controller.q_dot_max.copy()

        if self.add_domain_rand:
            state_info["controller_kp"] *= jax.random.uniform(
                rng_kp, (self.nu,), minval=self.kp_range[0], maxval=self.kp_range[1]
            )
            state_info["controller_kd"] *= jax.random.uniform(
                rng_kd, (self.nu,), minval=self.kd_range[0], maxval=self.kd_range[1]
            )
            state_info["controller_tau_max"] *= jax.random.uniform(
                rng_kd,
                (self.nu,),
                minval=self.tau_max_range[0],
                maxval=self.tau_max_range[1],
            )
            state_info["controller_q_dot_tau_max"] *= jax.random.uniform(
                rng_q_dot_tau_max,
                (self.nu,),
                minval=self.q_dot_tau_max_range[0],
                maxval=self.q_dot_tau_max_range[1],
            )
            state_info["controller_q_dot_max"] *= jax.random.uniform(
                rng_q_dot_max,
                (self.nu,),
                minval=self.q_dot_max_range[0],
                maxval=self.q_dot_max_range[1],
            )

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

    def pipeline_step(self, state: State, action: jax.Array) -> base.State:
        """Takes a physics step using the physics pipeline."""

        def f(pipeline_state, _):
            ctrl = self.controller.step(
                pipeline_state.q[self.q_start_idx + self.motor_indices],
                pipeline_state.qd[self.qd_start_idx + self.motor_indices],
                action,
                state.info["controller_kp"],
                state.info["controller_kd"],
                state.info["controller_tau_max"],
                state.info["controller_q_dot_tau_max"],
                state.info["controller_q_dot_max"],
            )
            return (
                self._pipeline.step(self.sys, pipeline_state, ctrl, self._debug),
                None,
            )

        return jax.lax.scan(f, state.pipeline_state, (), self._n_frames)[0]

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        rng, cmd_rng, push_rng = jax.random.split(state.info["rng"], 3)

        time_curr = state.info["step"] * self.dt
        state_ref = self.motion_ref.get_state_ref(
            state.info["state_ref"], time_curr, state.info["command"]
        )
        state.info["state_ref"] = state_ref
        state.info["phase_signal"] = self.motion_ref.get_phase_signal(time_curr)
        state.info["action_buffer"] = (
            jnp.roll(state.info["action_buffer"], self.nu).at[: self.nu].set(action)
        )

        action_delay: jax.Array = state.info["action_buffer"][-self.nu :]
        motor_target = self.default_motor_pos + self.action_scale * action_delay
        motor_target = self.motion_ref.override_motor_target(motor_target, state_ref)

        if self.filter_type == "ema":
            motor_target = exponential_moving_average(
                self.ema_alpha, motor_target, state.info["last_motor_target"]
            )
        elif self.filter_type == "butter":
            (
                motor_target,
                state.info["butter_past_inputs"],
                state.info["butter_past_outputs"],
            ) = butterworth(
                self.butter_b_coef,
                self.butter_a_coef,
                motor_target,
                state.info["butter_past_inputs"],
                state.info["butter_past_outputs"],
            )

        motor_target = jnp.clip(
            motor_target, self.motor_limits[:, 0], self.motor_limits[:, 1]
        )

        assert isinstance(motor_target, jax.Array)
        state.info["last_motor_target"] = motor_target.copy()

        if self.add_domain_rand:
            push_theta = jax.random.uniform(push_rng, maxval=2 * jnp.pi)
            push = jnp.array([jnp.cos(push_theta), jnp.sin(push_theta)])
            push *= jnp.mod(state.info["step"], self.push_interval) == 0
            qvel = state.pipeline_state.qd
            qvel = qvel.at[:2].set(push * self.push_vel + qvel[:2])
            state = state.tree_replace({"pipeline_state.qd": qvel})
            state.info["push"] = push

        # jax.debug.breakpoint()
        pipeline_state = self.pipeline_step(state, motor_target)

        # jax.debug.print(
        #     "qfrc: {}",
        #     pipeline_state.qfrc_actuator[self.qd_start_idx + self.leg_motor_indices],
        # )
        # jax.debug.print("stance_mask: {}", state.info["stance_mask"])
        # jax.debug.print("feet_air_time: {}", state.info["feet_air_time"])
        # jax.debug.print("feet_air_dist: {}", state.info["feet_air_dist"])

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

            feet_z_delta = (
                pipeline_state.x.pos[self.feet_link_ids, 2]
                - state.info["feet_height_init"]
            )
            state.info["feet_air_dist"] += feet_z_delta
            state.info["feet_air_dist"] *= 1.0 - stance_mask

        state.info["last_last_act"] = state.info["last_act"].copy()
        state.info["last_act"] = action_delay.copy()
        state.info["last_torso_euler"] = torso_euler
        state.info["rewards"] = reward_dict
        state.info["rng"] = rng
        state.info["step"] += 1

        # jax.debug.print("step: {}", state.info["step"])

        state.info["command"] = jax.lax.cond(
            state.info["step"] % self.resample_steps == 0,
            lambda: self._sample_command(cmd_rng, state.info["command"]),
            lambda: state.info["command"],
        )

        # reset the step counter when done
        state.info["step"] = jnp.where(
            done | (state.info["step"] > self.reset_steps), 0, state.info["step"]
        )
        state.metrics.update(reward_dict)

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            privileged_obs=privileged_obs,
            reward=reward,
            done=done.astype(jnp.float32),
        )

    def _sample_command(
        self, rng: jax.Array, last_command: Optional[jax.Array] = None
    ) -> jax.Array:
        raise NotImplementedError

    def _sample_command_uniform(
        self, rng: jax.Array, command_range: jax.Array
    ) -> jax.Array:
        return jax.random.uniform(
            rng,
            (command_range.shape[0],),
            minval=command_range[:, 0],
            maxval=command_range[:, 1],
        )

    def _sample_command_normal(
        self, rng: jax.Array, command_range: jax.Array
    ) -> jax.Array:
        return jnp.clip(
            jax.random.normal(rng, (command_range.shape[0],))
            * command_range[:, 1]
            / 3.0,
            command_range[:, 0],
            command_range[:, 1],
        )

    def _sample_command_normal_reversion(
        self, rng: jax.Array, command_range: jax.Array, last_command: jax.Array
    ) -> jax.Array:
        return jnp.clip(
            jax.random.normal(rng, (command_range.shape[0],))
            * command_range[:, 1]
            / 3.0
            - self.mean_reversion * last_command,
            command_range[:, 0],
            command_range[:, 1],
        )

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
        motor_backlash = self.backlash_scale * jnp.tanh(
            pipeline_state.qfrc_actuator[self.motor_indices] / self.backlash_activation
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
                info["command"][self.command_obs_indices],
                motor_pos_delta * self.obs_scales.dof_pos + motor_backlash,
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
                info["command"][self.command_obs_indices],
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
            obs += self.obs_noise_scale * jax.random.normal(info["rng"], obs.shape)

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
        torso_pos = pipeline_state.x.pos[0]  # Assuming [:2] extracts xy components
        torso_pos_ref = info["state_ref"][:3]
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
        return -(info["done"] & (info["step"] < self.reset_steps)).astype(jnp.float32)
