from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import mujoco  # type: ignore
import numpy as np
from brax import base, math  # type: ignore
from brax.envs.base import PipelineEnv, State  # type: ignore
from brax.io import mjcf  # type: ignore
from jax import numpy as jnp
from mujoco import mjx  # type: ignore
from mujoco.mjx._src import support  # type: ignore

from toddlerbot.envs.mjx_config import MJXConfig
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_robot_file_path


class MuJoCoEnv(PipelineEnv):
    def __init__(
        self,
        name: str,
        cfg: MJXConfig,
        robot: Robot,
        ref_motion_name: str = "walk_simple",
        fixed_base: bool = False,
        fixed_command: Optional[jax.Array] = None,
        add_noise: bool = True,
        **kwargs: Any,
    ):
        self.name = name
        self.cfg = cfg
        self.robot = robot
        self.ref_motion_name = ref_motion_name
        self.fixed_base = fixed_base
        self.fixed_command = fixed_command
        self.add_noise = add_noise

        if fixed_base:
            xml_path = find_robot_file_path(robot.name, suffix="_fixed_scene.xml")
        else:
            xml_path = find_robot_file_path(robot.name, suffix="_scene.xml")

        sys = mjcf.load(xml_path)  # type: ignore
        sys = sys.tree_replace(  # type: ignore
            {
                "opt.timestep": cfg.mj.timestep,
                "opt.solver": cfg.mj.solver,
                "opt.iterations": cfg.mj.iterations,
                "opt.ls_iterations": cfg.mj.ls_iterations,
            }
        )

        kwargs["n_frames"] = cfg.action.n_frames
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)  # type: ignore

        self._init_env()
        self._init_reward()

    def _init_env(self):
        self.nu = self.sys.nu  # type:ignore
        self.nq = self.sys.nq  # type:ignore
        self.nv = self.sys.nv  # type:ignore

        # colliders
        collider_names = ["floor"] + self.robot.collider_names
        self.num_colliders = len(collider_names)
        indices = jnp.arange(self.num_colliders)  # type:ignore

        foot_mask = jnp.array(  # type:ignore
            np.char.find(collider_names, self.robot.foot_name) >= 0
        )
        self.feet_indices = indices[foot_mask]
        self.collision_contact_indices = indices[~foot_mask]

        self.collider_link_ids = jnp.zeros(self.num_colliders, dtype=jnp.int32)  # type:ignore
        for link_id, link_name in enumerate(self.sys.link_names):  # type:ignore
            if link_name in collider_names:
                self.collider_link_ids = self.collider_link_ids.at[
                    collider_names.index(link_name)
                ].set(link_id)  # type:ignore

        self.feet_link_ids = self.collider_link_ids[self.feet_indices]

        self.collider_geom_ids = jnp.zeros(self.num_colliders, dtype=jnp.int32)  # type:ignore
        for geom_id in range(self.sys.ngeom):  # type:ignore
            geom_name = support.id2name(self.sys, mujoco.mjtObj.mjOBJ_GEOM, geom_id)  # type:ignore
            if geom_name is None:
                continue

            if "floor" in geom_name:
                body_name = "floor"
            elif "collision" in geom_name:
                link_id = self.sys.geom_bodyid[geom_id]  # type:ignore
                body_name = support.id2name(self.sys, mujoco.mjtObj.mjOBJ_BODY, link_id)  # type:ignore
            else:
                continue

            if body_name in collider_names:
                self.collider_geom_ids = self.collider_geom_ids.at[
                    collider_names.index(body_name)
                ].set(geom_id)  # type:ignore

        self.contact_force_threshold = self.cfg.action.contact_force_threshold
        # This leads to CPU memory leak
        # self.jit_contact_force = jax.jit(support.contact_force, static_argnums=(2, 3))  # type:ignore
        self.jit_contact_force = support.contact_force

        # joint indices
        self.joint_indices = jnp.array(  # type:ignore
            [
                support.name2id(self.sys, mujoco.mjtObj.mjOBJ_JOINT, name)  # type:ignore
                for name in self.robot.joint_ordering
            ]
        )
        if not self.fixed_base:
            # Disregard the free joint
            self.joint_indices -= 1

        joint_groups = np.array(  # type:ignore
            [self.robot.joint_groups[name] for name in self.robot.joint_ordering]
        )
        self.leg_joint_indices = self.joint_indices[joint_groups == "leg"]
        self.arm_joint_indices = self.joint_indices[joint_groups == "arm"]
        self.neck_joint_indices = self.joint_indices[joint_groups == "neck"]
        self.waist_joint_indices = self.joint_indices[joint_groups == "waist"]

        motor_indices = np.array(  # type:ignore
            [
                support.name2id(self.sys, mujoco.mjtObj.mjOBJ_ACTUATOR, name)  # type:ignore
                for name in self.robot.motor_ordering
            ]
        )
        self.motor_indices = jnp.array(motor_indices)  # type:ignore
        motor_groups = np.array(  # type:ignore
            [self.robot.joint_groups[name] for name in self.robot.motor_ordering]
        )
        self.leg_motor_indices = self.motor_indices[motor_groups == "leg"]
        self.arm_motor_indices = self.motor_indices[motor_groups == "arm"]
        self.neck_motor_indices = self.motor_indices[motor_groups == "neck"]
        self.waist_motor_indices = self.motor_indices[motor_groups == "waist"]

        joint_ref_indices = jnp.arange(len(self.robot.joint_ordering))  # type:ignore
        self.leg_joint_ref_indices = joint_ref_indices[joint_groups == "leg"]  # type:ignore
        self.arm_joint_ref_indices = joint_ref_indices[joint_groups == "arm"]  # type:ignore
        self.neck_joint_ref_indices = joint_ref_indices[joint_groups == "neck"]  # type:ignore
        self.waist_joint_ref_indices = joint_ref_indices[joint_groups == "waist"]  # type:ignore

        # default qpos
        self.default_qpos = jnp.array(self.sys.mj_model.keyframe("home").qpos)  # type:ignore
        # default action
        self.default_motor_pos = jnp.array(  # type:ignore
            list(self.robot.default_motor_angles.values())
        )

        # commands
        # x vel, y vel, yaw vel, heading
        self.num_commands = self.cfg.commands.num_commands
        self.command_ranges = asdict(self.cfg.commands.ranges)
        self.resample_time = self.cfg.commands.resample_time
        self.resample_steps = int(self.resample_time / self.dt)

        # observation
        self.ref_start_idx = 7 + 6
        self.state_ref_size = 7 + 6 + 2 * self.nu + 2
        self.num_obs_history = self.cfg.obs.frame_stack
        self.num_privileged_obs_history = self.cfg.obs.c_frame_stack
        self.obs_size = self.cfg.obs.num_single_obs
        self.privileged_obs_size = self.cfg.obs.num_single_privileged_obs
        self.obs_scales = self.cfg.obs.scales

        self.q_start_idx = 0 if self.fixed_base else 7
        self.qd_start_idx = 0 if self.fixed_base else 6

        # actions
        self.action_scale = self.cfg.action.action_scale
        self.cycle_time = self.cfg.action.cycle_time

        # noise
        self.obs_noise_scale = self.cfg.noise.obs_noise_scale * jnp.concatenate(  # type:ignore
            [
                jnp.zeros(5),  # type:ignore
                jnp.ones_like(self.motor_indices) * self.cfg.noise.dof_pos,  # type:ignore
                jnp.ones_like(self.motor_indices) * self.cfg.noise.dof_vel,  # type:ignore
                jnp.zeros_like(self.motor_indices),  # type:ignore
                # jnp.ones(3) * self.cfg.noise.lin_vel,  # type:ignore
                jnp.ones(3) * self.cfg.noise.ang_vel,  # type:ignore
                jnp.ones(3) * self.cfg.noise.euler,  # type:ignore
            ]
        )
        self.reset_noise_pos = self.cfg.noise.reset_noise_pos

        self.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.push_vel = self.cfg.domain_rand.push_vel
        # # forces
        # self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(
        #     (self.num_envs, 1)
        # )
        # self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # self.rand_push = torch.zeros(
        #     (self.num_envs, 6), dtype=torch.float32, device=self.device
        # )

        with jax.disable_jit():
            if self.ref_motion_name == "walk_simple":
                from toddlerbot.ref_motion.walk_simple_ref import WalkSimpleReference

                self.motion_ref = WalkSimpleReference(
                    self.robot,
                    default_joint_pos=jnp.array(  # type:ignore
                        list(self.robot.default_joint_angles.values())
                    ),
                )
            elif self.ref_motion_name == "walk_zmp":
                from toddlerbot.ref_motion.walk_zmp_ref import WalkZMPReference

                self.motion_ref = WalkZMPReference(
                    self.robot,
                    list(self.command_ranges.values()),
                    self.cycle_time,
                    default_joint_pos=jnp.array(  # type:ignore
                        list(self.robot.default_joint_angles.values())
                    ),
                    control_dt=float(self.dt),
                )
            elif self.ref_motion_name == "squat":
                from toddlerbot.ref_motion.squat_ref import SquatReference

                self.motion_ref = SquatReference(
                    self.robot,
                    default_joint_pos=jnp.array(  # type:ignore
                        list(self.robot.default_joint_angles.values())
                    ),
                )
            else:
                raise ValueError(f"Unknown env {self.name}")

    def _init_reward(self):
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
        self.reward_scales = jnp.zeros(len(reward_scale_dict))  # type:ignore
        for i, (name, scale) in enumerate(reward_scale_dict.items()):
            self.reward_functions.append(getattr(self, "_reward_" + name))
            self.reward_scales = self.reward_scales.at[i].set(scale)  # type:ignore

        self.healthy_z_range = self.cfg.rewards.healthy_z_range
        self.tracking_sigma = self.cfg.rewards.tracking_sigma
        self.min_feet_distance = self.cfg.rewards.min_feet_distance
        self.max_feet_distance = self.cfg.rewards.max_feet_distance
        self.target_feet_z_delta = self.cfg.rewards.target_feet_z_delta

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)  # type:ignore

        qpos = self.default_qpos

        if self.add_noise:
            noise_pos = jax.random.uniform(  # type:ignore
                rng1,
                (self.nq - self.q_start_idx,),
                minval=-self.reset_noise_pos,
                maxval=self.reset_noise_pos,
            )
            qpos = qpos.at[self.q_start_idx :].add(noise_pos)

        qvel = jnp.zeros(self.nv)  # type:ignore
        pipeline_state = self.pipeline_init(qpos, qvel)

        state_info = {
            "rng": rng,
            "command": self._sample_command(pipeline_state, rng2),
            "path_pos": jnp.zeros(3),  # type:ignore
            "path_quat": jnp.array([1.0, 0.0, 0.0, 0.0]),  # type:ignore
            "phase": 0.0,
            "phase_signal": jnp.array([0.0, 1.0]),  # type:ignore
            "state_ref": jnp.zeros(self.state_ref_size),  # type:ignore
            "contact_forces": jnp.zeros(  # type:ignore
                (self.num_colliders, self.num_colliders, 3)
            ),
            "stance_mask": jnp.zeros(2),  # type:ignore
            "last_stance_mask": jnp.zeros(2),  # type:ignore
            "feet_air_time": jnp.zeros(2),  # type:ignore
            "init_feet_height": pipeline_state.x.pos[self.feet_link_ids, 2],
            "last_last_act": jnp.zeros(self.nu),  # type:ignore
            "last_act": jnp.zeros(self.nu),  # type:ignore
            "last_torso_euler": jnp.zeros(3),  # type:ignore
            "rewards": {k: 0.0 for k in self.reward_names},
            "push": jnp.zeros(2),  # type:ignore
            "done": False,
            "step": 0,
        }

        obs_history = jnp.zeros(self.num_obs_history * self.obs_size)  # type:ignore
        privileged_obs_history = jnp.zeros(  # type:ignore
            self.num_privileged_obs_history * self.privileged_obs_size
        )
        obs, privileged_obs = self._get_obs(
            pipeline_state,
            state_info,
            obs_history,
            privileged_obs_history,
        )
        reward, done, zero = jnp.zeros(3)  # type:ignore

        metrics: Dict[str, Any] = {}
        for k in self.reward_names:
            metrics[k] = zero

        return State(
            pipeline_state, obs, privileged_obs, reward, done, metrics, state_info
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

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        rng, cmd_rng, push_rng = jax.random.split(state.info["rng"], 3)  # type:ignore

        push_theta = jax.random.uniform(push_rng, maxval=2 * jnp.pi)  # type:ignore
        push = jnp.array([jnp.cos(push_theta), jnp.sin(push_theta)])  # type:ignore
        push *= jnp.mod(state.info["step"], self.push_interval) == 0  # type:ignore
        qvel = state.pipeline_state.qd  # type:ignore
        qvel = qvel.at[:2].set(push * self.push_vel + qvel[:2])  # type:ignore
        state = state.tree_replace({"pipeline_state.qd": qvel})  # type:ignore

        action = action.at[self.arm_motor_indices].set(0)  # type:ignore
        action = action.at[self.neck_motor_indices].set(0)  # type:ignore
        # action = action.at[self.waist_motor_indices[-1]].set(0)  # type:ignore
        motor_target = self.default_motor_pos + action * self.action_scale

        # jax.debug.breakpoint()

        pipeline_state = self.pipeline_step(state.pipeline_state, motor_target)

        # jax.debug.print(
        #     "qfrc: {}",
        #     pipeline_state.qfrc_actuator[self.qd_start_idx + self.leg_motor_indices],
        # )
        # jax.debug.print("stance_mask: {}", state.info["stance_mask"])
        # jax.debug.print("feet_air_time: {}", state.info["feet_air_time"])

        phase = state.info["step"] * self.dt / self.cycle_time
        phase_signal = jnp.array(  # type:ignore
            [jnp.sin(2 * jnp.pi * phase), jnp.cos(2 * jnp.pi * phase)]  # type:ignore
        )
        path_pos, path_quat = self._integrate_path_frame(state.info)
        state_ref = self.motion_ref.get_state_ref(
            path_pos, path_quat, phase, state.info["command"]
        )
        contact_forces, stance_mask = self._get_contact_forces(
            pipeline_state  # type:ignore
        )

        torso_height = pipeline_state.x.pos[0, 2]
        done = jnp.logical_or(  # type:ignore
            torso_height < self.healthy_z_range[0],  # type:ignore
            torso_height > self.healthy_z_range[1],  # type:ignore
        )

        state.info["path_pos"] = path_pos
        state.info["path_quat"] = path_quat
        state.info["phase"] = phase
        state.info["phase_signal"] = phase_signal
        state.info["state_ref"] = state_ref
        state.info["contact_forces"] = contact_forces
        state.info["stance_mask"] = stance_mask
        state.info["done"] = done

        obs, privileged_obs = self._get_obs(
            pipeline_state, state.info, state.obs, state.privileged_obs
        )

        torso_euler = math.quat_to_euler(pipeline_state.x.rot[0])
        torso_euler_delta = torso_euler - state.info["last_torso_euler"]
        torso_euler_delta = (torso_euler_delta + jnp.pi) % (2 * jnp.pi) - jnp.pi
        torso_euler = state.info["last_torso_euler"] + torso_euler_delta

        reward_dict = self._compute_reward(pipeline_state, state.info, action)  # type:ignore
        reward = sum(reward_dict.values()) * self.dt  # type:ignore
        # reward = jnp.clip(reward, 0.0)  # type:ignore

        state.info["push"] = push
        state.info["last_last_act"] = state.info["last_act"].copy()
        state.info["last_act"] = action.copy()
        state.info["last_torso_euler"] = torso_euler
        state.info["last_stance_mask"] = stance_mask.copy()
        state.info["feet_air_time"] += self.dt
        state.info["feet_air_time"] *= 1.0 - stance_mask
        state.info["rewards"] = reward_dict
        state.info["rng"] = rng
        state.info["step"] += 1

        # sample new command if more than 500 timesteps achieved
        state.info["command"] = jnp.where(  # type:ignore
            state.info["step"] > self.resample_steps,
            self._sample_command(pipeline_state, cmd_rng),
            state.info["command"],
        )
        # reset the step counter when done
        state.info["step"] = jnp.where(  # type:ignore
            done | (state.info["step"] > self.resample_steps), 0, state.info["step"]
        )
        state.metrics.update(reward_dict)

        return state.replace(  # type:ignore
            pipeline_state=pipeline_state,
            obs=obs,
            privileged_obs=privileged_obs,
            reward=reward,
            done=done.astype(jnp.float32),  # type:ignore
        )

    def _integrate_path_frame(
        self, info: Dict[str, Any]
    ) -> Tuple[jax.Array, jax.Array]:
        pos = info["path_pos"]
        quat = info["path_quat"]
        x_vel = info["command"][0]
        y_vel = info["command"][1]
        yaw_vel = info["command"][2]

        # Update position
        pos += jnp.array([x_vel, y_vel, 0.0]) * self.dt  # type:ignore

        # Update quaternion for yaw rotation
        theta = yaw_vel * self.dt / 2.0
        yaw_quat = jnp.array(  # type:ignore
            [jnp.cos(theta), 0.0, 0.0, jnp.sin(theta)],  # type:ignore
        )
        yaw_quat /= jnp.linalg.norm(yaw_quat)  # type:ignore
        quat = math.quat_mul(quat, yaw_quat)  # type:ignore
        quat /= jnp.linalg.norm(quat)  #   type:ignore

        return pos, quat  # type:ignore

    def _get_contact_forces(self, data: mjx.Data):
        # Extract geom1 and geom2 directly
        geom1 = data.contact.geom1
        geom2 = data.contact.geom2

        def get_body_index(geom_id: jax.Array) -> jax.Array:
            return jnp.argmax(self.collider_geom_ids == geom_id)  # type:ignore

        # Vectorized computation of body indices for geom1 and geom2
        body_indices_1 = jax.vmap(get_body_index)(geom1)
        body_indices_2 = jax.vmap(get_body_index)(geom2)

        contact_forces_global = jnp.zeros(  # type:ignore
            (self.num_colliders, self.num_colliders, 3)
        )
        for i in range(data.ncon):
            contact_force = self.jit_contact_force(self.sys, data, i, True)[:3]  # type:ignore
            # Update the contact forces for both body_indices_1 and body_indices_2
            # Add instead of set to accumulate forces from multiple contacts
            contact_forces_global = contact_forces_global.at[
                body_indices_1[i], body_indices_2[i]
            ].add(contact_force)  # type:ignore
            contact_forces_global = contact_forces_global.at[
                body_indices_2[i], body_indices_1[i]
            ].add(contact_force)  # type:ignore

        stance_mask = (
            contact_forces_global[0, self.feet_indices, 2]
            > self.contact_force_threshold
        ).astype(jnp.float32)  # type:ignore

        return contact_forces_global, stance_mask

    def _get_obs(
        self,
        pipeline_state: base.State,
        info: dict[str, Any],
        obs_history: jax.Array,
        privileged_obs_history: Optional[jax.Array] = None,
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

        obs = jnp.concatenate(  # type:ignore
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
        privileged_obs = jnp.concatenate(  # type:ignore
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
            obs += self.obs_noise_scale * jax.random.uniform(  # type:ignore
                info["rng"], obs.shape, minval=-1, maxval=1
            )

        # jax.debug.breakpoint()

        # obs = jnp.clip(obs, -100.0, 100.0)  # type:ignore
        # stack observations through time
        obs = jnp.roll(obs_history, obs.size).at[: obs.size].set(obs)  # type:ignore

        privileged_obs = (
            jnp.roll(privileged_obs_history, privileged_obs.size)  # type:ignore
            .at[: privileged_obs.size]
            .set(privileged_obs)
        )

        return obs, privileged_obs

    def _compute_reward(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        # Create an array of indices to map over
        indices = jnp.arange(len(self.reward_names))  # type:ignore
        # Use jax.lax.map to compute rewards
        reward_arr = jax.lax.map(  # type:ignore
            lambda i: jax.lax.switch(  # type:ignore
                i,  # type:ignore
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
        error = jnp.linalg.norm(torso_pos - torso_pos_ref, axis=-1)  # type:ignore
        reward = jnp.exp(-200.0 * error**2)  # type:ignore
        return reward

    def _reward_torso_quat(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Reward for track torso orientation"""
        torso_quat = pipeline_state.x.rot[0]
        torso_quat_ref = info["state_ref"][3:7]
        # Quaternion dot product (cosine of the half-angle)
        dot_product = jnp.sum(torso_quat * torso_quat_ref, axis=-1)  # type:ignore
        # Ensure the dot product is within the valid range
        dot_product = jnp.clip(dot_product, -1.0, 1.0)  # type:ignore
        # Quaternion angle difference
        angle_diff = 2.0 * jnp.arccos(jnp.abs(dot_product))  # type:ignore
        reward = jnp.exp(-20.0 * (angle_diff**2))  # type:ignore
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
        error = jnp.linalg.norm(lin_vel_xy - lin_vel_xy_ref, axis=-1)  # type:ignore
        reward = jnp.exp(-self.tracking_sigma * error**2)  # type:ignore
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
        reward = jnp.exp(-self.tracking_sigma * error**2)  # type:ignore
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
        error = jnp.linalg.norm(ang_vel_xy - ang_vel_xy_ref, axis=-1)  # type:ignore
        reward = jnp.exp(-self.tracking_sigma / 4 * error**2)  # type:ignore
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
        reward = jnp.exp(-self.tracking_sigma / 4 * error**2)  # type:ignore
        return reward

    def _reward_leg_joint_pos(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking leg joint positions"""
        joint_pos = pipeline_state.q[self.q_start_idx + self.leg_joint_indices]
        joint_pos_ref = info["state_ref"][
            self.ref_start_idx + self.leg_joint_ref_indices
        ]
        error = joint_pos - joint_pos_ref
        reward = -jnp.mean(error**2)  # type:ignore
        return reward

    def _reward_leg_joint_vel(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking leg joint velocities"""
        joint_vel = pipeline_state.qd[self.qd_start_idx + self.leg_joint_indices]
        joint_vel_ref = info["state_ref"][
            self.ref_start_idx + self.nu + self.leg_joint_ref_indices
        ]
        error = joint_vel - joint_vel_ref
        reward = -jnp.mean(error**2)  # type:ignore
        return reward

    def _reward_arm_joint_pos(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking arm joint positions"""
        joint_pos = pipeline_state.q[self.q_start_idx + self.arm_joint_indices]
        joint_pos_ref = info["state_ref"][
            self.ref_start_idx + self.arm_joint_ref_indices
        ]
        error = joint_pos - joint_pos_ref
        reward = -jnp.mean(error**2)  # type:ignore
        return reward

    def _reward_arm_joint_vel(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking arm joint velocities"""
        joint_vel = pipeline_state.qd[self.qd_start_idx + self.arm_joint_indices]
        joint_vel_ref = info["state_ref"][
            self.ref_start_idx + self.nu + self.arm_joint_ref_indices
        ]
        error = joint_vel - joint_vel_ref
        reward = -jnp.mean(error**2)  # type:ignore
        return reward

    def _reward_neck_joint_pos(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking neck joint positions"""
        joint_pos = pipeline_state.q[self.q_start_idx + self.neck_joint_indices]
        joint_pos_ref = info["state_ref"][
            self.ref_start_idx + self.neck_joint_ref_indices
        ]
        error = joint_pos - joint_pos_ref
        reward = -jnp.mean(error**2)  # type:ignore
        return reward

    def _reward_neck_joint_vel(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking neck joint velocities"""
        joint_vel = pipeline_state.qd[self.qd_start_idx + self.neck_joint_indices]
        joint_vel_ref = info["state_ref"][
            self.ref_start_idx + self.nu + self.neck_joint_ref_indices
        ]
        error = joint_vel - joint_vel_ref
        reward = -jnp.mean(error**2)  # type:ignore
        return reward

    def _reward_waist_joint_pos(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking waist joint positions"""
        joint_pos = pipeline_state.q[self.q_start_idx + self.waist_joint_indices]
        joint_pos_ref = info["state_ref"][
            self.ref_start_idx + self.waist_joint_ref_indices
        ]
        error = joint_pos - joint_pos_ref
        reward = -jnp.mean(error**2)  # type:ignore
        return reward

    def _reward_waist_joint_vel(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking waist joint velocities"""
        joint_vel = pipeline_state.qd[self.qd_start_idx + self.waist_joint_indices]
        joint_vel_ref = info["state_ref"][
            self.ref_start_idx + self.nu + self.waist_joint_ref_indices
        ]
        error = joint_vel - joint_vel_ref
        reward = -jnp.mean(error**2)  # type:ignore
        return reward

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

    def _reward_collision(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        collision_forces = jnp.linalg.norm(  # type:ignore
            info["contact_forces"][1:, 1:],  # exclude the floor
            axis=-1,
        )
        collision_contact = collision_forces > 0.1  # type:ignore
        reward = -jnp.sum(collision_contact.astype(jnp.float32))  # type:ignore
        return reward

    def _reward_joint_torque(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for minimizing joint torques"""
        torque = pipeline_state.qfrc_actuator[self.motor_indices]  # type:ignore
        error = jnp.square(torque)  # type:ignore
        reward = -jnp.mean(error)  # type:ignore
        return reward  # type:ignore

    def _reward_joint_acc(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for minimizing joint accelerations"""
        joint_acc = pipeline_state.qacc[self.qd_start_idx + self.joint_indices]  # type:ignore
        error = jnp.square(joint_acc)  # type:ignore
        reward = -jnp.mean(error)  # type:ignore
        return reward  # type:ignore

    def _reward_leg_action_rate(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking leg action rates"""
        leg_action = action[self.leg_motor_indices]
        last_leg_action = info["last_act"][self.leg_motor_indices]
        error = jnp.square(leg_action - last_leg_action)  # type:ignore
        reward = -jnp.mean(error)  # type:ignore
        return reward  # type:ignore

    def _reward_leg_action_acc(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking leg action accelerations"""
        leg_action = action[self.leg_motor_indices]
        last_leg_action = info["last_act"][self.leg_motor_indices]
        last_last_leg_action = info["last_last_act"][self.leg_motor_indices]
        error = jnp.square(  # type:ignore
            leg_action - 2 * last_leg_action + last_last_leg_action
        )
        reward = -jnp.mean(error)  # type:ignore
        return reward  # type:ignore

    def _reward_arm_action_rate(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking arm action rates"""
        arm_action = action[self.arm_motor_indices]
        last_arm_action = info["last_act"][self.arm_motor_indices]
        error = jnp.square(arm_action - last_arm_action)  # type:ignore
        reward = -jnp.mean(error)  # type:ignore
        return reward  # type:ignore

    def _reward_arm_action_acc(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking arm action accelerations"""
        arm_action = action[self.arm_motor_indices]
        last_arm_action = info["last_act"][self.arm_motor_indices]
        last_last_arm_action = info["last_last_act"][self.arm_motor_indices]
        error = jnp.square(  # type:ignore
            arm_action - 2 * last_arm_action + last_last_arm_action
        )
        reward = -jnp.mean(error)  # type:ignore
        return reward  # type:ignore

    def _reward_neck_action_rate(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking neck action rates"""
        neck_action = action[self.neck_motor_indices]
        last_neck_action = info["last_act"][self.neck_motor_indices]
        error = jnp.square(neck_action - last_neck_action)  # type:ignore
        reward = -jnp.mean(error)  # type:ignore
        return reward  # type:ignore

    def _reward_neck_action_acc(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking neck action accelerations"""
        neck_action = action[self.neck_motor_indices]
        last_neck_action = info["last_act"][self.neck_motor_indices]
        last_last_neck_action = info["last_last_act"][self.neck_motor_indices]
        error = jnp.square(  # type:ignore
            neck_action - 2 * last_neck_action + last_last_neck_action
        )
        reward = -jnp.mean(error)  # type:ignore
        return reward  # type:ignore

    def _reward_waist_action_rate(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking waist action rates"""
        waist_action = action[self.waist_motor_indices]
        last_waist_action = info["last_act"][self.waist_motor_indices]
        error = jnp.square(waist_action - last_waist_action)  # type:ignore
        reward = -jnp.mean(error)  # type:ignore
        return reward  # type:ignore

    def _reward_waist_action_acc(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        """Reward for tracking waist action accelerations"""
        waist_action = action[self.waist_motor_indices]
        last_waist_action = info["last_act"][self.waist_motor_indices]
        last_last_waist_action = info["last_last_act"][self.waist_motor_indices]
        error = jnp.square(  # type:ignore
            waist_action - 2 * last_waist_action + last_last_waist_action
        )
        reward = -jnp.mean(error)  # type:ignore
        return reward  # type:ignore

    def _reward_survival(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ) -> jax.Array:
        return -(info["done"] & (info["step"] < self.resample_steps)).astype(
            jnp.float32
        )
