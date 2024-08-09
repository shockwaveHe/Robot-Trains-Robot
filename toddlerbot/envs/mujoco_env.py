from dataclasses import asdict
from typing import Any, Callable, List, Tuple

import jax
import mujoco  # type: ignore
import numpy as np
from brax.envs.base import PipelineEnv, State  # type: ignore
from brax.io import mjcf  # type: ignore
from jax import numpy as jnp
from mujoco import mjx  # type: ignore

from toddlerbot.envs.mujoco_config import MuJoCoConfig
from toddlerbot.motion_reference.motion_ref import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.jax_utils import quat_apply, quat_mult, wrap_to_pi


class MuJoCoEnv(PipelineEnv):
    def __init__(
        self,
        robot: Robot,
        motion_ref: MotionReference,
        cfg: MuJoCoConfig,
        fixed_base: bool = False,
        # forward_reward_weight=1.25,
        # ctrl_cost_weight=0.1,
        # healthy_reward=5.0,
        # terminate_when_unhealthy=True,
        # healthy_z_range=(1.0, 2.0),
        # reset_noise_scale=1e-2,
        # exclude_current_positions_from_observation=True,
        **kwargs: Any,
    ):
        self.robot = robot
        self.fixed_base = fixed_base
        self.motion_ref = motion_ref
        self.cfg = cfg

        if fixed_base:
            xml_path = find_robot_file_path(robot.name, suffix="_fixed_scene.xml")
        else:
            xml_path = find_robot_file_path(robot.name, suffix="_scene.xml")

        mj_model = mujoco.MjModel.from_xml_path(xml_path)  # type: ignore
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG  # type: ignore
        mj_model.opt.iterations = 6  # type: ignore
        mj_model.opt.ls_iterations = 6  # type: ignore

        sys = mjcf.load_model(mj_model)  # type: ignore

        kwargs["n_frames"] = cfg.control.decimation
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)  # type: ignore

        self._init_env()
        self._init_joint_indices()
        self._init_reward()

        # self._forward_reward_weight = forward_reward_weight
        # self._ctrl_cost_weight = ctrl_cost_weight
        # self._healthy_reward = healthy_reward
        # self._terminate_when_unhealthy = terminate_when_unhealthy
        # self._healthy_z_range = healthy_z_range
        # self._reset_noise_scale = reset_noise_scale
        # self._exclude_current_positions_from_observation = (
        #     exclude_current_positions_from_observation
        # )

    def _init_env(self):
        # self.num_envs = self.cfg.env.num_envs
        # self.num_obs = self.cfg.env.num_observations
        # self.num_privileged_obs = self.cfg.env.num_privileged_obs
        # self.num_actions = len(self.robot.motor_ordering)

        # self.obs_scales = self.cfg.normalization.scales
        # self.command_ranges = asdict(self.cfg.commands.ranges)
        # # if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
        # #     self.cfg.terrain.curriculum = False

        # self.max_episode_length_s = self.cfg.env.episode_length_s
        # self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        # self.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

        # # buffers
        # self.obs_buf = torch.zeros(
        #     self.num_envs, self.num_obs, device=self.device, dtype=torch.float
        # )
        # self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # # new reward buffers for exp rewrads
        # self.neg_reward_buf = torch.zeros(
        #     self.num_envs, device=self.device, dtype=torch.float
        # )
        # self.pos_reward_buf = torch.zeros(
        #     self.num_envs, device=self.device, dtype=torch.float
        # )
        self.only_positive_rewards = self.cfg.rewards.only_positive_rewards

        self.cycle_time = self.cfg.rewards.cycle_time

        # self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_step = 0

        self.path_frame = jnp.zeros(7)  # type:ignore
        self.path_frame = self.path_frame.at[3].set(1.0)  # type:ignore

        self.phase = jnp.zeros(1)  # type:ignore
        self.phase_signal = jnp.zeros(2)  # type:ignore
        self.state_ref_buf = jnp.zeros(7 + 6 + 2 * self.action_size + 2)  # type:ignore

        self.last_actions = jnp.zeros(self.action_size)  # type:ignore
        self.last_last_actions = jnp.zeros(self.action_size)  # type:ignore

        # self.time_out_buf = torch.zeros(
        #     self.num_envs, device=self.device, dtype=torch.bool
        # )
        # if self.num_privileged_obs:
        #     self.privileged_obs_buf = torch.zeros(
        #         self.num_envs,
        #         self.num_privileged_obs,
        #         device=self.device,
        #         dtype=torch.float,
        #     )
        # else:
        #     self.privileged_obs_buf = None

        # # history
        # self.obs_history = deque(maxlen=self.cfg.env.frame_stack)  # type: ignore
        # self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)  # type: ignore
        # for _ in range(self.cfg.env.frame_stack):
        #     self.obs_history.append(  # type: ignore
        #         torch.zeros(
        #             self.num_envs,
        #             self.cfg.env.num_single_obs,
        #             dtype=torch.float,
        #             device=self.device,
        #         )
        #     )
        # for _ in range(self.cfg.env.c_frame_stack):
        #     self.critic_history.append(  # type: ignore
        #         torch.zeros(
        #             self.num_envs,
        #             self.cfg.env.num_single_privileged_obs,
        #             dtype=torch.float,
        #             device=self.device,
        #         )
        #     )

        # # actions
        # self.actions = torch.zeros(
        #     self.num_envs,
        #     self.num_actions,
        #     dtype=torch.float,
        #     device=self.device,
        #     requires_grad=False,
        # )
        # self.last_actions = torch.zeros(
        #     self.num_envs,
        #     self.num_actions,
        #     dtype=torch.float,
        #     device=self.device,
        #     requires_grad=False,
        # )
        # self.last_last_actions = torch.zeros(
        #     self.num_envs,
        #     self.num_actions,
        #     dtype=torch.float,
        #     device=self.device,
        #     requires_grad=False,
        # )

        # commands
        self.heading_command = self.cfg.commands.heading_command
        # x vel, y vel, yaw vel, heading
        self.command_buf = jnp.zeros(self.cfg.commands.num_commands)  # type:ignore
        # self.commands_scale = torch.tensor(
        #     [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
        #     device=self.device,
        #     requires_grad=False,
        # )

        self.forward_vec = jnp.array([1.0, 0.0, 0.0])  # type:ignore

        self.contact_forces = jnp.zeros((len(self.robot.collider_names), 3))  # type:ignore
        self.contact_mask = jnp.zeros(2)  # type:ignore

        self.contact_force_threshold = self.cfg.rewards.contact_force_threshold
        # contact
        feet_indices: List[int] = []
        termination_contact_indices: List[int] = []
        for i, name in enumerate(self.robot.collider_names):
            if self.robot.foot_name in name:
                feet_indices.append(i)
            else:
                termination_contact_indices.append(i)

        self.feet_indices = jnp.array(feet_indices, dtype=jnp.int32)  # type:ignore
        self.termination_contact_indices = jnp.array(  # type:ignore
            termination_contact_indices, dtype=jnp.int32
        )
        # Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        # Otherwise create a grid.

        # if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        #     self.custom_origins = True
        #     self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        #     # put robots at the origins defined by the terrain
        #     max_init_level = self.cfg.terrain.max_init_terrain_level
        #     if not self.cfg.terrain.curriculum:
        #         max_init_level = self.cfg.terrain.num_rows - 1
        #     self.terrain_levels = torch.randint(
        #         0, max_init_level + 1, (self.num_envs,), device=self.device
        #     )
        #     self.terrain_types = torch.div(
        #         torch.arange(self.num_envs, device=self.device),
        #         (self.num_envs / self.cfg.terrain.num_cols),
        #         rounding_mode="floor",
        #     ).to(torch.long)
        #     self.max_terrain_level = self.cfg.terrain.num_rows
        #     self.terrain_origins = (
        #         torch.from_numpy(self.terrain.env_origins)  # type: ignore
        #         .to(self.device)
        #         .to(torch.float32)
        #     )
        #     self.env_origins[:] = self.terrain_origins[
        #         self.terrain_levels, self.terrain_types
        #     ]
        # else:
        # self.custom_origins = False
        # self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        # # create a grid of robots
        # num_cols = np.floor(np.sqrt(self.num_envs))
        # num_rows = np.ceil(self.num_envs / num_cols)
        # xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        # spacing = self.cfg.env.env_spacing
        # self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
        # self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
        # self.env_origins[:, 2] = 0.0

    # def _init_dof(self):
    #     self.dof_names = self.robot.joint_ordering
    #     self.num_dof = len(self.dof_names)

    #     # dof state
    #     self.dof_state = (
    #         torch.from_numpy(self.sim.get_dof_state())  # type: ignore
    #         .to(self.device)
    #         .tile((self.num_envs, 1, 1))
    #     )
    #     self.dof_pos = self.dof_state[..., 0]
    #     self.dof_vel = self.dof_state[..., 1]
    #     self.last_dof_vel = torch.zeros_like(self.dof_vel)
    #     # default dof pos
    #     self.default_dof_pos = self.dof_pos[:1].clone()

    # def _init_root(self):
    #     # root state
    #     self.base_init_state = torch.from_numpy(self.sim.get_root_state()).to(  # type: ignore
    #         self.device
    #     )
    #     self.root_states = self.base_init_state.tile((self.num_envs, 1))
    #     self.base_quat = self.root_states[..., 3:7]
    #     self.base_euler_xyz = quat_to_euler_tensor(self.base_quat)
    #     self.base_lin_vel = quat_rotate_inverse(
    #         self.base_quat, self.root_states[:, 7:10]
    #     )
    #     self.base_ang_vel = quat_rotate_inverse(
    #         self.base_quat, self.root_states[:, 10:13]
    #     )
    #     self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

    # def _init_body(self):
    #     body_names = self.robot.collider_names
    #     self.num_bodies = len(body_names)
    #     self.contact_force_threshold = self.cfg.rewards.contact_force_threshold

    #     # body state
    #     self.body_state = (
    #         torch.from_numpy(self.sim.get_body_state())  # type: ignore
    #         .to(self.device)
    #         .tile((self.num_envs, 1, 1))
    #     )
    #     self.last_body_state = torch.zeros_like(self.body_state)
    #     self.body_mass = torch.zeros(
    #         self.num_envs, 1, dtype=torch.float32, device=self.device
    #     )

    #     self.feet_air_time = torch.zeros(
    #         self.num_envs,
    #         len(self.feet_names),
    #         dtype=torch.float,
    #         device=self.device,
    #         requires_grad=False,
    #     )
    #     self.last_contacts = torch.zeros(
    #         self.num_envs,
    #         len(self.feet_names),
    #         dtype=torch.bool,
    #         device=self.device,
    #         requires_grad=False,
    #     )
    #     # self.contact_forces = (
    #     #     torch.from_numpy(self.sim.get_contact_forces())  # type: ignore
    #     #     .to(self.device)
    #     #     .tile((self.num_envs, 1, 1))
    #     # )
    #     self.env_frictions = torch.zeros(
    #         self.num_envs, 1, dtype=torch.float32, device=self.device
    #     )

    #     # self.measured_heights = 0
    #     # if self.cfg.terrain.measure_heights:
    #     #     self.height_points = self._init_height_points()

    #     # forces
    #     self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(
    #         (self.num_envs, 1)
    #     )
    #     self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
    #     self.forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(
    #         (self.num_envs, 1)
    #     )
    #     self.rand_push = torch.zeros(
    #         (self.num_envs, 6), dtype=torch.float32, device=self.device
    #     )

    def _init_joint_indices(self):
        self.joint_indices = jnp.zeros(len(self.robot.joint_ordering), dtype=jnp.int32)  # type:ignore
        leg_joint_indices: List[int] = []
        arm_joint_indices: List[int] = []
        neck_joint_indices: List[int] = []
        waist_joint_indices: List[int] = []
        for i, name in enumerate(self.robot.joint_ordering):
            joint_idx: int = int(self.sys.mj_model.joint(name).id)  # type:ignore
            self.joint_indices = self.joint_indices.at[i].set(joint_idx)  # type:ignore
            joint_group = self.robot.joint_group[name]
            if joint_group == "leg":
                leg_joint_indices.append(joint_idx)
            elif joint_group == "arm":
                arm_joint_indices.append(joint_idx)
            elif joint_group == "neck":
                neck_joint_indices.append(joint_idx)
            elif joint_group == "waist":
                waist_joint_indices.append(joint_idx)

        self.leg_joint_indices = jnp.array(leg_joint_indices, dtype=jnp.int32)  # type:ignore
        self.arm_joint_indices = jnp.array(arm_joint_indices, dtype=jnp.int32)  # type:ignore
        self.neck_joint_indices = jnp.array(neck_joint_indices, dtype=jnp.int32)  # type:ignore
        self.waist_joint_indices = jnp.array(waist_joint_indices, dtype=jnp.int32)  # type:ignore

    def _init_reward(self):
        """Prepares a list of reward functions, which will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """

        self.reward_scales = asdict(self.cfg.rewards.scales)
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        # prepare list of functions
        self.reward_functions: List[Callable[..., jax.Array]] = []
        self.reward_names: List[str] = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            reward_function_name = "_reward_" + name
            self.reward_functions.append(getattr(self, reward_function_name))

        # reward episode sums
        self.reward_values = {name: jnp.zeros(1) for name in self.reward_names}  # type:ignore

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)  # type:ignore

        # low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0
        # TODO: Bring them back
        # jax.random.uniform(
        #     rng1, (self.sys.nq,), minval=low, maxval=hi
        # )
        qvel = jnp.zeros(self.sys.nv)  # type:ignore
        # jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)

        obs, privileged_obs = self._get_obs(data, jnp.zeros(self.sys.nu))  # type:ignore
        reward, done, zero = jnp.zeros(3)  # type:ignore
        metrics = dict(zip(self.reward_names, [zero] * len(self.reward_names)))
        metrics["x_position"] = zero
        metrics["y_position"] = zero
        metrics["distance_from_origin"] = zero
        metrics["x_velocity"] = zero
        metrics["y_velocity"] = zero
        return State(data, obs, privileged_obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""

        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        self._post_physics_step(data, action)  # type:ignore

        com_before = data0.subtree_com[1]  # type:ignore
        com_after = data.subtree_com[1]  # type:ignore
        velocity = (com_after - com_before) / self.dt  # type:ignore
        # forward_reward = self._forward_reward_weight * velocity[0]

        # min_z, max_z = self._healthy_z_range
        # is_healthy = jnp.where(data.q[2] < min_z, 0.0, 1.0)
        # is_healthy = jnp.where(data.q[2] > max_z, 0.0, is_healthy)
        # if self._terminate_when_unhealthy:
        #     healthy_reward = self._healthy_reward
        # else:
        #     healthy_reward = self._healthy_reward * is_healthy

        # ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))
        # reward = forward_reward + healthy_reward - ctrl_cost

        obs, privileged_obs = self._get_obs(data, action)  # type:ignore

        reward = self._compute_reward(data)  # type:ignore

        done = self._check_termination()
        # done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        state.metrics.update(**self.reward_values)
        state.metrics["x_position"] = com_after[0]
        state.metrics["y_position"] = com_after[1]
        state.metrics["distance_from_origin"] = jnp.linalg.norm(com_after)  # type:ignore
        state.metrics["x_velocity"] = velocity[0]
        state.metrics["y_velocity"] = velocity[1]

        return state.replace(  # type:ignore
            pipeline_state=data,
            obs=obs,
            privileged_obs=privileged_obs,
            reward=reward,
            done=done,
        )

    def _post_physics_step(self, data: mjx.Data, action: jax.Array):
        self._integrate_path_frame(data)
        self._get_phase()
        self._get_state_ref()
        self._get_contact_forces(data)

        self.last_actions = action.copy()
        self.last_last_actions = self.last_actions.copy()

    def _integrate_path_frame(self, data: mjx.Data):
        pos, quat = self.path_frame[:3], self.path_frame[3:]
        x_vel = self.command_buf[0]
        y_vel = self.command_buf[1]

        if self.heading_command:
            forward = quat_apply(data.q[3:7], self.forward_vec)  # type:ignore
            heading = jnp.atan2(forward[1], forward[0])  # type:ignore
            self.command_buf[2] = jnp.clip(  # type:ignore
                0.5 * wrap_to_pi(self.commands[3] - heading),  # type:ignore
                -1.0,
                1.0,
            )

        yaw_vel = self.command_buf[:, 2]

        # Update position
        pos += jnp.array([x_vel, y_vel, 0.0]) * self.dt  # type:ignore

        # Update quaternion for yaw rotation
        theta = yaw_vel * self.dt / 2.0
        yaw_quat = jnp.array(  # type:ignore
            [jnp.cos(theta), 0.0, 0.0, jnp.sin(theta)],  # type:ignore
        )
        yaw_quat /= jnp.linalg.norm(yaw_quat)  # type:ignore
        quat = quat_mult(quat, yaw_quat)  # type:ignore
        quat /= jnp.linalg.norm(quat)  #   type:ignore

        self.path_frame = jnp.concatenate([pos, quat])  # type:ignore

    def _get_phase(self):
        self.phase = self.episode_step * self.dt / self.cycle_time
        self.phase_signal = jnp.array(  # type:ignore
            [jnp.sin(2 * np.pi * self.phase), jnp.cos(2 * np.pi * self.phase)]  # type:ignore
        )  # type:ignore

    def _get_state_ref(self):
        self.state_ref_buf = self.motion_ref.get_state_ref(
            self.path_frame, self.phase, self.command_buf
        )

    def _get_contact_forces(self, data: mjx.Data):
        # Access contact information
        for i in range(data.ncon):  # type: ignore
            contact = data.contact[i]  # type: ignore
            # Check if the contact involves the ground plane
            geom1 = self.sys.mj_model.geom(contact.geom[0])  # type: ignore
            geom2 = self.sys.mj_model.geom(contact.geom[1])  # type: ignore

            body_name = ""
            if "floor" in geom1.name:  # type: ignore
                body_name = str(self.sys.mj_model.body(geom2.bodyid).name)  # type: ignore
            elif "floor" in geom2.name:  # type: ignore
                body_name = str(self.sys.mj_model.body(geom1.bodyid).name)  # type: ignore
            else:
                continue

            body_idx = self.robot.collider_names.index(body_name)
            # Extract the contact forces
            c_array = np.zeros(6)  # To hold contact forces

            mujoco.mj_contactForce(self.sys.mj_model, data, i, c_array)  # type: ignore
            contact_force_local = c_array[:3].astype(np.float32)
            contact_force_global = contact.frame.reshape(-1, 3).T @ contact_force_local  # type: ignore

            self.contact_forces = self.contact_forces.at[body_idx].set(  # type:ignore
                contact_force_global
            )

        self.contact_mask = (
            self.contact_forces[:, self.feet_indices, 2] > self.contact_force_threshold
        )

    def _get_obs(
        self, data: mjx.Data, action: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """Observes humanoid body position, velocities, and angles."""
        joint_pos = data.qpos[7 + self.joint_indices]
        joint_vel = data.qvel[7 + self.joint_indices]
        # position = data.qpos
        # if self._exclude_current_positions_from_observation:
        #     position = position[2:]

        obs = jnp.concatenate(  # type:ignore
            [self.phase_signal, self.command_buf[:3], joint_pos, joint_vel, data.ctrl]
        )
        # jnp.concatenate(
        #     [
        #         position,
        #         data.qvel,
        #         data.cinert[1:].ravel(),
        #         data.cvel[1:].ravel(),
        #         data.qfrc_actuator,
        #     ]
        # )

        joint_pos_diff = -self.state_ref_buf[13 : 13 + self.action_size]

        # TODO: check each field. Multiply scales, Add push, friction
        privileged_obs = jnp.concatenate(  # type:ignore
            [
                self.phase_signal,
                self.command_buf[:3],
                joint_pos,
                joint_vel,
                data.ctrl,
                data.cinert[1:].ravel(),
                data.cvel[1:].ravel(),
                joint_pos_diff,
                self.contact_mask,
                self.state_ref_buf[-2:],
            ]
        )

        # external_contact_forces are excluded
        return obs, privileged_obs

    def _check_termination(self):
        """Check if environments need to be reset"""
        termination_contact = self.contact_forces[self.termination_contact_indices, :]
        termination = jnp.any(  # type: ignore
            jnp.linalg.norm(termination_contact, axis=-1) > 1.0,  # type: ignore
            axis=0,
        )
        return termination

    def _compute_reward(self, data: mjx.Data):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        reward = jnp.zeros(1)  # type:ignore
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i](data) * self.reward_scales[name]
            reward += rew
            self.reward_values[name] = rew

        if self.only_positive_rewards:
            reward = jnp.clip(reward, min=0.0)  # type:ignore

        return reward

    def _reward_torso_pos(self, data: mjx.Data):
        """Reward for track torso position"""
        torso_pos = data.qpos[:2]  # Assuming [:2] extracts xy components
        torso_pos_ref = self.state_ref_buf[:2]
        error = jnp.linalg.norm(torso_pos - torso_pos_ref, axis=-1)  # type:ignore
        reward = jnp.exp(-200.0 * error**2)  # type:ignore
        return reward

    def _reward_torso_quat(self, data: mjx.Data):
        """Reward for track torso orientation"""
        torso_quat = data.qpos[3:7]
        torso_quat_ref = self.state_ref_buf[3:7]
        # Quaternion dot product (cosine of the half-angle)
        dot_product = jnp.sum(torso_quat * torso_quat_ref, axis=-1)  # type:ignore
        # Ensure the dot product is within the valid range
        dot_product = jnp.clip(dot_product, -1.0, 1.0)  # type:ignore
        # Quaternion angle difference
        angle_diff = 2.0 * jnp.arccos(jnp.abs(dot_product))  # type:ignore
        reward = jnp.exp(-20.0 * (angle_diff**2))  # type:ignore
        return reward

    def _reward_lin_vel_xy(self, data: mjx.Data):
        """Reward for track linear velocity in xy"""
        lin_vel_xy = data.qvel[:2]
        lin_vel_xy_ref = self.state_ref_buf[7:9]
        error = jnp.linalg.norm(lin_vel_xy - lin_vel_xy_ref, axis=-1)  # type:ignore
        reward = jnp.exp(-8.0 * error**2)  # type:ignore
        return reward

    def _reward_lin_vel_z(self, data: mjx.Data):
        """Reward for track linear velocity in z"""
        lin_vel_z = data.qvel[2]
        lin_vel_z_ref = self.state_ref_buf[9]
        error = jnp.linalg.norm(lin_vel_z - lin_vel_z_ref, axis=-1)  # type:ignore
        reward = jnp.exp(-8.0 * error**2)  # type:ignore
        return reward

    def _reward_ang_vel_xy(self, data: mjx.Data):
        """Reward for track angular velocity in xy"""
        ang_vel_xy = data.qvel[3:5]
        ang_vel_xy_ref = self.state_ref_buf[10:12]
        error = jnp.linalg.norm(ang_vel_xy - ang_vel_xy_ref, axis=-1)  # type:ignore
        reward = jnp.exp(-2.0 * error**2)  # type:ignore
        return reward

    def _reward_ang_vel_z(self, data: mjx.Data):
        """Reward for track angular velocity in z"""
        ang_vel_z = data.qvel[5]
        ang_vel_z_ref = self.state_ref_buf[12]
        error = jnp.linalg.norm(ang_vel_z - ang_vel_z_ref, axis=-1)  # type:ignore
        reward = jnp.exp(-2.0 * error**2)  # type:ignore
        return reward

    def _reward_leg_joint_pos(self, data: mjx.Data) -> jax.Array:
        """Reward for track leg joint positions"""
        joint_pos = data.qpos[7 + self.leg_joint_indices]
        joint_pos_ref = self.state_ref_buf[13 + self.leg_joint_indices]
        error = jnp.linalg.norm(joint_pos - joint_pos_ref, axis=-1)  # type:ignore
        reward = -(error**2)  # type:ignore
        return reward  # type:ignore

    def _reward_leg_joint_vel(self, data: mjx.Data) -> jax.Array:
        """Reward for track leg joint velocities"""
        joint_vel = data.qvel[6 + self.leg_joint_indices]
        joint_vel_ref = self.state_ref_buf[
            13 + self.action_size + self.leg_joint_indices
        ]
        error = jnp.linalg.norm(joint_vel - joint_vel_ref, axis=-1)  # type:ignore
        reward = -(error**2)  # type:ignore
        return reward  # type:ignore

    def _reward_arm_joint_pos(self, data: mjx.Data) -> jax.Array:
        """Reward for tracking arm joint positions"""
        joint_pos = data.qpos[7 + self.arm_joint_indices]
        joint_pos_ref = self.state_ref_buf[13 + self.arm_joint_indices]
        error = jnp.linalg.norm(joint_pos - joint_pos_ref, axis=-1)  # type:ignore
        reward = -(error**2)  # type:ignore
        return reward  # type:ignore

    def _reward_arm_joint_vel(self, data: mjx.Data) -> jax.Array:
        """Reward for tracking arm joint velocities"""
        joint_vel = data.qvel[6 + self.arm_joint_indices]
        joint_vel_ref = self.state_ref_buf[
            13 + self.action_size + self.arm_joint_indices
        ]
        error = jnp.linalg.norm(joint_vel - joint_vel_ref, axis=-1)  # type:ignore
        reward = -(error**2)  # type:ignore
        return reward  # type:ignore

    def _reward_neck_joint_pos(self, data: mjx.Data) -> jax.Array:
        """Reward for track neck joint positions"""
        joint_pos = data.qpos[7 + self.neck_joint_indices]
        joint_pos_ref = self.state_ref_buf[13 + self.neck_joint_indices]
        error = jnp.linalg.norm(joint_pos - joint_pos_ref, axis=-1)  # type:ignore
        reward = -(error**2)  # type:ignore
        return reward  # type:ignore

    def _reward_neck_joint_vel(self, data: mjx.Data) -> jax.Array:
        """Reward for track neck joint velocities"""
        joint_vel = data.qvel[6 + self.neck_joint_indices]
        joint_vel_ref = self.state_ref_buf[
            13 + self.action_size + self.neck_joint_indices
        ]
        error = jnp.linalg.norm(joint_vel - joint_vel_ref, axis=-1)  # type:ignore
        reward = -(error**2)  # type:ignore
        return reward  # type:ignore

    def _reward_waist_joint_pos(self, data: mjx.Data) -> jax.Array:
        """Reward for tracking waist joint positions"""
        joint_pos = data.qpos[7 + self.waist_joint_indices]
        joint_pos_ref = self.state_ref_buf[13 + self.waist_joint_indices]
        error = jnp.linalg.norm(joint_pos - joint_pos_ref, axis=-1)  # type:ignore
        reward = -(error**2)  # type:ignore
        return reward  # type:ignore

    def _reward_waist_joint_vel(self, data: mjx.Data) -> jax.Array:
        """Reward for tracking waist joint velocities"""
        joint_vel = data.qvel[6 + self.waist_joint_indices]
        joint_vel_ref = self.state_ref_buf[
            13 + self.action_size + self.waist_joint_indices
        ]
        error = jnp.linalg.norm(joint_vel - joint_vel_ref, axis=-1)  # type:ignore
        reward = -(error**2)  # type:ignore
        return reward  # type:ignore

    def _reward_contact(self, data: mjx.Data):
        """Reward for contact"""
        reward = jnp.sum(self.contact_mask == self.state_ref_buf[-2:])  # type:ignore
        return reward

    ##### Regularization rewards #####

    def _reward_joint_torque(self, data: mjx.Data) -> jax.Array:
        """Reward for minimizing joint torques"""
        torque = data.qfrc_actuator[self.joint_indices]
        error = jnp.linalg.norm(torque, axis=-1)  # type:ignore
        reward = -(error**2)  # type:ignore
        return reward  # type:ignore

    def _reward_joint_acc(self, data: mjx.Data) -> jax.Array:
        """Reward for minimizing joint accelerations"""
        joint_acc = data.qacc[6 + self.joint_indices]
        error = jnp.linalg.norm(joint_acc, axis=-1)  # type:ignore
        reward = -(error**2)  # type:ignore
        return reward  # type:ignore

    def _reward_leg_action_rate(self, data: mjx.Data) -> jax.Array:
        """Reward for tracking leg action rates"""
        leg_action = data.ctrl[self.leg_joint_indices]
        last_leg_action = self.last_actions[self.leg_joint_indices]
        error = jnp.linalg.norm(leg_action - last_leg_action, axis=-1)  # type:ignore
        reward = -(error**2)  # type:ignore
        return reward  # type:ignore

    def _reward_leg_action_acc(self, data: mjx.Data) -> jax.Array:
        """Reward for tracking leg action accelerations"""
        leg_action = data.ctrl[self.leg_joint_indices]
        last_leg_action = self.last_actions[self.leg_joint_indices]
        last_last_leg_action = self.last_last_actions[self.leg_joint_indices]
        error = jnp.linalg.norm(  # type:ignore
            leg_action - 2 * last_leg_action + last_last_leg_action, axis=-1
        )
        reward = -(error**2)  # type:ignore
        return reward  # type:ignore

    def _reward_arm_action_rate(self, data: mjx.Data) -> jax.Array:
        """Reward for tracking arm action rates"""
        arm_action = data.ctrl[self.arm_joint_indices]
        last_arm_action = self.last_actions[self.arm_joint_indices]
        error = jnp.linalg.norm(arm_action - last_arm_action, axis=-1)  # type:ignore
        reward = -(error**2)  # type:ignore
        return reward  # type:ignore

    def _reward_arm_action_acc(self, data: mjx.Data) -> jax.Array:
        """Reward for tracking arm action accelerations"""
        arm_action = data.ctrl[self.arm_joint_indices]
        last_arm_action = self.last_actions[self.arm_joint_indices]
        last_last_arm_action = self.last_last_actions[self.arm_joint_indices]
        error = jnp.linalg.norm(  # type:ignore
            arm_action - 2 * last_arm_action + last_last_arm_action, axis=-1
        )
        reward = -(error**2)  # type:ignore
        return reward  # type:ignore

    def _reward_neck_action_rate(self, data: mjx.Data) -> jax.Array:
        """Reward for tracking neck action rates"""
        neck_action = data.ctrl[self.neck_joint_indices]
        last_neck_action = self.last_actions[self.neck_joint_indices]
        error = jnp.linalg.norm(neck_action - last_neck_action, axis=-1)  # type:ignore
        reward = -(error**2)  # type:ignore
        return reward  # type:ignore

    def _reward_neck_action_acc(self, data: mjx.Data) -> jax.Array:
        """Reward for tracking neck action accelerations"""
        neck_action = data.ctrl[self.neck_joint_indices]
        last_neck_action = self.last_actions[self.neck_joint_indices]
        last_last_neck_action = self.last_last_actions[self.neck_joint_indices]
        error = jnp.linalg.norm(  # type:ignore
            neck_action - 2 * last_neck_action + last_last_neck_action, axis=-1
        )
        reward = -(error**2)  # type:ignore
        return reward  # type:ignore

    def _reward_waist_action_rate(self, data: mjx.Data) -> jax.Array:
        """Reward for tracking waist action rates"""
        waist_action = data.ctrl[self.waist_joint_indices]
        last_waist_action = self.last_actions[self.waist_joint_indices]
        error = jnp.linalg.norm(waist_action - last_waist_action, axis=-1)  # type:ignore
        reward = -(error**2)  # type:ignore
        return reward  # type:ignore

    def _reward_waist_action_acc(self, data: mjx.Data) -> jax.Array:
        """Reward for tracking waist action accelerations"""
        waist_action = data.ctrl[self.waist_joint_indices]
        last_waist_action = self.last_actions[self.waist_joint_indices]
        last_last_waist_action = self.last_last_actions[self.waist_joint_indices]
        error = jnp.linalg.norm(  # type:ignore
            waist_action - 2 * last_waist_action + last_last_waist_action, axis=-1
        )
        reward = -(error**2)  # type:ignore
        return reward  # type:ignore

    def _reward_survival(self, data: mjx.Data):
        return 1.0  # Constant survival reward
