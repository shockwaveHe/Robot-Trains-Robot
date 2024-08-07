from dataclasses import asdict
from typing import Callable, List

import jax
from brax.envs.base import PipelineEnv, State  # type: ignore
from brax.io import mjcf  # type: ignore
from jax import jit  # type: ignore
from jax import numpy as jnp
from mujoco import mjx  # type: ignore

from toddlerbot.envs.mujoco_config import MuJoCoConfig
from toddlerbot.motion_reference.motion_ref import MotionReference
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.jax_utils import quat_apply, quat_mult, wrap_to_pi


class MuJoCoEnv(PipelineEnv):
    def __init__(
        self,
        robot: Robot,
        sim: MuJoCoSim,
        motion_ref: MotionReference,
        cfg: MuJoCoConfig,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        self.robot = robot
        self.sim = sim
        self.motion_ref = motion_ref
        self.cfg = cfg

        sim.update_config(asdict(cfg.mj))

        sys = mjcf.load_model(sim.model)  # type: ignore

        kwargs["n_frames"] = cfg.control.decimation
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)  # type: ignore

        self._init_env()

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

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

        self.cycle_time = self.cfg.rewards.cycle_time

        # self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_step = 0

        self.path_frame = jnp.zeros(7)  # type:ignore
        self.path_frame[3] = 1.0

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
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
        self.env_origins[:, 2] = 0.0

    def _init_dof(self):
        self.dof_names = self.robot.joint_ordering
        self.num_dof = len(self.dof_names)

        # dof state
        self.dof_state = (
            torch.from_numpy(self.sim.get_dof_state())  # type: ignore
            .to(self.device)
            .tile((self.num_envs, 1, 1))
        )
        self.dof_pos = self.dof_state[..., 0]
        self.dof_vel = self.dof_state[..., 1]
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        # default dof pos
        self.default_dof_pos = self.dof_pos[:1].clone()

    def _init_root(self):
        # root state
        self.base_init_state = torch.from_numpy(self.sim.get_root_state()).to(  # type: ignore
            self.device
        )
        self.root_states = self.base_init_state.tile((self.num_envs, 1))
        self.base_quat = self.root_states[..., 3:7]
        self.base_euler_xyz = quat_to_euler_tensor(self.base_quat)
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

    def _init_body(self):
        body_names = self.robot.collider_names
        self.num_bodies = len(body_names)
        self.contact_force_threshold = self.cfg.rewards.contact_force_threshold

        # body state
        self.body_state = (
            torch.from_numpy(self.sim.get_body_state())  # type: ignore
            .to(self.device)
            .tile((self.num_envs, 1, 1))
        )
        self.last_body_state = torch.zeros_like(self.body_state)
        self.body_mass = torch.zeros(
            self.num_envs, 1, dtype=torch.float32, device=self.device
        )

        # contact
        self.feet_names: List[str] = []
        feet_indices: List[int] = []
        penalized_contact_names: List[str] = []
        penalized_contact_indices: List[int] = []
        termination_contact_names: List[str] = []
        termination_contact_indices: List[int] = []
        for i, name in enumerate(body_names):
            if self.robot.foot_name in name:
                self.feet_names.append(name)
                feet_indices.append(i)
            else:
                penalized_contact_names.append(name)
                penalized_contact_indices.append(i)
                termination_contact_names.append(name)
                termination_contact_indices.append(i)

        self.feet_indices = torch.tensor(
            feet_indices, dtype=torch.long, device=self.device
        )
        self.penalized_contact_indices = torch.tensor(
            penalized_contact_indices, dtype=torch.long, device=self.device
        )
        self.termination_contact_indices = torch.tensor(
            termination_contact_indices, dtype=torch.long, device=self.device
        )

        self.feet_air_time = torch.zeros(
            self.num_envs,
            len(self.feet_names),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_names),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.contact_forces = (
            torch.from_numpy(self.sim.get_contact_forces())  # type: ignore
            .to(self.device)
            .tile((self.num_envs, 1, 1))
        )
        self.env_frictions = torch.zeros(
            self.num_envs, 1, dtype=torch.float32, device=self.device
        )

        # self.measured_heights = 0
        # if self.cfg.terrain.measure_heights:
        #     self.height_points = self._init_height_points()

        # forces
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rand_push = torch.zeros(
            (self.num_envs, 6), dtype=torch.float32, device=self.device
        )

    def _init_noise_vec(self):
        # Sets a vector used to scale the noise added to the observations.
        # [NOTE]: Must be adapted when changing the observations structure
        self.noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.scales
        self.noise_vec[0:5] = 0.0  # commands
        self.noise_vec[5:17] = noise_scales.dof_pos * self.obs_scales.dof_pos
        self.noise_vec[17:29] = noise_scales.dof_vel * self.obs_scales.dof_vel
        self.noise_vec[29:41] = 0.0  # previous actions
        self.noise_vec[41:44] = (
            noise_scales.ang_vel * self.obs_scales.ang_vel
        )  # ang vel
        self.noise_vec[44:47] = noise_scales.quat * self.obs_scales.quat  # euler x,y

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
        self.reward_functions: List[Callable[..., jnp.ndarray]] = []
        self.reward_names: List[str] = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            reward_function_name = "_reward_" + name
            self.reward_functions.append(getattr(self, reward_function_name))

        # reward episode sums
        self.episode_sums = {
            name: jnp.zeros(self.num_envs)  # type:ignore
            for name in self.reward_scales.keys()
        }

    def reset(self, rng: jnp.ndarray) -> State:
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

        obs: jax.Array = self._get_obs(data, jnp.zeros(self.sys.nu))  # type:ignore
        reward, done, zero = jnp.zeros(3)
        metrics = {
            "forward_reward": zero,
            "reward_linvel": zero,
            "reward_quadctrl": zero,
            "reward_alive": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
        }
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""

        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        self._integrate_path_frame(data)

        com_before = data0.subtree_com[1]
        com_after = data.subtree_com[1]
        velocity = (com_after - com_before) / self.dt
        forward_reward = self._forward_reward_weight * velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jnp.where(data.q[2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(data.q[2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))

        obs = self._get_obs(data, action)
        reward = forward_reward + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        state.metrics.update(
            forward_reward=forward_reward,
            reward_linvel=forward_reward,
            reward_quadctrl=-ctrl_cost,
            reward_alive=healthy_reward,
            x_position=com_after[0],
            y_position=com_after[1],
            distance_from_origin=jnp.linalg.norm(com_after),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
        )

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def _get_obs(self, data: mjx.Data, action: jnp.ndarray) -> jnp.ndarray:
        """Observes humanoid body position, velocities, and angles."""
        phase = self.episode_step * self.dt / self.cycle_time
        state_ref = self.motion_ref.get_state(self.path_frame, phase, self.command_buf)

        position = data.qpos
        if self._exclude_current_positions_from_observation:
            position = position[2:]

        # external_contact_forces are excluded
        return jnp.concatenate(
            [
                position,
                data.qvel,
                data.cinert[1:].ravel(),
                data.cvel[1:].ravel(),
                data.qfrc_actuator,
            ]
        )

    @jit
    def _integrate_path_frame(self, data: mjx.Data):
        pos, quat = self.path_frame[:3], self.path_frame[3:]
        x_vel = self.command_buf[0]
        y_vel = self.command_buf[1]

        if self.heading_command:
            forward = quat_apply(data.qpos[:3], self.forward_vec)  # type:ignore
            heading = jnp.atan2(forward[1], forward[0])  # type:ignore
            self.command_bwrap_to_piuf[2] = jnp.clip(  # type:ignore
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

    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
