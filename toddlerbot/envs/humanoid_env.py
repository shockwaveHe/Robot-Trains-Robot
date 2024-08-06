#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from collections import deque
from dataclasses import asdict
from typing import Any, Callable, Dict, List

import numpy as np
import torch

from toddlerbot.envs.humanoid_config import HumanoidCfg
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import (
    quat_apply,
    quat_rotate_inverse,
    quat_to_euler_tensor,
    torch_rand_float,
    wrap_to_pi,
)


class HumanoidEnv:
    def __init__(self, sim: MuJoCoSim, robot: Robot, cfg: HumanoidCfg):
        self.sim = sim
        self.robot = robot
        self.cfg = cfg

        # misc
        self.device = torch.device(self.sim.device_type)
        self.dt = self.cfg.control.decimation * self.sim.dt
        self.common_step_counter = 0
        self.extras: Dict[str, Any] = {}

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)  # type: ignore
        torch._C._jit_set_profiling_executor(False)  # type: ignore

        self.init_done = False

        self.sim.forward()

        self._init_env()
        self._init_dof()
        self._init_root()
        self._init_body()
        self._init_noise_vec()
        self._init_reward()

        self.init_done = True

    def _init_env(self):
        self.num_envs = self.cfg.env.num_envs
        self.num_obs = self.cfg.env.num_observations
        self.num_privileged_obs = self.cfg.env.num_privileged_obs
        self.num_actions = len(self.robot.motor_ordering)

        self.obs_scales = self.cfg.normalization.scales
        self.command_ranges = asdict(self.cfg.commands.ranges)
        # if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
        #     self.cfg.terrain.curriculum = False

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

        # buffers
        self.obs_buf = torch.zeros(
            self.num_envs, self.num_obs, device=self.device, dtype=torch.float
        )
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # new reward buffers for exp rewrads
        self.neg_reward_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.pos_reward_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )

        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.time_out_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        if self.num_privileged_obs:
            self.privileged_obs_buf = torch.zeros(
                self.num_envs,
                self.num_privileged_obs,
                device=self.device,
                dtype=torch.float,
            )
        else:
            self.privileged_obs_buf = None

        # history
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)  # type: ignore
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)  # type: ignore
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(  # type: ignore
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(  # type: ignore
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_privileged_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )

        # actions
        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # commands
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device,
            requires_grad=False,
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

    # def _init_height_points(self):
    #     """Returns points at which the height measurments are sampled (in base frame)

    #     Returns:
    #         [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
    #     """
    #     y = torch.tensor(
    #         self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False
    #     )
    #     x = torch.tensor(
    #         self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False
    #     )
    #     grid_x, grid_y = torch.meshgrid(x, y)

    #     self.num_height_points = grid_x.numel()
    #     points = torch.zeros(
    #         self.num_envs,
    #         self.num_height_points,
    #         3,
    #         device=self.device,
    #         requires_grad=False,
    #     )
    #     points[:, :, 0] = grid_x.flatten()
    #     points[:, :, 1] = grid_y.flatten()
    #     return points

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
        self.reward_functions: List[Callable[..., torch.Tensor]] = []
        self.reward_names: List[str] = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for name in self.reward_scales.keys()
        }

    # def _create_envs(self):
    #     start_pose = gymapi.Transform()
    #     start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

    #     self._get_env_origins()
    #     env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
    #     env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
    #     self.actor_handles = []
    #     self.envs = []
    #     self.env_frictions = torch.zeros(
    #         self.num_envs, 1, dtype=torch.float32, device=self.device
    #     )

    #     for i in range(self.num_envs):
    #         # create env instance
    #         env_handle = self.gym.create_env(
    #             self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
    #         )
    #         pos = self.env_origins[i].clone()
    #         pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(
    #             1
    #         )
    #         start_pose.p = gymapi.Vec3(*pos)

    #         rigid_shape_props = self._process_body_props(
    #             rigid_shape_props_asset, i
    #         )
    #         self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
    #         actor_handle = self.gym.create_actor(
    #             env_handle,
    #             robot_asset,
    #             start_pose,
    #             self.cfg.asset.name,
    #             i,
    #             self.cfg.asset.self_collisions,
    #             0,
    #         )
    #         dof_props = self._process_dof_props(dof_props_asset, i)
    #         self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
    #         body_props = self.gym.get_actor_rigid_body_properties(
    #             env_handle, actor_handle
    #         )
    #         body_props = self._process_rigid_body_props(body_props, i)
    #         self.gym.set_actor_rigid_body_properties(
    #             env_handle, actor_handle, body_props, recomputeInertia=True
    #         )
    #         self.envs.append(env_handle)
    #         self.actor_handles.append(actor_handle)

    def step(self, actions: torch.Tensor):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        for _ in range(self.cfg.control.decimation):
            self.sim.set_motor_angles(self.actions.squeeze().cpu().numpy())
            self.sim.step()

        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )

        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def post_physics_step(self):
        self.dof_state = (
            torch.from_numpy(self.sim.get_dof_state())  # type: ignore
            .to(self.device)
            .tile((self.num_envs, 1, 1))
        )
        self.dof_pos = self.dof_state[..., 0]
        self.dof_vel = self.dof_state[..., 1]

        self.root_states = (
            torch.from_numpy(self.sim.get_root_state())  # type: ignore
            .to(self.device)
            .tile((self.num_envs, 1))
        )
        # TODO: should this be in local frame or global frame?
        self.body_state = (
            torch.from_numpy(self.sim.get_body_state())  # type: ignore
            .to(self.device)
            .tile((self.num_envs, 1, 1))
        )
        self.contact_forces = (
            torch.from_numpy(self.sim.get_contact_forces())  # type: ignore
            .to(self.device)
            .tile((self.num_envs, 1, 1))
        )

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        self.base_euler_xyz = quat_to_euler_tensor(self.base_quat)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()

        self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self.compute_observations()

        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_body_state[:] = self.body_state[:]

    def _post_physics_step_callback(self):
        env_ids = (
            (
                self.episode_length_buf
                % int(self.cfg.commands.resampling_time / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )

        self._resample_commands(env_ids)

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0
            )

        # if self.cfg.terrain.measure_heights:
        #     self.measured_heights = self._get_heights()

        if self.cfg.domain_rand.push_robots and (
            self.common_step_counter % self.push_interval == 0
        ):
            self._push_robots()

    def _resample_commands(self, env_ids: torch.Tensor):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1) > 0.2  # type: ignore
        ).unsqueeze(1)

    # def _get_heights(self) -> torch.Tensor:
    #     """Samples heights of the terrain at required points around each robot.
    #         The points are offset by the base's position and rotated by the base's yaw

    #     Args:
    #         env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

    #     Raises:
    #         NameError: [description]

    #     Returns:
    #         [type]: [description]
    #     """
    #     if self.cfg.terrain.mesh_type == "plane":
    #         return torch.zeros(
    #             self.num_envs,
    #             self.num_height_points,
    #             device=self.device,
    #             requires_grad=False,
    #         )
    #     elif self.cfg.terrain.mesh_type == "none":
    #         raise NameError("Can't measure height with terrain mesh type 'none'")

    #     points = quat_apply_yaw(
    #         self.base_quat.repeat(1, self.num_height_points), self.height_points
    #     ) + (self.root_states[:, :3]).unsqueeze(1)

    #     points += self.terrain.cfg.border_size  # type: ignore
    #     points = (points / self.terrain.cfg.horizontal_scale).long()  # type: ignore
    #     px = torch.clip(points[:, :, 0].view(-1), 0, self.height_samples.shape[0] - 2)  # type: ignore
    #     py = torch.clip(points[:, :, 1].view(-1), 0, self.height_samples.shape[1] - 2)  # type: ignore

    #     heights1 = self.height_samples[px, py]
    #     heights2 = self.height_samples[px + 1, py]
    #     heightXBotL = self.height_samples[px, py + 1]
    #     heights = torch.min(heights1, heights2)  # type: ignore
    #     heights = torch.min(heights, heightXBotL)  # type: ignore

    #     return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale  # type: ignore

    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        max_push_xy = self.cfg.domain_rand.max_push_xy_vel
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push[:, :2] = torch_rand_float(
            -max_push_xy, max_push_xy, (self.num_envs, 2), device=self.device
        )
        self.rand_push[:, 3:] = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device
        )
        push_duration = np.random.uniform(0.0, self.cfg.domain_rand.max_push_duration)

        self.sim.start_push(self.rand_push.squeeze().cpu().numpy(), push_duration)

    def check_termination(self):
        """Check if environments need to be reset"""
        termination_contact = self.contact_forces[
            :, self.termination_contact_indices, :
        ]
        self.reset_buf = torch.any(
            torch.norm(termination_contact, dim=-1) > 1.0,  # type: ignore
            dim=1,
        )
        self.time_out_buf = (
            self.episode_length_buf > self.max_episode_length
        )  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset(self):
        """Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device)
        )
        return obs, privileged_obs

    def reset_idx(self, env_ids: torch.Tensor):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        # update curriculum
        # if self.cfg.terrain.curriculum:
        #     self._update_terrain_curriculum(env_ids)

        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (
            self.common_step_counter % self.max_episode_length == 0
        ):
            self._update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)

        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_last_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_body_state[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0

        # log additional curriculum info
        # if self.cfg.terrain.mesh_type == "trimesh":
        #     self.extras["episode"]["terrain_level"] = torch.mean(
        #         self.terrain_levels.float()
        #     )
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][
                1
            ]

        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # fix reset gravity bug
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz = quat_to_euler_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.gravity_vec[env_ids]
        )

        for i in range(self.obs_history.maxlen):  # type: ignore
            self.obs_history[i][env_ids] *= 0  # type: ignore

        for i in range(self.critic_history.maxlen):  # type: ignore
            self.critic_history[i][env_ids] *= 0  # type: ignore

    # def _update_terrain_curriculum(self, env_ids: torch.Tensor):
    #     # don't change on initial reset
    #     if not self.init_done:
    #         return

    #     distance = torch.norm(  # type: ignore
    #         self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1
    #     )
    #     # robots that walked far enough progress to harder terains
    #     move_up = distance > self.terrain.env_length / 2  # type: ignore
    #     # robots that walked less than half of their required distance go to simpler terrains
    #     move_down = (  # type: ignore
    #         (
    #             distance
    #             < torch.norm(self.commands[env_ids, :2], dim=1)  # type: ignore
    #             * self.max_episode_length_s
    #             * 0.5
    #         )
    #         * ~move_up
    #     )
    #     self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
    #     # Robots that solve the last level are sent to a random one
    #     self.terrain_levels[env_ids] = torch.where(
    #         self.terrain_levels[env_ids] >= self.max_terrain_level,
    #         torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
    #         torch.clip(self.terrain_levels[env_ids], 0),
    #     )  # (the minumum level is zero)
    #     self.env_origins[env_ids] = self.terrain_origins[
    #         self.terrain_levels[env_ids], self.terrain_types[env_ids]
    #     ]

    def _update_command_curriculum(self, env_ids: torch.Tensor):
        """Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if (
            torch.mean(self.episode_sums["tracking_lin_vel"][env_ids])
            / self.max_episode_length
            > 0.8 * self.reward_scales["tracking_lin_vel"]
        ):
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.5,
                -self.cfg.commands.max_curriculum,
                0.0,
            )
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5,
                0.0,
                self.cfg.commands.max_curriculum,
            )

    def _reset_dofs(self, env_ids: torch.Tensor):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_state[env_ids, :, 0] = self.default_dof_pos
        # TODO: bring this back
        # self.dof_state[env_ids, :, 0] += torch_rand_float(
        #     -0.1, 0.1, (len(env_ids), self.num_dof), device=self.device
        # )
        self.dof_state[env_ids, :, 1] = 0.0

        self.sim.reset_dof_state()

    def _reset_root_states(self, env_ids: torch.Tensor):
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0, 1.0, (len(env_ids), 2), device=self.device
            )  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # base velocities
            # self.root_states[env_ids, 7:13] = torch_rand_float(-0.05, 0.05, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
            if self.sim.fixed_base:
                self.root_states[env_ids, 7:13] = 0

        self.sim.set_root_state(self.root_states.squeeze().cpu().numpy())

    def compute_observations(self):
        pass

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

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

        # add termination reward after clipping
        # if "termination" in self.reward_scales:
        #     rew = self._reward_termination() * self.reward_scales["termination"]
        #     self.rew_buf += rew
        #     self.episode_sums["termination"] += rew

    # TODO: Bring these domain randomization functions back
    # ------------- Callbacks --------------
    # def _process_body_props(self, props, env_id: int):
    #     """Callback allowing to store/change/randomize the rigid shape properties of each environment.
    #         Called During environment creation.
    #         Base behavior: randomizes the friction of each environment

    #     Args:
    #         props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
    #         env_id (int): Environment id

    #     Returns:
    #         [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
    #     """
    #     if self.cfg.domain_rand.randomize_friction:
    #         if env_id == 0:
    #             # prepare friction randomization
    #             friction_range = self.cfg.domain_rand.friction_range
    #             num_buckets = 256
    #             bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
    #             friction_buckets = torch_rand_float(
    #                 friction_range[0],
    #                 friction_range[1],
    #                 (num_buckets, 1),
    #                 device=self.device,
    #             )
    #             self.friction_coeffs = friction_buckets[bucket_ids]

    #         for s in range(len(props)):
    #             props[s].friction = self.friction_coeffs[env_id]

    #     return props

    # def _process_rigid_body_props(self, props, env_id):
    #     # randomize base mass
    #     if self.cfg.domain_rand.randomize_base_mass:
    #         rng = self.cfg.domain_rand.added_mass_range
    #         props[0].mass += np.random.uniform(rng[0], rng[1])

    #     return props

    # def _create_heightfield(self):
    #     """Adds a heightfield terrain to the simulation, sets parameters based on the cfg."""
    #     hf_params = gymapi.HeightFieldParams()
    #     hf_params.column_scale = self.terrain.cfg.horizontal_scale
    #     hf_params.row_scale = self.terrain.cfg.horizontal_scale
    #     hf_params.vertical_scale = self.terrain.cfg.vertical_scale
    #     hf_params.nbRows = self.terrain.tot_cols
    #     hf_params.nbColumns = self.terrain.tot_rows
    #     hf_params.transform.p.x = -self.terrain.cfg.border_size
    #     hf_params.transform.p.y = -self.terrain.cfg.border_size
    #     hf_params.transform.p.z = 0.0
    #     hf_params.static_friction = self.cfg.terrain.static_friction
    #     hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
    #     hf_params.restitution = self.cfg.terrain.restitution

    #     self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
    #     self.height_samples = (
    #         torch.tensor(self.terrain.heightsamples)
    #         .view(self.terrain.tot_rows, self.terrain.tot_cols)
    #         .to(self.device)
    #     )

    # def _create_trimesh(self):
    #     """Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
    #     #"""
    #     tm_params = gymapi.TriangleMeshParams()
    #     tm_params.nb_vertices = self.terrain.vertices.shape[0]
    #     tm_params.nb_triangles = self.terrain.triangles.shape[0]

    #     tm_params.transform.p.x = -self.terrain.cfg.border_size
    #     tm_params.transform.p.y = -self.terrain.cfg.border_size
    #     tm_params.transform.p.z = 0.0
    #     tm_params.static_friction = self.cfg.terrain.static_friction
    #     tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
    #     tm_params.restitution = self.cfg.terrain.restitution
    #     self.gym.add_triangle_mesh(
    #         self.sim,
    #         self.terrain.vertices.flatten(order="C"),
    #         self.terrain.triangles.flatten(order="C"),
    #         tm_params,
    #     )
    #     self.height_samples = (
    #         torch.tensor(self.terrain.heightsamples)
    #         .view(self.terrain.tot_rows, self.terrain.tot_cols)
    #         .to(self.device)
    #     )
