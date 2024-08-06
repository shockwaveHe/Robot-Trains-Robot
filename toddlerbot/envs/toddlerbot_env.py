#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


import torch

from toddlerbot.envs.humanoid_config import HumanoidCfg
from toddlerbot.envs.humanoid_env import HumanoidEnv
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot


class ToddlerbotEnv(HumanoidEnv):
    def __init__(self, sim: MuJoCoSim, robot: Robot, cfg: HumanoidCfg):
        super().__init__(sim, robot, cfg)

        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.last_foot_z = self.robot.foot_z

        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.compute_observations()

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        # > or < depends on the robot configuration
        # When sin_pos > 0, right foot is in stance phase
        # When sin_pos < 0, left foot is in stance phase
        sin_pos_l[sin_pos_l < 0] = 0
        self.ref_dof_pos[:, 2] = sin_pos_l * scale_1
        self.ref_dof_pos[:, 3] = sin_pos_l * scale_2
        self.ref_dof_pos[:, 4] = sin_pos_l * scale_1
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r > 0] = 0
        self.ref_dof_pos[:, 8] = sin_pos_r * scale_1
        self.ref_dof_pos[:, 9] = sin_pos_r * scale_2
        self.ref_dof_pos[:, 10] = sin_pos_r * scale_1
        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

        self.ref_action = 2 * self.ref_dof_pos

    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    # def create_sim(self):
    #     """Creates simulation, terrain and evironments"""
    #     self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
    #     self.sim = self.gym.create_sim(
    #         self.sim_device_id,
    #         self.graphics_device_id,
    #         self.physics_engine,
    #         self.sim_params,
    #     )
    #     mesh_type = self.cfg.terrain.mesh_type
    #     if mesh_type in ["heightfield", "trimesh"]:
    #         self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
    #     if mesh_type == "plane":
    #         self._create_ground_plane()
    #     elif mesh_type == "heightfield":
    #         self._create_heightfield()
    #     elif mesh_type == "trimesh":
    #         self._create_trimesh()
    #     elif mesh_type is not None:
    #         raise ValueError(
    #             "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]"
    #         )
    #     self._create_envs()

    def step(self, actions: torch.Tensor):
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action

        # dynamic randomization
        # TODO: why delay like this?
        delay: torch.Tensor = torch.rand((self.num_envs, 1), device=self.device)
        actions = (1 - delay) * actions + delay * self.actions
        actions += (
            self.cfg.domain_rand.dynamic_randomization
            * torch.randn_like(actions)
            * actions
        )
        return super().step(actions)

    def compute_observations(self):
        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = (
            self.contact_forces[:, self.feet_indices, 2] > self.contact_force_threshold
        )

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1
        )

        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        diff = self.dof_pos - self.ref_dof_pos

        self.privileged_obs_buf = torch.cat(
            (
                self.command_input,  # 2 + 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 30
                self.dof_vel * self.obs_scales.dof_vel,  # 30
                self.actions,  # 30
                diff,  # 30
                self.base_lin_vel * self.obs_scales.lin_vel,  # 3
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.obs_scales.quat,  # 3
                self.rand_push,  # 6
                self.env_frictions,  # 1
                self.body_mass / 30.0,  # 1
                stance_mask,  # 2
                contact_mask,  # 2
            ),
            dim=-1,
        )

        obs_buf = torch.cat(
            (
                self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
                q,  # 30
                dq,  # 30
                self.actions,  # 30
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.obs_scales.quat,  # 3
            ),
            dim=-1,
        )

        # if self.cfg.terrain.measure_heights:
        #     heights = (
        #         torch.clip(
        #             self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
        #             -1,
        #             1.0,
        #         )
        #         * self.obs_scales.height_measurements
        #     )
        #     self.privileged_obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        if self.add_noise:
            obs_now = (
                obs_buf.clone()
                + torch.randn_like(obs_buf)
                * self.noise_vec
                * self.cfg.noise.noise_level
            )
        else:
            obs_now = obs_buf.clone()

        self.obs_history.append(obs_now)  # type: ignore
        self.critic_history.append(self.privileged_obs_buf)  # type: ignore

        obs_buf_all = torch.stack(
            [self.obs_history[i] for i in range(self.obs_history.maxlen)],  # type: ignore
            dim=1,
        )  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat(
            [self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)],  # type: ignore
            dim=1,
        )

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask

    # ================================================ Rewards ================================================== #

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes.
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed.
        This function checks if the robot is moving too slow, too fast, or at the desired speed,
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(self.base_lin_vel[:, 0]) != torch.sign(
            self.commands[:, 0]
        )

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.0
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)

    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(
            -torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10
        )
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)  # type: ignore
        return (quat_mismatch + orientation) / 2.0

    def _reward_default_dof_pos(self) -> torch.Tensor:
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_dof_pos
        return -0.01 * torch.norm(joint_diff, dim=1)  # type: ignore

    # TODO: check this
    def _reward_dof_pos(self) -> torch.Tensor:
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        dof_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = dof_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(  # type: ignore
            diff, dim=1
        ).clamp(0, 0.5)
        return r  # type: ignore

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(
            torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1
        )

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height
        of its feet when they are in contact with the ground.
        """
        stance_mask = self._get_gait_phase()
        measured_heights = torch.sum(
            self.body_state[:, self.feet_indices, 2] * stance_mask, dim=1
        ) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - self.robot.foot_z)
        return torch.exp(
            -torch.abs(base_height - self.cfg.rewards.base_height_target) * 100
        )

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)  # type: ignore
        return rew

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = (
            self.contact_forces[:, self.feet_indices, 2] > self.contact_force_threshold
        )
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(
            torch.logical_or(contact, stance_mask), self.last_contacts
        )
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    # TODO: check this
    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = (
            self.contact_forces[:, self.feet_indices, 2] > self.contact_force_threshold
        )

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.body_state[:, self.feet_indices, 2] - self.robot.foot_z
        delta_z = feet_z - self.last_foot_z
        self.feet_height += delta_z
        self.last_foot_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = (
            torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        )
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum(
            (
                torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)  # type: ignore
                - self.cfg.rewards.max_contact_force
            ).clip(0, 400),
            dim=1,
        )

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase.
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = (
            self.contact_forces[:, self.feet_indices, 2] > self.contact_force_threshold
        )
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1.0, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        feet_pos = self.body_state[:, self.feet_indices, :2]
        feet_dist = torch.norm(feet_pos[:, 0, :] - feet_pos[:, 1, :], dim=1)  # type: ignore
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(feet_dist - fd, -0.5, 0.0)  # type: ignore
        d_max = torch.clamp(feet_dist - max_df, 0, 0.5)  # type: ignore
        return (
            torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)
        ) / 2

    # TODO: check this
    def _reward_feet_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = (
            self.contact_forces[:, self.feet_indices, 2] > self.contact_force_threshold
        )
        foot_speed_norm = torch.norm(  # type: ignore
            self.body_state[:, self.feet_indices, 10:12], dim=2
        )
        rew = torch.sqrt(foot_speed_norm)  # type: ignore
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(
            1.0
            * (
                torch.norm(  # type: ignore
                    self.contact_forces[:, self.penalized_contact_indices, :], dim=-1
                )
                > 0.1
            ),
            dim=1,
        )

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(
            torch.square(self.actions + self.last_last_actions - 2 * self.last_actions),
            dim=1,
        )
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3

    # def _reward_knee_distance(self):
    #     """
    #     Calculates the reward based on the distance between the knee of the humanoid.
    #     """
    #     feet_pos = self.body_state[:, self.feet_indices, :2]
    #     foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
    #     fd = self.cfg.rewards.min_dist
    #     max_df = self.cfg.rewards.max_dist / 2
    #     d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
    #     d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
    #     return (
    #         torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)
    #     ) / 2
