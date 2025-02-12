from typing import List, Tuple

import numpy
from tqdm import tqdm

from toddlerbot.algorithms.zmp_planner import ZMPPlanner
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update, loop_update
from toddlerbot.utils.array_utils import array_lib as np
from toddlerbot.utils.math_utils import quat2euler


class ZMPWalk:
    """A class for generating ZMP-based walking trajectories for a robot."""

    def __init__(
        self,
        robot: Robot,
        cycle_time: float,
        single_double_ratio: float = 2.0,
        foot_step_height: float = 0.05,
        control_dt: float = 0.02,
        control_cost_Q: float = 1.0,
        control_cost_R: float = 1e-1,
    ):
        """Initializes the trajectory planner for a robot, precomputing and storing trajectories for a range of commands.

        Args:
            robot (Robot): The robot instance for which the trajectories are being planned.
            cycle_time (float): The time duration of one complete cycle of the robot's movement.
            single_double_ratio (float, optional): The ratio of single support phase to double support phase. Defaults to 2.0.
            foot_step_height (float, optional): The height of the foot step during the robot's movement. Defaults to 0.05.
            control_dt (float, optional): The time step for control updates. Defaults to 0.02.
            control_cost_Q (float, optional): The weight for the control cost in the Q matrix. Defaults to 1.0.
            control_cost_R (float, optional): The weight for the control cost in the R matrix. Defaults to 0.1.
        """
        self.robot = robot
        self.cycle_time = cycle_time
        self.double_support_phase = cycle_time / 2 / (single_double_ratio + 1)
        self.single_support_phase = single_double_ratio * self.double_support_phase
        self.footstep_height = foot_step_height
        self.control_dt = control_dt
        self.control_cost_Q = control_cost_Q
        self.control_cost_R = control_cost_R

        default_joint_pos = np.array(
            list(robot.default_joint_angles.values()), dtype=np.float32
        )
        joint_groups = numpy.array(
            [robot.joint_groups[name] for name in robot.joint_ordering]
        )
        self.default_leg_joint_pos = default_joint_pos[joint_groups == "leg"]

        self.com_z = robot.config["general"]["offsets"]["default_torso_z"]
        self.foot_to_com_x = float(robot.config["general"]["offsets"]["foot_to_com_x"])
        self.foot_to_com_y = float(robot.config["general"]["offsets"]["foot_to_com_y"])

        self.default_target_z = (
            float(robot.config["general"]["offsets"]["torso_z"]) - self.com_z
        )
        self.zmp_planner = ZMPPlanner()

    def build_lookup_table(
        self,
        command_range: List[List[float]] = [[-0.2, 0.4], [-0.2, 0.2], [-0.8, 0.8]],
        interval: float = 0.02,
    ):
        """Builds a lookup table for command references and associated data.

        Args:
            command_range (List[List[float]]): A list of command ranges for each dimension, where each range is defined by a start and stop value. Defaults to [[-0.2, 0.4], [-0.2, 0.2], [-0.8, 0.8]].
            interval (float): The interval at which to sample command values within each range. Defaults to 0.02.

        Returns:
            Tuple[List[Tuple[float, ...]], List[ArrayType], List[ArrayType], List[ArrayType]]: A tuple containing:
                - A list of tuples representing the command keys.
                - A list of arrays for the center of mass reference data.
                - A list of arrays for the leg joint position reference data.
                - A list of arrays for the stance mask reference data.
        """
        lookup_keys: List[Tuple[float, ...]] = []
        com_ref_list: List[ArrayType] = []
        stance_mask_ref_list: List[ArrayType] = []
        leg_joint_pos_ref_list: List[ArrayType] = []
        path_pos = np.zeros(3, dtype=np.float32)
        path_quat = np.array([1, 0, 0, 0], dtype=np.float32)

        # Create linspace arrays for each command range
        linspaces = [
            np.arange(start, stop + 1e-6, interval, dtype=np.float32)
            for start, stop in command_range
        ]

        # Generate all combinations of command values using meshgrid and then flatten
        command_xy_grid = np.meshgrid(*linspaces[:2], indexing="ij")
        command_xy = np.stack([g.flatten() for g in command_xy_grid], axis=-1)
        command_set = np.concatenate(
            [command_xy, np.zeros((command_xy.shape[0], 1), dtype=np.float32)], axis=1
        )

        zeros_z = np.zeros_like(linspaces[2])
        command_z = np.stack([zeros_z, zeros_z, linspaces[2]], axis=1)
        command_set = np.concatenate([command_set, command_z], axis=0)

        for command in tqdm(command_set, desc="Building Lookup Table"):
            # if np.linalg.norm(command) < 1e-6:
            #     continue

            _, com_ref, leg_joint_pos_ref, stance_mask_ref = self.plan(
                path_pos, path_quat, command
            )
            first_cycle_idx = int(np.ceil(self.cycle_time / self.control_dt))
            com_ref_truncated = com_ref[first_cycle_idx:]
            leg_joint_pos_ref_truncated = leg_joint_pos_ref[first_cycle_idx:]
            stance_mask_ref_truncated = stance_mask_ref[first_cycle_idx:]

            lookup_keys.append(tuple(map(float, command)))
            com_ref_list.append(com_ref_truncated)
            leg_joint_pos_ref_list.append(leg_joint_pos_ref_truncated)
            stance_mask_ref_list.append(stance_mask_ref_truncated)

        return lookup_keys, com_ref_list, leg_joint_pos_ref_list, stance_mask_ref_list

    def plan(
        self,
        path_pos: ArrayType,
        path_quat: ArrayType,
        command: ArrayType,
        total_time: float = 20.0,
        rotation_radius: float = 0.1,
    ) -> Tuple[ArrayType, ArrayType, ArrayType, ArrayType]:
        """Plans the trajectory for a bipedal robot's movement based on the given path and command.

        Args:
            path_pos (ArrayType): The initial position of the path as a 2D array.
            path_quat (ArrayType): The initial orientation of the path in quaternion form.
            command (ArrayType): The desired movement command, including translation and rotation.
            total_time (float, optional): The total time for the planned movement. Defaults to 20.0.
            rotation_radius (float, optional): The radius for rotation movements. Defaults to 0.1.

        Returns:
            Tuple[ArrayType, ArrayType, ArrayType, ArrayType]: A tuple containing:
                - The desired Zero Moment Points (ZMPs) as an array.
                - The trajectory of the center of mass (CoM) as an array.
                - The reference positions for leg joints as an array.
                - The stance mask reference indicating foot contact states as an array.
        """
        path_euler = quat2euler(path_quat)
        pose_curr = np.array(
            [path_pos[0], path_pos[1], path_euler[2]], dtype=np.float32
        )

        footsteps: List[ArrayType] = []
        left_footstep_init = np.array(
            [pose_curr[0], pose_curr[1] + self.foot_to_com_y, pose_curr[2], 0],
            dtype=np.float32,
        )
        right_footstep_init = np.array(
            [pose_curr[0], pose_curr[1] - self.foot_to_com_y, pose_curr[2], 1],
            dtype=np.float32,
        )
        num_cycles = int(np.ceil(total_time / self.cycle_time))
        if np.linalg.norm(command) < 1e-6:
            for _ in range(num_cycles):
                footsteps.append(left_footstep_init)
                footsteps.append(right_footstep_init)

        elif np.linalg.norm(command[:2]) > 1e-6:
            stride = command[:2] * total_time / (2 * num_cycles - 1)
            for i in range(num_cycles):
                left_footstep = left_footstep_init.copy()
                right_footstep = right_footstep_init.copy()
                left_footstep[:2] += i * 2 * stride
                right_footstep[:2] += (i * 2 + 1) * stride
                footsteps.append(left_footstep)
                footsteps.append(right_footstep)

        elif np.abs(command[2]) > 1e-6:
            stride = command[2] * total_time / (2 * num_cycles - 1)
            r = np.sign(command[2]) * rotation_radius
            for i in range(num_cycles):
                # Calculate the new angles for the left and right foot
                angle_left = i * 2 * stride
                angle_right = (i * 2 + 1) * stride

                left_footstep = np.array(
                    [
                        pose_curr[0]
                        + r * np.sin(angle_left)
                        - self.foot_to_com_y * np.sin(angle_left),
                        pose_curr[1]
                        + r * (1 - np.cos(angle_left))
                        + self.foot_to_com_y * np.cos(angle_left),
                        angle_left,
                        0,
                    ],
                    dtype=np.float32,
                )
                right_footstep = np.array(
                    [
                        pose_curr[0]
                        + r * np.sin(angle_right)
                        + self.foot_to_com_y * np.sin(angle_right),
                        pose_curr[1]
                        + r * (1 - np.cos(angle_right))
                        - self.foot_to_com_y * np.cos(angle_right),
                        angle_right,
                        1,
                    ],
                    dtype=np.float32,
                )

                footsteps.append(left_footstep)
                footsteps.append(right_footstep)

        # import numpy

        # from toddlerbot.visualization.vis_plot import plot_footsteps

        # plot_footsteps(
        #     numpy.array(
        #         [numpy.asarray(fs[:3]) for fs in footsteps], dtype=numpy.float32
        #     ),
        #     [int(fs[-1]) for fs in footsteps],
        #     (0.12, 0.042),
        #     self.foot_to_com_y,
        #     title="Footsteps Planning",
        #     x_label="Position X",
        #     y_label="Position Y",
        #     save_config=False,
        #     save_path=".",
        #     file_name=f"footsteps_{'_'.join([str(c) for c in command])}.png",
        # )()

        time_list = np.array(
            [0, self.double_support_phase]
            + [self.single_support_phase, self.double_support_phase]
            * (len(footsteps) - 1),
            dtype=np.float32,
        )
        time_steps = np.cumsum(time_list)
        num_total_steps = int(
            np.ceil((time_steps[-1] - time_steps[0]) / self.control_dt)
        )
        x0 = np.array([path_pos[0], path_pos[1], 0.0, 0.0], dtype=np.float32)

        desired_zmps = [step[:2] for step in footsteps for _ in range(2)]

        if np.linalg.norm(command) < 1e-6:
            x_traj = np.tile(x0, (num_total_steps, 1))
            leg_joint_pos_ref = np.tile(
                self.default_leg_joint_pos, (num_total_steps, 1)
            )
            stance_mask_ref = np.ones((num_total_steps, 2), dtype=np.float32)

        else:
            self.zmp_planner.plan(
                time_steps,
                desired_zmps,
                x0,
                self.com_z,
                Qy=np.eye(2, dtype=np.float32) * self.control_cost_Q,
                R=np.eye(2, dtype=np.float32) * self.control_cost_R,
            )

            def update_step(
                carry: Tuple[ArrayType, ArrayType], idx: int
            ) -> Tuple[Tuple[ArrayType, ArrayType], ArrayType]:
                x_traj, u_traj = carry
                t = time_steps[0] + idx * self.control_dt
                xd = np.hstack((x_traj[idx - 1, 2:], u_traj[idx - 1, :]))
                x_traj = inplace_update(
                    x_traj, idx, x_traj[idx - 1, :] + xd * self.control_dt
                )
                u = self.zmp_planner.get_optim_com_acc(t, x_traj[idx, :])
                u_traj = inplace_update(u_traj, idx, u)

                return (x_traj, u_traj), x_traj[idx]

            # Initialize the arrays
            x_traj = np.zeros((num_total_steps, 4), dtype=np.float32)
            u_traj = np.zeros((num_total_steps, 2), dtype=np.float32)
            # Set the initial conditions
            x_traj = inplace_update(x_traj, 0, x0)
            u_traj = inplace_update(
                u_traj,
                0,
                self.zmp_planner.get_optim_com_acc(time_steps[0], x0),
            )
            x_traj = loop_update(update_step, x_traj, u_traj, (1, num_total_steps))

            (
                left_foot_pos_traj,
                left_foot_ori_traj,
                right_foot_pos_traj,
                right_foot_ori_traj,
                torso_ori_traj,
                stance_mask_ref,
            ) = self.compute_foot_trajectories(
                time_steps,
                np.repeat(np.stack(footsteps), 2, axis=0),
            )

            com_pose_traj = np.concatenate(
                [x_traj[:, :2], torso_ori_traj[:, 2:]],
                axis=-1,
            )
            leg_joint_pos_ref = self.solve_ik(
                left_foot_pos_traj,
                left_foot_ori_traj,
                right_foot_pos_traj,
                right_foot_ori_traj,
                com_pose_traj,
            )

        return np.array(desired_zmps), x_traj, leg_joint_pos_ref, stance_mask_ref

    def compute_foot_trajectories(
        self, time_steps: ArrayType, footsteps: List[ArrayType]
    ) -> Tuple[ArrayType, ...]:
        """Compute the trajectories for the left and right foot positions and orientations, as well as the torso orientation and stance mask over a series of time steps.

        Args:
            time_steps (ArrayType): An array of time steps at which the trajectories are computed.
            footsteps (List[ArrayType]): A list of arrays representing the footstep positions and orientations, with each array containing the x, y, and orientation values.

        Returns:
            Tuple[ArrayType, ...]: A tuple containing arrays for the left foot position trajectory, left foot orientation trajectory, right foot position trajectory, right foot orientation trajectory, torso orientation trajectory, and stance mask trajectory.
        """
        offset = np.array(
            [
                -np.sin(footsteps[0][2]) * self.foot_to_com_y,
                np.cos(footsteps[0][2]) * self.foot_to_com_y,
            ]
        )
        last_pos = np.concatenate(
            [
                footsteps[0][:2] + offset,
                np.zeros(1, dtype=np.float32),
                footsteps[0][:2] - offset,
                np.zeros(1, dtype=np.float32),
            ]
        )
        last_ori = np.array(
            [0.0, 0.0, footsteps[0][2], 0.0, 0.0, footsteps[0][2]], dtype=np.float32
        )
        last_base_ori = np.zeros(3, dtype=np.float32)

        num_total_steps = int(
            np.ceil((time_steps[-1] - time_steps[0]) / self.control_dt)
        )
        left_foot_pos_traj = np.zeros((num_total_steps, 3), dtype=np.float32)
        left_foot_ori_traj = np.zeros((num_total_steps, 3), dtype=np.float32)
        right_foot_pos_traj = np.zeros((num_total_steps, 3), dtype=np.float32)
        right_foot_ori_traj = np.zeros((num_total_steps, 3), dtype=np.float32)
        torso_ori_traj = np.zeros((num_total_steps, 3), dtype=np.float32)
        stance_mask_traj = np.zeros((num_total_steps, 2), dtype=np.float32)
        step_curr = 0
        for i in range(len(time_steps) - 1):
            num_steps = round((time_steps[i + 1] - time_steps[i]) / self.control_dt)
            if num_steps + step_curr > num_total_steps:
                num_steps = num_total_steps - step_curr

            stance_mask = np.tile(np.ones(2, dtype=np.float32), (num_steps, 1))
            if i % 2 == 0:  # Double support
                foot_pos_traj = np.tile(last_pos, (num_steps, 1))
                foot_ori_traj = np.tile(last_ori, (num_steps, 1))
                base_ori_traj = np.tile(last_base_ori, (num_steps, 1))
            else:
                support_leg_curr = int(footsteps[i][-1])
                support_leg_next = int(footsteps[i + 1][-1])
                if support_leg_curr == 2:
                    current_pos = last_pos.copy()
                    current_ori = last_ori.copy()
                    swing_leg = support_leg_next
                else:
                    current_pos = inplace_update(
                        last_pos,
                        slice(support_leg_curr * 3, support_leg_curr * 3 + 2),
                        footsteps[i][:2],
                    )
                    current_ori = inplace_update(
                        last_ori,
                        support_leg_curr * 3 + 2,
                        footsteps[i][2],
                    )
                    swing_leg = 1 - support_leg_curr

                if support_leg_next == 2:
                    offset = np.array(
                        [
                            -np.sin(footsteps[i][2]) * self.foot_to_com_y,
                            np.cos(footsteps[i][2]) * self.foot_to_com_y,
                        ]
                    ) * (-1 if support_leg_curr == 1 else 1)
                else:
                    offset = np.zeros(2, dtype=np.float32)

                target_pos = inplace_update(
                    current_pos,
                    slice(swing_leg * 3, swing_leg * 3 + 2),
                    footsteps[i + 1][:2] + offset,
                )
                target_ori = inplace_update(
                    current_ori, swing_leg * 3 + 2, footsteps[i + 1][2]
                )
                last_pos = target_pos.copy()
                last_ori = target_ori.copy()

                up_delta = self.footstep_height / (num_steps // 2 - 1)
                up_traj = up_delta * np.concatenate(
                    (
                        np.arange(num_steps // 2, dtype=np.float32),
                        np.arange(
                            num_steps - num_steps // 2 - 1, -1, -1, dtype=np.float32
                        ),
                    )
                )
                pos_delta = (target_pos - current_pos) / num_steps
                foot_pos_traj = current_pos + pos_delta * np.arange(num_steps)[:, None]
                foot_pos_traj = inplace_update(
                    foot_pos_traj,
                    (slice(None), swing_leg * 3 + 2),
                    up_traj,
                )

                ori_delta = (target_ori - current_ori) / num_steps
                foot_ori_traj = current_ori + ori_delta * np.arange(num_steps)[:, None]

                base_ori_delta = (
                    target_ori[swing_leg * 3 : swing_leg * 3 + 3] - last_base_ori
                ) / num_steps
                base_ori_traj = (
                    last_base_ori + base_ori_delta * np.arange(num_steps)[:, None]
                )
                last_base_ori = base_ori_traj[-1]

                stance_mask = inplace_update(stance_mask, (slice(None), swing_leg), 0)

            slice_curr = slice(step_curr, step_curr + num_steps)
            left_foot_pos_traj = inplace_update(
                left_foot_pos_traj, slice_curr, foot_pos_traj[:, :3]
            )
            left_foot_ori_traj = inplace_update(
                left_foot_ori_traj, slice_curr, foot_ori_traj[:, :3]
            )
            right_foot_pos_traj = inplace_update(
                right_foot_pos_traj, slice_curr, foot_pos_traj[:, 3:]
            )
            right_foot_ori_traj = inplace_update(
                right_foot_ori_traj, slice_curr, foot_ori_traj[:, 3:]
            )
            torso_ori_traj = inplace_update(torso_ori_traj, slice_curr, base_ori_traj)
            stance_mask_traj = inplace_update(stance_mask_traj, slice_curr, stance_mask)

            step_curr += int(num_steps)

        return (
            left_foot_pos_traj,
            left_foot_ori_traj,
            right_foot_pos_traj,
            right_foot_ori_traj,
            torso_ori_traj,
            stance_mask_traj,
        )

    def solve_ik(
        self,
        left_foot_pos_traj: ArrayType,
        left_foot_ori_traj: ArrayType,
        right_foot_pos_traj: ArrayType,
        right_foot_ori_traj: ArrayType,
        com_pose_traj: ArrayType,
    ):
        """Solves the inverse kinematics (IK) for a bipedal robot's legs based on foot and center of mass (COM) trajectories.

        This function computes the joint positions for the left and right legs by adjusting the foot position and orientation trajectories relative to the COM trajectory. It then uses inverse kinematics to determine the necessary joint angles for achieving the desired foot trajectories.

        Args:
            left_foot_pos_traj (ArrayType): Trajectory of the left foot positions.
            left_foot_ori_traj (ArrayType): Trajectory of the left foot orientations.
            right_foot_pos_traj (ArrayType): Trajectory of the right foot positions.
            right_foot_ori_traj (ArrayType): Trajectory of the right foot orientations.
            com_pose_traj (ArrayType): Trajectory of the center of mass poses.

        Returns:
            ArrayType: Combined joint position trajectories for both left and right legs.
        """
        com_pos_traj = np.hstack(
            [com_pose_traj[:, :2], np.zeros((com_pose_traj.shape[0], 1))]
        )
        com_ori_traj = np.hstack(
            [np.zeros((com_pose_traj.shape[0], 2)), com_pose_traj[:, 2:]]
        )
        foot_to_com_offset = np.concatenate(
            [
                np.sin(com_pose_traj[:, 2:]) * self.foot_to_com_y,
                -np.cos(com_pose_traj[:, 2:]) * self.foot_to_com_y,
                np.zeros((com_pose_traj.shape[0], 1)),
            ],
            axis=-1,
        )
        left_foot_adjusted_pos = left_foot_pos_traj - com_pos_traj + foot_to_com_offset
        right_foot_adjusted_pos = (
            right_foot_pos_traj - com_pos_traj - foot_to_com_offset
        )
        left_foot_adjusted_ori = left_foot_ori_traj - com_ori_traj
        right_foot_adjusted_ori = right_foot_ori_traj - com_ori_traj

        left_leg_joint_pos_traj = self.foot_ik(
            left_foot_adjusted_pos,
            left_foot_adjusted_ori,
            side="left",
        )
        right_leg_joint_pos_traj = self.foot_ik(
            right_foot_adjusted_pos,
            right_foot_adjusted_ori,
            side="right",
        )

        # Combine the results for left and right legs
        leg_joint_pos_traj = np.hstack(
            [left_leg_joint_pos_traj, right_leg_joint_pos_traj]
        )

        return leg_joint_pos_traj

    def foot_ik(
        self,
        target_foot_pos: ArrayType,
        target_foot_ori: ArrayType,
        side: str = "left",
    ) -> ArrayType:
        """Calculates the inverse kinematics for a robot's foot, determining the necessary joint angles to achieve a specified foot position and orientation.

        Args:
            target_foot_pos (ArrayType): The target position of the foot in 3D space, specified as an array with shape (n, 3), where each row represents the x, y, and z coordinates.
            target_foot_ori (ArrayType): The target orientation of the foot, specified as an array with shape (n, 3), where each row represents the roll, pitch, and yaw angles.
            side (str, optional): The side of the robot for which the calculations are performed. Defaults to "left".

        Returns:
            ArrayType: An array of joint angles required to achieve the specified foot position and orientation, with shape (n, 6), where each row contains the angles for hip pitch, hip roll, hip yaw, knee pitch, ankle roll, and ankle pitch.
        """
        target_x = target_foot_pos[:, 0]
        target_y = target_foot_pos[:, 1]
        target_z = target_foot_pos[:, 2]
        ank_roll = target_foot_ori[:, 0]
        ank_pitch = target_foot_ori[:, 1]
        hip_yaw = target_foot_ori[:, 2]

        offsets = self.robot.config["general"]["offsets"]

        transformed_x = target_x * np.cos(hip_yaw) + target_y * np.sin(hip_yaw)
        transformed_y = target_x * np.sin(hip_yaw) - target_y * np.cos(hip_yaw)
        transformed_z = (
            offsets["hip_pitch_to_knee_z"]
            + offsets["knee_to_ank_pitch_z"]
            - target_z
            - self.default_target_z
        )

        hip_roll = np.arctan2(
            transformed_y, transformed_z + offsets["hip_roll_to_pitch_z"]
        )

        leg_projected_yz_length = np.sqrt(transformed_y**2 + transformed_z**2)
        leg_length = np.sqrt(transformed_x**2 + leg_projected_yz_length**2)
        leg_pitch = np.arctan2(transformed_x, leg_projected_yz_length)
        hip_disp_cos = (
            leg_length**2
            + offsets["hip_pitch_to_knee_z"] ** 2
            - offsets["knee_to_ank_pitch_z"] ** 2
        ) / (2 * leg_length * offsets["hip_pitch_to_knee_z"])
        hip_disp = np.arccos(np.clip(hip_disp_cos, -1.0, 1.0))
        ank_disp = np.arcsin(
            offsets["hip_pitch_to_knee_z"]
            / offsets["knee_to_ank_pitch_z"]
            * np.sin(hip_disp)
        )
        hip_pitch = leg_pitch + hip_disp
        knee_pitch = hip_disp + ank_disp
        ank_pitch += knee_pitch - hip_pitch

        if side == "left":
            return np.vstack(
                [
                    hip_pitch,
                    hip_roll,
                    -hip_yaw,
                    -knee_pitch,
                    hip_roll - ank_roll,
                    -ank_pitch,
                ]
            ).T
        else:
            return np.vstack(
                [
                    -hip_pitch,
                    -hip_roll,
                    -hip_yaw,
                    knee_pitch,
                    hip_roll - ank_roll,
                    ank_pitch,
                ]
            ).T
