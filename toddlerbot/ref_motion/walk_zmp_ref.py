import os
import pickle
from typing import List, Optional, Tuple

import jax
import numpy
from tqdm import tqdm

from toddlerbot.algorithms.zmp.zmp_planner import ZMPPlanner
from toddlerbot.ref_motion import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update, loop_update
from toddlerbot.utils.array_utils import array_lib as np
from toddlerbot.utils.math_utils import quat2euler


class WalkZMPReference(MotionReference):
    def __init__(
        self,
        robot: Robot,
        cycle_time: float,
        command_ranges: List[List[float]],
        single_double_ratio: float = 2.0,
        foot_step_height: float = 0.03,
        control_dt: float = 0.02,
        control_cost_Q: float = 1.0,
        control_cost_R: float = 0.1,
    ):
        super().__init__("walk_zmp", "periodic", robot)

        self.cycle_time = cycle_time
        self.double_support_phase = cycle_time / 2 / (single_double_ratio + 1)
        self.single_support_phase = single_double_ratio * self.double_support_phase
        self.footstep_height = foot_step_height
        self.control_dt = control_dt
        self.control_cost_Q = control_cost_Q
        self.control_cost_R = control_cost_R

        self.com_z = robot.config["general"]["offsets"]["torso_z_default"]
        self.foot_to_com_x = float(robot.data_dict["offsets"]["foot_to_com_x"])
        self.foot_to_com_y = float(robot.data_dict["offsets"]["foot_to_com_y"])

        self.default_joint_pos = np.array(
            list(robot.default_joint_angles.values()), dtype=np.float32
        )
        self.default_joint_vel = np.zeros_like(self.default_joint_pos)

        self.default_target_z = (
            float(robot.config["general"]["offsets"]["torso_z"]) - self.com_z
        )

        joint_groups = numpy.array(
            [robot.joint_groups[name] for name in robot.joint_ordering]
        )
        self.leg_joint_indices = np.arange(len(robot.joint_ordering))[
            joint_groups == "leg"
        ]

        self.left_hip_yaw_idx = robot.motor_ordering.index("left_hip_yaw_drive")
        self.right_hip_yaw_idx = robot.motor_ordering.index("right_hip_yaw_drive")

        self.zmp_planner = ZMPPlanner()

        self.lookup_table_path = os.path.join(
            "toddlerbot", "ref_motion", "walk_zmp_lookup_table.pkl"
        )
        self.build_lookup_table(command_ranges)

    def get_phase_signal(
        self, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        phase_signal = np.array(
            [
                np.sin(2 * np.pi * time_curr / self.cycle_time),
                np.cos(2 * np.pi * time_curr / self.cycle_time),
            ],
            dtype=np.float32,
        )
        return phase_signal

    # @profile()
    def get_state_ref(
        self,
        path_pos: ArrayType,
        path_quat: ArrayType,
        time_curr: Optional[float | ArrayType] = None,
        command: Optional[ArrayType] = None,
    ) -> ArrayType:
        if time_curr is None:
            raise ValueError(f"time_curr is required for {self.name}")

        if command is None:
            raise ValueError(f"command is required for {self.name}")

        linear_vel = np.array([command[0], command[1], 0.0], dtype=np.float32)
        angular_vel = np.array([0.0, 0.0, command[2]], dtype=np.float32)

        # is_zero_commmand = np.linalg.norm(command) < 1e-6
        nearest_command_idx = np.argmin(
            np.linalg.norm(self.lookup_keys - command, axis=1)
        )
        idx = np.round(time_curr / self.control_dt).astype(int)
        joint_pos = self.default_joint_pos.copy()
        joint_pos = inplace_update(
            joint_pos,
            self.leg_joint_indices,
            self.leg_joint_pos_lookup[nearest_command_idx][
                (idx % self.lookup_length[nearest_command_idx]).astype(int)
            ],
        )

        joint_vel = self.default_joint_vel.copy()

        stance_mask = self.stance_mask_lookup[nearest_command_idx][
            (idx % self.lookup_length[nearest_command_idx]).astype(int)
        ]

        return np.concatenate(
            (
                path_pos,
                path_quat,
                linear_vel,
                angular_vel,
                joint_pos,
                joint_vel,
                stance_mask,
            )
        )

    def override_motor_target(
        self, motor_target: ArrayType, state_ref: ArrayType
    ) -> ArrayType:
        motor_target = inplace_update(
            motor_target,
            self.neck_actuator_indices,
            self.default_motor_pos[self.neck_actuator_indices],
        )
        motor_target = inplace_update(
            motor_target,
            self.arm_actuator_indices,
            self.default_motor_pos[self.arm_actuator_indices],
        )
        # motor_target = inplace_update(
        #     motor_target,
        #     self.waist_actuator_indices,
        #     self.default_motor_pos[self.waist_actuator_indices],
        # )
        motor_target = inplace_update(
            motor_target,
            self.left_hip_yaw_idx,
            self.default_motor_pos[self.left_hip_yaw_idx],
        )
        motor_target = inplace_update(
            motor_target,
            self.right_hip_yaw_idx,
            self.default_motor_pos[self.right_hip_yaw_idx],
        )

        return motor_target

    def build_lookup_table(
        self, command_ranges: List[List[float]], interval: float = 0.01
    ):
        """
        Precompute and store the trajectories for a range of commands.
        """
        lookup_keys: List[Tuple[float, ...]] = []
        stance_mask_ref_list: List[ArrayType] = []
        leg_joint_pos_ref_list: List[ArrayType] = []
        if os.path.exists(self.lookup_table_path):
            with open(self.lookup_table_path, "rb") as f:
                lookup_keys, stance_mask_ref_list, leg_joint_pos_ref_list = pickle.load(
                    f
                )
        else:
            path_pos = np.zeros(3, dtype=np.float32)
            path_quat = np.array([1, 0, 0, 0], dtype=np.float32)

            # Create linspace arrays for each command range
            linspaces = [
                np.arange(start, stop + 1e-6, interval, dtype=np.float32)
                for start, stop in command_ranges
            ]

            zeros_x = np.zeros_like(linspaces[0])
            zeros_y = np.zeros_like(linspaces[1])
            zeros_z = np.zeros_like(linspaces[2])

            command_spectrum_x = np.stack([linspaces[0], zeros_x, zeros_x], axis=1)
            command_spectrum_y = np.stack([zeros_y, linspaces[1], zeros_y], axis=1)
            command_spectrum_z = np.stack([zeros_z, zeros_z, linspaces[2]], axis=1)
            # Concatenate all command spectrums
            command_spectrum = np.concatenate(
                [command_spectrum_x, command_spectrum_y, command_spectrum_z], axis=0
            )
            for command in tqdm(command_spectrum, desc="Building Lookup Table"):
                # if np.linalg.norm(command) < 1e-6:
                #     continue

                leg_joint_pos_ref, stance_mask_ref = self.plan(
                    path_pos, path_quat, command
                )
                lookup_keys.append(tuple(map(float, command)))
                stance_mask_ref_list.append(stance_mask_ref)
                leg_joint_pos_ref_list.append(leg_joint_pos_ref)

            with open(self.lookup_table_path, "wb") as f:
                pickle.dump(
                    (lookup_keys, stance_mask_ref_list, leg_joint_pos_ref_list), f
                )

        self.lookup_keys = np.array(lookup_keys, dtype=np.float32)
        self.lookup_length = np.array(
            [len(stance_mask_ref) for stance_mask_ref in stance_mask_ref_list],
            dtype=np.float32,
        )

        num_commands = len(stance_mask_ref_list)
        num_total_steps_max = max(
            [len(stance_mask_ref) for stance_mask_ref in stance_mask_ref_list]
        )
        self.stance_mask_lookup = np.zeros(
            (num_commands, num_total_steps_max, 2), dtype=np.float32
        )
        self.leg_joint_pos_lookup = np.zeros(
            (num_commands, num_total_steps_max, 12), dtype=np.float32
        )
        for i, (stance_mask_ref, leg_joint_pos_ref) in enumerate(
            zip(stance_mask_ref_list, leg_joint_pos_ref_list)
        ):
            self.stance_mask_lookup = inplace_update(
                self.stance_mask_lookup,
                (i, slice(None, len(stance_mask_ref))),
                stance_mask_ref,
            )
            self.leg_joint_pos_lookup = inplace_update(
                self.leg_joint_pos_lookup,
                (i, slice(None, len(leg_joint_pos_ref))),
                leg_joint_pos_ref,
            )

        if os.environ.get("USE_JAX", "false") == "true":
            self.lookup_keys = jax.device_put(self.lookup_keys)
            self.lookup_length = jax.device_put(self.lookup_length)
            self.stance_mask_lookup = jax.device_put(self.stance_mask_lookup)
            self.leg_joint_pos_lookup = jax.device_put(self.leg_joint_pos_lookup)

    def plan(
        self,
        path_pos: ArrayType,
        path_quat: ArrayType,
        command: ArrayType,
        total_time: float = 20.0,
    ) -> Tuple[ArrayType, ArrayType]:
        path_euler = quat2euler(path_quat)
        pose_curr = np.array(
            [path_pos[0], path_pos[1], path_euler[2]], dtype=np.float32
        )

        num_cycles = int(np.ceil(total_time / self.cycle_time))
        footsteps: List[ArrayType] = []
        left_footstep_init = np.array(
            [pose_curr[0], pose_curr[1] + self.foot_to_com_y, pose_curr[2], 0],
            dtype=np.float32,
        )
        right_footstep_init = np.array(
            [pose_curr[0], pose_curr[1] - self.foot_to_com_y, pose_curr[2], 1],
            dtype=np.float32,
        )
        if np.linalg.norm(command) < 1e-6:
            for _ in range(num_cycles):
                footsteps.append(left_footstep_init)
                footsteps.append(right_footstep_init)

        elif np.abs(command[0]) > 1e-6:
            stride = command[0] * total_time / (2 * num_cycles - 1)
            for i in range(num_cycles):
                left_footstep = left_footstep_init.copy()
                right_footstep = right_footstep_init.copy()
                left_footstep[0] += i * 2 * stride
                right_footstep[0] += (i * 2 + 1) * stride
                footsteps.append(left_footstep)
                footsteps.append(right_footstep)

        elif np.abs(command[1]) > 1e-6:
            stride = command[1] * total_time / (2 * num_cycles - 1)
            for i in range(num_cycles):
                left_footstep = left_footstep_init.copy()
                right_footstep = right_footstep_init.copy()
                left_footstep[1] += i * 2 * stride
                right_footstep[1] += (i * 2 + 1) * stride
                footsteps.append(left_footstep)
                footsteps.append(right_footstep)

        elif np.abs(command[2]) > 1e-6:
            raise NotImplementedError("Rotation not implemented")

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
        desired_zmps = [step[:2] for step in footsteps for _ in range(2)]

        x0 = np.array([path_pos[0], path_pos[1], 0.0, 0.0], dtype=np.float32)

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
        num_total_steps = int(
            np.ceil((time_steps[-1] - time_steps[0]) / self.control_dt)
        )
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
            stance_mask_ref,
        ) = self.compute_foot_trajectories(
            time_steps,
            np.repeat(np.stack(footsteps), 2, axis=0),
        )

        leg_joint_pos_ref = self.solve_ik(
            left_foot_pos_traj,
            left_foot_ori_traj,
            right_foot_pos_traj,
            right_foot_ori_traj,
            x_traj[:, :2],
        )

        first_cycle_idx = int(np.ceil(self.cycle_time / self.control_dt))
        leg_joint_pos_ref_truncated = leg_joint_pos_ref[first_cycle_idx:]
        stance_mask_ref_truncated = stance_mask_ref[first_cycle_idx:]

        return leg_joint_pos_ref_truncated, stance_mask_ref_truncated

    def compute_foot_trajectories(
        self, time_steps: ArrayType, footsteps: List[ArrayType]
    ) -> Tuple[ArrayType, ...]:
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

        num_total_steps = int(
            np.ceil((time_steps[-1] - time_steps[0]) / self.control_dt)
        )
        left_foot_pos_traj = np.zeros((num_total_steps, 3), dtype=np.float32)
        left_foot_ori_traj = np.zeros((num_total_steps, 3), dtype=np.float32)
        right_foot_pos_traj = np.zeros((num_total_steps, 3), dtype=np.float32)
        right_foot_ori_traj = np.zeros((num_total_steps, 3), dtype=np.float32)
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

                # TODO: check this
                ori_delta = (target_ori - current_ori) / num_steps
                foot_ori_traj = current_ori + ori_delta * np.arange(num_steps)[:, None]
                foot_ori_traj = inplace_update(
                    foot_ori_traj,
                    (slice(None), swing_leg * 3 + 2),
                    np.zeros(num_steps, dtype=np.float32),
                )
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
            stance_mask_traj = inplace_update(stance_mask_traj, slice_curr, stance_mask)

            step_curr += int(num_steps)

        return (
            left_foot_pos_traj,
            left_foot_ori_traj,
            right_foot_pos_traj,
            right_foot_ori_traj,
            stance_mask_traj,
        )

    def solve_ik(
        self,
        left_foot_pos_traj: ArrayType,
        left_foot_ori_traj: ArrayType,
        right_foot_pos_traj: ArrayType,
        right_foot_ori_traj: ArrayType,
        com_pos_traj: ArrayType,
    ):
        com_pos_traj_padded = np.hstack(
            [com_pos_traj, np.zeros((com_pos_traj.shape[0], 1))]
        )
        left_foot_adjusted_pos = (
            left_foot_pos_traj
            - com_pos_traj_padded
            - np.array([self.foot_to_com_x, self.foot_to_com_y, 0], dtype=np.float32)
        )
        right_foot_adjusted_pos = (
            right_foot_pos_traj
            - com_pos_traj_padded
            - np.array([self.foot_to_com_x, -self.foot_to_com_y, 0], dtype=np.float32)
        )

        left_leg_joint_pos_traj = self.foot_ik(
            left_foot_adjusted_pos,
            left_foot_ori_traj,
            side="left",
        )
        right_leg_joint_pos_traj = self.foot_ik(
            right_foot_adjusted_pos,
            right_foot_ori_traj,
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
        target_foot_ori: ArrayType = np.zeros(3, dtype=np.float32),
        side: str = "left",
    ) -> ArrayType:
        target_x = target_foot_pos[:, 0]
        target_y = target_foot_pos[:, 1]
        target_z = target_foot_pos[:, 2]
        ank_roll = target_foot_ori[:, 0]
        ank_pitch = target_foot_ori[:, 1]
        hip_yaw = target_foot_ori[:, 2]

        offsets = self.robot.data_dict["offsets"]

        transformed_x = target_x * np.cos(hip_yaw) + target_y * np.sin(hip_yaw)
        transformed_y = -target_x * np.sin(hip_yaw) + target_y * np.cos(hip_yaw)
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
        hip_pitch = -leg_pitch - hip_disp
        knee_pitch = hip_disp + ank_disp
        ank_pitch += knee_pitch + hip_pitch

        if side == "left":
            return np.vstack(
                [
                    -hip_yaw,
                    -hip_roll,
                    hip_pitch,
                    knee_pitch,
                    -ank_roll + hip_roll,
                    -ank_pitch,
                ]
            ).T
        else:
            return np.vstack(
                [
                    -hip_yaw,
                    hip_roll,
                    -hip_pitch,
                    -knee_pitch,
                    -ank_roll + hip_roll,
                    -ank_pitch,
                ]
            ).T
