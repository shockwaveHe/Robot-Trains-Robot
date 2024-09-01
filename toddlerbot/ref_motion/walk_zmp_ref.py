from typing import List, Optional, Tuple

from toddlerbot.algorithms.zmp.footstep_planner import FootStepPlanner
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
        default_joint_pos: Optional[ArrayType] = None,
        default_joint_vel: Optional[ArrayType] = None,
        plan_max_stride: List[float] = [0.12, 0.05, np.pi / 8],
        single_double_ratio: float = 2.0,
        foot_step_height: float = 0.03,
        control_dt: float = 0.012,
        control_cost_Q: float = 1.0,
        control_cost_R: float = 0.1,
    ):
        super().__init__("walk_zmp", "periodic", robot)

        self.default_joint_pos = default_joint_pos
        self.default_joint_vel = default_joint_vel
        self.single_double_ratio = single_double_ratio
        self.footstep_height = foot_step_height
        self.control_dt = control_dt
        self.control_cost_Q = control_cost_Q
        self.control_cost_R = control_cost_R

        # TODO: Read from config
        self.com_z = 0.336
        self.foot_to_com_x = float(robot.data_dict["offsets"]["foot_to_com_x"])
        self.foot_to_com_y = float(robot.data_dict["offsets"]["foot_to_com_y"])

        if self.default_joint_pos is None:
            self.default_target_z = 0.0
        else:
            self.default_target_z = (
                float(robot.config["general"]["offsets"]["torso_z"]) - self.com_z
            )

        self.leg_joint_slice = slice(
            self.get_joint_idx("left_hip_yaw_driven"),
            self.get_joint_idx("right_ank_pitch") + 1,
        )

        self.fsp = FootStepPlanner(
            np.array(plan_max_stride, dtype=np.float32),  # type: ignore
            self.foot_to_com_y,
        )
        self.zmp_planner = ZMPPlanner()

        self.reset()

    def reset(self):
        self.planned = False

    # @profile()
    def get_state_ref(
        self,
        path_pos: ArrayType,
        path_quat: ArrayType,
        phase: Optional[float | ArrayType] = None,
        command: Optional[ArrayType] = None,
        duration: Optional[float] = None,
    ) -> ArrayType:
        if phase is None:
            raise ValueError(f"phase is required for {self.name}")

        if command is None:
            raise ValueError(f"command is required for {self.name}")

        if duration is None:
            raise ValueError(f"duration is required for {self.name}")

        linear_vel = np.array([command[0], command[1], 0.0], dtype=np.float32)  # type: ignore
        angular_vel = np.array([0.0, 0.0, command[2]], dtype=np.float32)  # type: ignore

        if self.default_joint_vel is None:
            joint_vel = np.zeros(len(self.robot.joint_ordering), dtype=np.float32)  # type: ignore
        else:
            joint_vel = self.default_joint_vel.copy()  # type: ignore

        joint_pos = self.default_joint_pos.copy()  # type: ignore
        command_scale = np.linalg.norm(command)  # type: ignore
        if float(command_scale) < 1e-6:  # type: ignore
            stance_mask = np.ones(2, dtype=np.float32)  # type: ignore
        else:
            if not self.planned:
                self.plan(path_pos, path_quat, command, duration)

            idx = int(phase / self.control_dt)
            joint_pos = inplace_update(
                joint_pos,
                self.leg_joint_slice,
                self.leg_joint_pos_ref[idx],
            )
            stance_mask = self.stance_mask_ref[idx]

        return np.concatenate(  # type: ignore
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

    def plan(
        self,
        path_pos: ArrayType,
        path_quat: ArrayType,
        command: ArrayType,
        duration: float,
    ):
        path_euler = quat2euler(path_quat)
        pose_curr = np.array(  # type: ignore
            [path_pos[0], path_pos[1], path_euler[2]], dtype=np.float32
        )
        target_pose = self.integrate_motion(pose_curr, command, duration)
        path, footsteps = self.fsp.compute_steps(
            pose_curr, target_pose, has_start=False, has_stop=False
        )

        import numpy

        from toddlerbot.visualization.vis_plot import plot_footsteps

        # You can plot the footsteps with your existing plotting utility here
        plot_footsteps(
            numpy.asarray(path, dtype=numpy.float32),
            numpy.array(
                [numpy.asarray(fs[:3]) for fs in footsteps], dtype=numpy.float32
            ),
            [int(fs[-1]) for fs in footsteps],
            (0.1, 0.05),
            self.foot_to_com_y,
            fig_size=(8, 8),
            title=f"Footsteps Planning: {target_pose[0]:.2f} {target_pose[1]:.2f} {target_pose[2]:.2f}",
            x_label="Position X",
            y_label="Position Y",
            save_config=False,
            save_path=".",
            file_name="footsteps.png",
        )()

        double_support_phase = duration / (
            (len(footsteps) - 1) * (1 + self.single_double_ratio) + 1
        )
        single_support_phase = double_support_phase * self.single_double_ratio
        time_list = np.array(  # type: ignore
            [0, double_support_phase]
            + [single_support_phase, double_support_phase] * (len(footsteps) - 1),
            dtype=np.float32,
        )
        time_steps = np.cumsum(time_list)  # type: ignore
        desired_zmps = [step[:2] for step in footsteps for _ in range(2)]

        x0 = np.array(  # type: ignore
            [path_pos[0], path_pos[1], 0.0, 0.0], dtype=np.float32
        )
        self.zmp_planner.plan(
            time_steps,
            desired_zmps,
            x0,
            self.com_z,
            Qy=np.eye(2, dtype=np.float32) * self.control_cost_Q,  # type: ignore
            R=np.eye(2, dtype=np.float32) * self.control_cost_R,  # type: ignore
        )

        def update_step(
            carry: Tuple[ArrayType, ArrayType], idx: int
        ) -> Tuple[Tuple[ArrayType, ArrayType], ArrayType]:
            x_traj, u_traj = carry
            t = time_steps[0] + idx * self.control_dt
            xd = np.hstack((x_traj[idx - 1, 2:], u_traj[idx - 1, :]))  # type: ignore
            x_traj = inplace_update(
                x_traj, idx, x_traj[idx - 1, :] + xd * self.control_dt
            )
            u = self.zmp_planner.get_optim_com_acc(t, x_traj[idx, :])
            u_traj = inplace_update(u_traj, idx, u)

            return (x_traj, u_traj), x_traj[idx]

        # Initialize the arrays
        N = int(np.ceil((time_steps[-1] - time_steps[0]) / self.control_dt))  # type: ignore
        x_traj = np.zeros((N, 4), dtype=np.float32)  # type: ignore
        u_traj = np.zeros((N, 2), dtype=np.float32)  # type: ignore
        # Set the initial conditions
        x_traj = inplace_update(x_traj, 0, x0)
        # u_traj = inplace_update(
        #     u_traj,
        #     0,
        #     self.zmp_planner.get_optim_com_acc(time_steps[0], x0),  # type: ignore
        # )
        x_traj = loop_update(update_step, x_traj, u_traj, (1, N))

        (
            left_foot_pos_traj,
            left_foot_ori_traj,
            right_foot_pos_traj,
            right_foot_ori_traj,
            self.stance_mask_ref,
        ) = self.compute_foot_trajectories(time_steps, np.repeat(footsteps, 2, axis=0))  # type: ignore

        self.leg_joint_pos_ref = self.solve_ik(
            left_foot_pos_traj,
            left_foot_ori_traj,
            right_foot_pos_traj,
            right_foot_ori_traj,
            x_traj[:, :2],
        )

        self.planned = True

    def integrate_motion(
        self,
        pose_curr: ArrayType,
        command: ArrayType,
        duration: float,
    ) -> ArrayType:
        # Linear velocities in local frame
        v_x, v_y, v_yaw = command

        # If angular velocity is not zero, handle the integration considering the changing yaw
        if v_yaw != 0:
            yaw_change = v_yaw * duration
            final_yaw = pose_curr[2] + yaw_change

            # Calculate the integrals for x and y considering the changing yaw
            integrated_x = (v_x / v_yaw) * (
                np.sin(final_yaw) - np.sin(pose_curr[2])  # type: ignore
            ) + (v_y / v_yaw) * (np.cos(final_yaw) - np.cos(pose_curr[2]))  # type: ignore
            integrated_y = -(v_x / v_yaw) * (
                np.cos(final_yaw) - np.cos(pose_curr[2])  # type: ignore
            ) + (v_y / v_yaw) * (np.sin(final_yaw) - np.sin(pose_curr[2]))  # type: ignore
        else:
            # If angular velocity is zero, it's a simple linear motion with no rotation
            integrated_x = v_x * duration * np.cos(  # type: ignore
                pose_curr[2]
            ) - v_y * duration * np.sin(pose_curr[2])  # type: ignore
            integrated_y = v_x * duration * np.sin(  # type: ignore
                pose_curr[2]
            ) + v_y * duration * np.cos(pose_curr[2])  # type: ignore

        # Return final pose
        final_pose = np.array(  # type: ignore
            [
                pose_curr[0] + integrated_x,
                pose_curr[1] + integrated_y,
                pose_curr[2] + v_yaw * duration,
            ],
            dtype=np.float32,
        )

        return final_pose

    def compute_foot_trajectories(
        self, time_steps: List[float], footsteps: List[ArrayType]
    ) -> Tuple[ArrayType, ...]:
        offset = np.array(  # type: ignore
            [
                -np.sin(footsteps[0][2]) * self.foot_to_com_y,  # type: ignore
                np.cos(footsteps[0][2]) * self.foot_to_com_y,  # type: ignore
            ]
        )
        last_pos = np.concatenate(  # type: ignore
            [
                footsteps[0][:2] + offset,
                np.zeros(1, dtype=np.float32),  # type: ignore
                footsteps[0][:2] - offset,
                np.zeros(1, dtype=np.float32),  # type: ignore
            ]
        )
        last_ori = np.array(  # type: ignore
            [0.0, 0.0, footsteps[0][2], 0.0, 0.0, footsteps[0][2]], dtype=np.float32
        )

        N = int(np.ceil((time_steps[-1] - time_steps[0]) / self.control_dt))  # type: ignore
        left_foot_pos_traj = np.zeros((N, 3), dtype=np.float32)  # type: ignore
        left_foot_ori_traj = np.zeros((N, 3), dtype=np.float32)  # type: ignore
        right_foot_pos_traj = np.zeros((N, 3), dtype=np.float32)  # type: ignore
        right_foot_ori_traj = np.zeros((N, 3), dtype=np.float32)  # type: ignore
        stance_mask_traj = np.zeros((N, 2), dtype=np.float32)  # type: ignore
        step_curr = 0
        for i in range(len(time_steps) - 1):
            if i == len(time_steps) - 2:
                # For the last interval, ensure the total number of steps is exactly N
                num_steps = N - step_curr
            else:
                num_steps = round((time_steps[i + 1] - time_steps[i]) / self.control_dt)

            stance_mask = np.tile(np.ones(2, dtype=np.float32), (num_steps, 1))  # type: ignore
            if i % 2 == 0:  # Double support
                foot_pos_traj = np.tile(last_pos, (num_steps, 1))  # type: ignore
                foot_ori_traj = np.tile(last_ori, (num_steps, 1))  # type: ignore
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
                    offset = np.array(  # type: ignore
                        [
                            -np.sin(footsteps[i][2]) * self.foot_to_com_y,
                            np.cos(footsteps[i][2]) * self.foot_to_com_y,  # type: ignore
                        ]
                    ) * (-1 if support_leg_curr == 1 else 1)
                else:
                    offset = np.zeros(2, dtype=np.float32)  # type: ignore

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
                up_traj = up_delta * np.concatenate(  # type: ignore
                    (
                        np.arange(num_steps // 2, dtype=np.float32),
                        np.arange(  # type: ignore
                            num_steps - num_steps // 2 - 1, -1, -1, dtype=np.float32
                        ),
                    )
                )
                pos_delta = (target_pos - current_pos) / num_steps
                foot_pos_traj = current_pos + pos_delta * np.arange(num_steps)[:, None]  # type: ignore
                foot_pos_traj = inplace_update(
                    foot_pos_traj, (slice(None), swing_leg * 3 + 2), up_traj
                )

                ori_delta = (target_ori - current_ori) / num_steps
                foot_ori_traj = current_ori + ori_delta * np.arange(num_steps)[:, None]  # type: ignore

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

            step_curr += num_steps

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
        com_pos_traj_padded = np.hstack(  # type: ignore
            [com_pos_traj, np.zeros((com_pos_traj.shape[0], 1))]  # type: ignore
        )
        left_foot_adjusted_pos = (
            left_foot_pos_traj
            - com_pos_traj_padded
            - np.array([self.foot_to_com_x, self.foot_to_com_y, 0], dtype=np.float32)  # type: ignore
        )
        right_foot_adjusted_pos = (
            right_foot_pos_traj
            - com_pos_traj_padded
            - np.array([self.foot_to_com_x, -self.foot_to_com_y, 0], dtype=np.float32)  # type: ignore
        )

        left_leg_joint_pos_traj = self.foot_ik(
            left_foot_adjusted_pos,  # type: ignore
            left_foot_ori_traj,
            side="left",
        )
        right_leg_joint_pos_traj = self.foot_ik(
            right_foot_adjusted_pos,  # type: ignore
            right_foot_ori_traj,
            side="right",
        )

        # Combine the results for left and right legs
        leg_joint_pos_traj = np.hstack(  # type: ignore
            [left_leg_joint_pos_traj, right_leg_joint_pos_traj]
        )

        return leg_joint_pos_traj

    def foot_ik(
        self,
        target_foot_pos: ArrayType,
        target_foot_ori: ArrayType = np.zeros(3, dtype=np.float32),  # type: ignore
        side: str = "left",
    ) -> ArrayType:
        target_x = target_foot_pos[:, 0]
        target_y = target_foot_pos[:, 1]
        target_z = target_foot_pos[:, 2]
        ank_roll = target_foot_ori[:, 0]
        ank_pitch = target_foot_ori[:, 1]
        hip_yaw = target_foot_ori[:, 2]

        offsets = self.robot.data_dict["offsets"]

        transformed_x = target_x * np.cos(hip_yaw) + target_y * np.sin(hip_yaw)  # type: ignore
        transformed_y = -target_x * np.sin(hip_yaw) + target_y * np.cos(hip_yaw)  # type: ignore
        transformed_z = (
            offsets["hip_pitch_to_knee_z"]
            + offsets["knee_to_ank_pitch_z"]
            - target_z
            - self.default_target_z
        )

        hip_roll = np.arctan2(  # type: ignore
            transformed_y, transformed_z + offsets["hip_roll_to_pitch_z"]
        )

        leg_projected_yz_length = np.sqrt(transformed_y**2 + transformed_z**2)  # type: ignore
        leg_length = np.sqrt(transformed_x**2 + leg_projected_yz_length**2)  # type: ignore
        leg_pitch = np.arctan2(transformed_x, leg_projected_yz_length)  # type: ignore
        hip_disp_cos = (
            leg_length**2
            + offsets["hip_pitch_to_knee_z"] ** 2
            - offsets["knee_to_ank_pitch_z"] ** 2
        ) / (2 * leg_length * offsets["hip_pitch_to_knee_z"])
        hip_disp = np.arccos(np.clip(hip_disp_cos, -1.0, 1.0))  # type: ignore
        ank_disp = np.arcsin(  # type: ignore
            offsets["hip_pitch_to_knee_z"]
            / offsets["knee_to_ank_pitch_z"]
            * np.sin(hip_disp)  # type: ignore
        )
        hip_pitch = -leg_pitch - hip_disp
        knee_pitch = hip_disp + ank_disp
        ank_pitch += knee_pitch + hip_pitch

        if side == "left":
            return np.vstack(  # type: ignore
                [
                    hip_yaw,
                    -hip_roll,
                    hip_pitch,
                    knee_pitch,
                    -ank_roll + hip_roll,
                    -ank_pitch,
                ]
            ).T
        else:
            return np.vstack(  # type: ignore
                [
                    hip_yaw,
                    hip_roll,
                    -hip_pitch,
                    -knee_pitch,  # type: ignore
                    -ank_roll + hip_roll,
                    -ank_pitch,
                ]  # type: ignore
            ).T
