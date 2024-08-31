from typing import List, Optional, Tuple

from toddlerbot.algorithms.lipm.lipm_3d import LIPM3DPlanner
from toddlerbot.algorithms.zmp.footstep_planner import FootStepPlanner
from toddlerbot.algorithms.zmp.zmp_planner import ZMPPlanner
from toddlerbot.motion_reference.motion_ref import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update, loop_update
from toddlerbot.utils.array_utils import array_lib as np
from toddlerbot.utils.math_utils import quat2euler, resample_trajectory


class WalkLIPMReference(MotionReference):
    def __init__(
        self,
        robot: Robot,
        default_joint_pos: Optional[ArrayType] = None,
        default_joint_vel: Optional[ArrayType] = None,
        max_knee_pitch: float = np.pi / 3,
        double_support_phase: float = 0.5,
        single_support_phase: float = 1.0,
        plan_max_stride: List[float] = [0.1, 0.05, np.pi / 8],
    ):
        super().__init__("walk", "periodic", robot)

        self.default_joint_pos = default_joint_pos
        self.default_joint_vel = default_joint_vel
        if self.default_joint_pos is None:
            self.knee_pitch_default = 0.0
        else:
            self.knee_pitch_default = self.default_joint_pos[
                self.get_joint_idx("left_knee_pitch")
            ]

        self.max_knee_pitch = max_knee_pitch
        self.double_support_phase = double_support_phase
        self.single_support_phase = single_support_phase
        self.num_joints = len(self.robot.joint_ordering)
        self.shin_thigh_ratio = (
            self.robot.data_dict["offsets"]["knee_to_ank_pitch_z"]
            / self.robot.data_dict["offsets"]["hip_pitch_to_knee_z"]
        )
        self.com_z = self.robot.config["general"]["offsets"]["torso_z"]

        self.zmp_planner = ZMPPlanner()

        self.control_dt = 0.01
        self.control_cost_Q = 1.0
        self.control_cost_R = 0.1
        self.footstep_height = 0.03
        self.foot_to_com_x = robot.data_dict["offsets"]["foot_to_com_x"]
        self.foot_to_com_y = robot.data_dict["offsets"]["foot_to_com_y"]

        self.init_joint_pos = np.array(  # type: ignore
            list(robot.init_joint_angles.values()), dtype=np.float32
        )

        self.fsp = FootStepPlanner(
            np.array(plan_max_stride, dtype=np.float32),  # type: ignore
            self.foot_to_com_y,
        )
        self.zmp_planner = ZMPPlanner()

        self.curr_pose = None
        self.idx = 0

        self.foot_steps = []
        self.com_ref_traj = []

        self.left_up = self.right_up = 0.0
        # Assume the initial state is the canonical pose
        self.theta_curr = 0.0
        self.left_pos = np.array([0.0, self.foot_to_com_y, 0.0], dtype=np.float32)  # type: ignore
        self.right_pos = np.array([0.0, -self.foot_to_com_y, 0.0], dtype=np.float32)  # type: ignore

    def reset(self):
        pass

    def plan(
        self,
        path_pos: ArrayType,
        path_quat: ArrayType,
        phase: float,
        command: ArrayType,
        duration: float,
    ):
        COM_pos_x, COM_pos_y = [], []
        COM_vel_x, COM_vel_y = [], []
        COM_dvel_x, COM_dvel_y = [], []
        left_foot_pos_x, left_foot_pos_y, left_foot_pos_z = [], [], []
        right_foot_pos_x, right_foot_pos_y, right_foot_pos_z = [], [], []
        step_length, dstep_length, step_width, dstep_width = [], [], [], []

        swing_data_len = int(LIPM_model.T / LIPM_model.dt)
        swing_foot_pos = np.zeros((swing_data_len, 3))
        j, step_num = 0, 0
        theta, COM_dvel = 0.0, COM_dvel_list[0]

        support_foot_pos = np.array(LIPM_model.support_foot_pos)
        prev_support_foot_pos = np.array(LIPM_model.support_foot_pos)

        for i in range(1, int(total_time / LIPM_model.dt)):
            LIPM_model.step()

            update_com_data(
                LIPM_model,
                COM_pos_x,
                COM_pos_y,
                COM_vel_x,
                COM_vel_y,
                COM_dvel_x,
                COM_dvel_y,
                COM_dvel,
            )

            update_foot_positions(
                LIPM_model,
                swing_foot_pos,
                j,
                left_foot_pos_x,
                left_foot_pos_y,
                left_foot_pos_z,
                right_foot_pos_x,
                right_foot_pos_y,
                right_foot_pos_z,
            )
            j += 1

            if i % swing_data_len == 0:
                j = 0
                prev_support_foot_pos = support_foot_pos
                step_num, theta, COM_dvel = update_step_and_switch_leg(
                    LIPM_model, step_num, step_to_cmdv, COM_dvel_list, w_d_list, theta
                )
                support_foot_pos = update_step_data(
                    LIPM_model,
                    prev_support_foot_pos,
                    np.array(LIPM_model.support_foot_pos),
                    step_length,
                    dstep_length,
                    step_width,
                    dstep_width,
                    theta,
                )

                LIPM_model.calculate_foot_location_world(theta)
                swing_foot_pos = calculate_swing_foot_positions(
                    swing_data_len,
                    LIPM_model.right_foot_pos
                    if LIPM_model.support_leg == "left_leg"
                    else LIPM_model.left_foot_pos,
                    [LIPM_model.u_x, LIPM_model.u_y, 0],
                )

        if not np.any(command):  # type: ignore
            raise ValueError("command cannot be all zeros")

        path_euler = quat2euler(path_quat)
        curr_pose = np.array(  # type: ignore
            [path_pos[0], path_pos[1], path_euler[2]], dtype=np.float32
        )
        target_pose = self.integrate_motion(curr_pose, path_euler[2], command, duration)
        path, footsteps = self.fsp.compute_steps(curr_pose, target_pose)

        # import numpy

        # from toddlerbot.visualization.vis_plot import plot_footsteps

        # # You can plot the footsteps with your existing plotting utility here
        # plot_footsteps(
        #     numpy.asarray(path, dtype=numpy.float32),
        #     numpy.array(
        #         [numpy.asarray(fs[:3]) for fs in footsteps], dtype=numpy.float32
        #     ),
        #     [int(fs[-1]) for fs in footsteps],
        #     (0.1, 0.05),
        #     self.foot_to_com_y,
        #     fig_size=(8, 8),
        #     title=f"Footsteps Planning: {target_pose[0]:.2f} {target_pose[1]:.2f} {target_pose[2]:.2f}",
        #     x_label="Position X",
        #     y_label="Position Y",
        #     save_config=False,
        #     save_path=".",
        #     file_name="footsteps.png",
        # )()

        time_list = np.array(  # type: ignore
            [0, self.double_support_phase]
            + [
                self.single_support_phase,
                self.double_support_phase,
            ]
            * (len(footsteps) - 1),
            dtype=np.float32,
        )
        time_steps = np.cumsum(time_list)  # type: ignore
        desired_zmps = [step[:2] for step in footsteps for _ in range(2)]

        self.zmp_planner.plan(
            time_steps,
            desired_zmps,
            np.array([0, 0, 0, 0], dtype=np.float32),  # type: ignore
            self.com_z,
            Qy=np.eye(2, dtype=np.float32) * self.control_cost_Q,  # type: ignore
            R=np.eye(2, dtype=np.float32) * self.control_cost_R,  # type: ignore
        )

        # TODO: update x0
        x0 = np.array([path_pos[0], path_pos[1], command[0], command[1]])  # type: ignore
        N = int((time_steps[-1] - time_steps[0]) / self.control_dt)

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
        x_traj = np.zeros((N, 4), dtype=np.float32)  # type: ignore
        u_traj = np.zeros((N, 2), dtype=np.float32)  # type: ignore
        # Set the initial conditions
        x_traj = inplace_update(x_traj, 0, x0)
        u_traj = inplace_update(
            u_traj,
            0,
            self.zmp_planner.get_optim_com_acc(time_steps[0], x0),  # type: ignore
        )
        x_traj = loop_update(update_step, x_traj, u_traj, (1, N))

        feet_pose_traj = self.get_feet_pose(time_steps, np.repeat(footsteps, 2, axis=0))  # type: ignore

        joint_pos_traj = self.solve_ik(feet_pose_traj, x_traj)

        self.joint_angles_list.append((t + 0.5, self.init_joint_pos))

        joint_angles_list = resample_trajectory(
            self.joint_angles_list,
            desired_interval=self.config.control_dt,
            interp_type="cubic",
        )

        return (
            path,
            foot_steps_copy,
            zmp_ref_traj,
            zmp_traj,
            com_ref_traj_copy,
            joint_angles_list,
        )

    def integrate_motion(
        self,
        curr_pose: ArrayType,
        initial_yaw: ArrayType,
        command: ArrayType,
        duration: float,
    ):
        # Linear velocities in local frame
        v_x, v_y, angular_velocity = command

        # If angular velocity is not zero, handle the integration considering the changing yaw
        if angular_velocity != 0:
            yaw_change = angular_velocity * duration
            final_yaw = initial_yaw + yaw_change

            # Calculate the integrals for x and y considering the changing yaw
            integrated_x = (v_x / angular_velocity) * (
                np.sin(final_yaw) - np.sin(initial_yaw)  # type: ignore
            ) + (v_y / angular_velocity) * (np.cos(final_yaw) - np.cos(initial_yaw))  # type: ignore
            integrated_y = -(v_x / angular_velocity) * (
                np.cos(final_yaw) - np.cos(initial_yaw)  # type: ignore
            ) + (v_y / angular_velocity) * (np.sin(final_yaw) - np.sin(initial_yaw))  # type: ignore
        else:
            # If angular velocity is zero, it's a simple linear motion with no rotation
            integrated_x = v_x * duration * np.cos(  # type: ignore
                initial_yaw
            ) - v_y * duration * np.sin(initial_yaw)  # type: ignore
            integrated_y = v_x * duration * np.sin(  # type: ignore
                initial_yaw
            ) + v_y * duration * np.cos(initial_yaw)  # type: ignore

        # Compute final position and yaw
        final_position = np.array(  # type: ignore
            [curr_pose[0] + integrated_x, curr_pose[1] + integrated_y]
        )
        final_yaw = initial_yaw + angular_velocity * duration

        # Return final pose
        final_pose = np.array(  # type: ignore
            [final_position[0], final_position[1], final_yaw], dtype=np.float32
        )
        return final_pose

    def get_feet_pose(
        self,
        time_steps: ArrayType,
        footsteps: ArrayType,
    ):
        feet_pose_list: List[ArrayType] = []
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

        for i in range(len(time_steps) - 1):
            num_steps = int((time_steps[i + 1] - time_steps[i]) / self.control_dt)
            if i % 2 == 0:  # Double support
                feet_pose_traj = np.tile(  # type: ignore
                    np.concatenate([last_pos, last_ori], axis=-1),  # type: ignore
                    (num_steps, 1),
                )
            else:
                support_leg_curr = int(footsteps[i][-1])
                support_leg_next = int(footsteps[i + 1][-1])
                if support_leg_curr == 2:
                    current_pos = last_pos.copy()
                    current_ori = last_ori.copy()
                    target_leg_next = support_leg_next
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
                    target_leg_next = 1 - support_leg_curr

                if support_leg_next == 2:
                    offset = np.array(  # type: ignore
                        [
                            -np.sin(footsteps[i][2]) * self.foot_to_com_y,  # type: ignore
                            np.cos(footsteps[i][2]) * self.foot_to_com_y,  # type: ignore
                        ]
                    ) * (-1 if support_leg_curr == 1 else 1)
                else:
                    offset = np.zeros(2, dtype=np.float32)  # type: ignore

                target_pos = inplace_update(
                    current_pos,
                    slice(target_leg_next * 3, target_leg_next * 3 + 2),
                    footsteps[i + 1][:2] + offset,
                )
                target_ori = inplace_update(
                    current_ori, target_leg_next * 3 + 2, footsteps[i + 1][2]
                )
                last_pos = target_pos.copy()
                last_ori = target_ori.copy()

                up_delta = self.footstep_height / (num_steps // 2 - 1)
                up_traj = up_delta * np.concatenate(  # type: ignore
                    (
                        np.arange(num_steps // 2, dtype=np.float32),  # type: ignore
                        np.arange(num_steps // 2 - 1, -1, -1, dtype=np.float32),  # type: ignore
                    )
                )
                pos_delta = (target_pos - current_pos) / num_steps
                pos_traj = current_pos + pos_delta * np.arange(num_steps)[:, None]  # type: ignore
                pos_traj = inplace_update(
                    pos_traj, (slice(None), target_leg_next * 3 + 2), up_traj
                )

                ori_delta = (target_ori - current_ori) / num_steps
                ori_traj = current_ori + ori_delta * np.arange(num_steps)[:, None]  # type: ignore

                feet_pose_traj = np.concatenate([pos_traj, ori_traj], axis=-1)  # type: ignore

            feet_pose_list.append(feet_pose_traj)

        feet_pose_traj_all: ArrayType = np.concatenate(feet_pose_list, axis=0)  # type: ignore

        return feet_pose_traj_all

    def solve_ik(self, footsteps: List[ArrayType], x_traj: ArrayType):
        joint_pos_traj = self.robot.solve_ik(footsteps, x_traj)

        return joint_pos_traj

    def get_state_ref(
        self,
        path_pos: ArrayType,
        path_quat: ArrayType,
        phase: Optional[float | ArrayType] = None,
        command: Optional[ArrayType] = None,
    ) -> ArrayType:
        if phase is None:
            raise ValueError(f"phase is required for {self.motion_type} motion")

        if command is None:
            raise ValueError(f"command is required for {self.motion_type} motion")

        linear_vel = np.array([command[0], command[1], 0.0], dtype=np.float32)
        angular_vel = np.array([0.0, 0.0, command[2]], dtype=np.float32)

        sin_phase_signal = np.sin(2 * np.pi * phase)
        signal_left = np.clip(sin_phase_signal, 0, None)
        signal_right = np.clip(sin_phase_signal, None, 0)

        if self.default_joint_pos is None:
            joint_pos = np.zeros(self.num_joints, dtype=np.float32)
        else:
            joint_pos = self.default_joint_pos.copy()

        if self.default_joint_vel is None:
            joint_vel = np.zeros(self.num_joints, dtype=np.float32)
        else:
            joint_vel = self.default_joint_vel.copy()

        left_leg_angles = self.calculate_leg_angles(signal_left, True)
        right_leg_angles = self.calculate_leg_angles(signal_right, False)

        leg_angles = {**left_leg_angles, **right_leg_angles}

        if self.use_jax:
            indices = np.array([self.get_joint_idx(name) for name in leg_angles])
            angles = np.array(list(leg_angles.values()))
            joint_pos = joint_pos.at[indices].set(angles)
        else:
            for name, angle in leg_angles.items():
                joint_pos[self.get_joint_idx(name)] = angle

        double_support_mask = np.abs(sin_phase_signal) < self.double_support_phase
        joint_pos = np.where(double_support_mask, self.default_joint_pos, joint_pos)

        stance_mask = np.zeros(2, dtype=np.float32)
        if self.use_jax:
            stance_mask = stance_mask.at[0].set(np.any(sin_phase_signal >= 0))
            stance_mask = stance_mask.at[1].set(np.any(sin_phase_signal < 0))
            stance_mask = np.where(double_support_mask, 1, stance_mask)
        else:
            stance_mask[0] = np.any(sin_phase_signal >= 0)
            stance_mask[1] = np.any(sin_phase_signal < 0)
            stance_mask[double_support_mask] = 1

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

    def calculate_leg_angles(self, signal: ArrayType, is_left: bool):
        knee_angle = np.abs(
            signal * (self.max_knee_pitch - self.knee_pitch_default)
            + (2 * int(is_left) - 1) * self.knee_pitch_default
        )
        ank_pitch_angle = np.arctan2(
            np.sin(knee_angle), np.cos(knee_angle) + self.shin_thigh_ratio
        )
        hip_pitch_angle = knee_angle - ank_pitch_angle

        if is_left:
            return {
                "left_hip_pitch": -hip_pitch_angle,
                "left_knee_pitch": knee_angle,
                "left_ank_pitch": -ank_pitch_angle,
            }
        else:
            return {
                "right_hip_pitch": hip_pitch_angle,
                "right_knee_pitch": -knee_angle,
                "right_ank_pitch": -ank_pitch_angle,
            }

    def initialize_lipm_model(
        COM_pos_0: List[float],
        COM_v0: List[float],
        left_foot_pos: List[float],
        right_foot_pos: List[float],
        support_leg: str = "left_leg",
    ) -> LIPM3DPlanner:
        model = LIPM3DPlanner(
            control_dt=0.02, T=0.34, s_d=0.6, w_d=0.4, support_leg=support_leg
        )
        model.initialize_model(COM_pos_0, left_foot_pos, right_foot_pos)
        model.x_0 = model.COM_pos[0] - model.support_foot_pos[0]
        model.y_0 = model.COM_pos[1] - model.support_foot_pos[1]
        model.vx_0 = COM_v0[0]
        model.vy_0 = COM_v0[1]
        model.x_t = model.x_0
        model.y_t = model.y_0
        model.vx_t = model.vx_0
        model.vy_t = model.vy_0
        return model

    def calculate_swing_foot_positions(
        swing_data_len: int, start_pos: List[float], target_pos: List[float]
    ) -> np.ndarray:
        swing_foot_pos = np.zeros((swing_data_len, 3))
        swing_foot_pos[:, 0] = np.linspace(start_pos[0], target_pos[0], swing_data_len)
        swing_foot_pos[:, 1] = np.linspace(start_pos[1], target_pos[1], swing_data_len)
        swing_foot_pos[1 : swing_data_len - 1, 2] = 0.1
        return swing_foot_pos

    def update_com_data(
        model: LIPM3DPlanner,
        COM_pos_x: List[float],
        COM_pos_y: List[float],
        COM_vel_x: List[float],
        COM_vel_y: List[float],
        COM_dvel_x: List[float],
        COM_dvel_y: List[float],
        COM_dvel: np.ndarray,
    ) -> None:
        COM_pos_x.append(model.x_t + model.support_foot_pos[0])
        COM_pos_y.append(model.y_t + model.support_foot_pos[1])
        COM_vel_x.append(model.vx_t)
        COM_vel_y.append(model.vy_t)
        COM_dvel_x.append(COM_dvel[0])
        COM_dvel_y.append(COM_dvel[1])

    def update_foot_positions(
        model: LIPM3DPlanner,
        swing_foot_pos: np.ndarray,
        j: int,
        left_foot_pos_x: List[float],
        left_foot_pos_y: List[float],
        left_foot_pos_z: List[float],
        right_foot_pos_x: List[float],
        right_foot_pos_y: List[float],
        right_foot_pos_z: List[float],
    ) -> None:
        if model.support_leg == "left_leg":
            model.right_foot_pos = [
                swing_foot_pos[j, 0],
                swing_foot_pos[j, 1],
                swing_foot_pos[j, 2],
            ]
        else:
            model.left_foot_pos = [
                swing_foot_pos[j, 0],
                swing_foot_pos[j, 1],
                swing_foot_pos[j, 2],
            ]

        left_foot_pos_x.append(model.left_foot_pos[0])
        left_foot_pos_y.append(model.left_foot_pos[1])
        left_foot_pos_z.append(model.left_foot_pos[2])
        right_foot_pos_x.append(model.right_foot_pos[0])
        right_foot_pos_y.append(model.right_foot_pos[1])
        right_foot_pos_z.append(model.right_foot_pos[2])

    def update_step_and_switch_leg(
        model: LIPM3DPlanner,
        step_num: int,
        step_to_cmdv: List[int],
        COM_dvel_list: np.ndarray,
        w_d_list: np.ndarray,
        theta: float,
    ) -> Tuple[int, float, np.ndarray]:
        model.switch_support_leg()
        step_num += 1

        if step_num >= step_to_cmdv[2]:
            theta = np.arctan2(COM_dvel_list[3, 1], COM_dvel_list[3, 0])
            model.s_d = np.linalg.norm(COM_dvel_list[3]) * model.T
            COM_dvel = COM_dvel_list[3]
            model.w_d = w_d_list[3]
        elif step_num >= step_to_cmdv[1]:
            theta = np.arctan2(COM_dvel_list[2, 1], COM_dvel_list[2, 0])
            model.s_d = np.linalg.norm(COM_dvel_list[2]) * model.T
            COM_dvel = COM_dvel_list[2]
            model.w_d = w_d_list[2]
        elif step_num >= step_to_cmdv[0]:
            theta = np.arctan2(COM_dvel_list[1, 1], COM_dvel_list[1, 0])
            model.s_d = np.linalg.norm(COM_dvel_list[1]) * model.T
            COM_dvel = COM_dvel_list[1]
            model.w_d = w_d_list[1]
        else:
            theta = np.arctan2(COM_dvel_list[0, 1], COM_dvel_list[0, 0])
            model.s_d = np.linalg.norm(COM_dvel_list[0]) * model.T
            COM_dvel = COM_dvel_list[0]
            model.w_d = w_d_list[0]

        return step_num, theta, COM_dvel

    def update_step_data(
        model: LIPM3DPlanner,
        prev_support_foot_pos: np.ndarray,
        support_foot_pos: np.ndarray,
        step_length: List[float],
        dstep_length: List[float],
        step_width: List[float],
        dstep_width: List[float],
        theta: float,
    ) -> np.ndarray:
        rsupport_foot_pos_x = (
            np.cos(theta) * support_foot_pos[0] + np.sin(theta) * support_foot_pos[1]
        )
        rsupport_foot_pos_y = (
            -np.sin(theta) * support_foot_pos[0] + np.cos(theta) * support_foot_pos[1]
        )
        rprev_support_foot_pos_x = (
            np.cos(theta) * prev_support_foot_pos[0]
            + np.sin(theta) * prev_support_foot_pos[1]
        )
        rprev_support_foot_pos_y = (
            -np.sin(theta) * prev_support_foot_pos[0]
            + np.cos(theta) * prev_support_foot_pos[1]
        )

        step_length.append(rsupport_foot_pos_x - rprev_support_foot_pos_x)
        dstep_length.append(model.s_d)
        step_width.append(np.abs(rsupport_foot_pos_y - rprev_support_foot_pos_y))
        dstep_width.append(model.w_d)

        return support_foot_pos
