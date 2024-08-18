from typing import Any, List, Optional, Tuple

from toddlerbot.algorithms.zmp.footstep_planner import FootStepPlanner
from toddlerbot.algorithms.zmp.zmp_planner import ZMPPlanner
from toddlerbot.motion_reference.motion_ref import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update, loop_update
from toddlerbot.utils.array_utils import array_lib as np
from toddlerbot.utils.math_utils import quat2euler, resample_trajectory


class WalkZMPReference(MotionReference):
    def __init__(
        self,
        robot: Robot,
        use_jax: bool = False,
        default_joint_pos: Optional[ArrayType] = None,
        default_joint_vel: Optional[ArrayType] = None,
        max_knee_pitch: float = np.pi / 3,
        double_support_phase: float = 0.5,
        single_support_phase: float = 1.0,
        plan_max_stride: List[float] = [0.1, 0.05, np.pi / 8],
    ):
        super().__init__("walk", "periodic", robot, use_jax)

        self.default_joint_pos = default_joint_pos
        self.default_joint_vel = default_joint_vel
        if self.default_joint_pos is None:
            self.knee_pitch_default = 0.0
        else:
            self.knee_pitch_default = self.default_joint_pos[
                self.get_joint_idx("left_knee_pitch")
            ]

        self.num_joints = len(self.robot.joint_ordering)
        self.shin_thigh_ratio = (
            self.robot.data_dict["offsets"]["knee_to_ank_pitch_z"]
            / self.robot.data_dict["offsets"]["hip_pitch_to_knee_z"]
        )
        self.com_z = self.robot.config["general"]["offsets"]["torso_z"]
        self.max_knee_pitch = max_knee_pitch
        self.double_support_phase = double_support_phase
        self.single_support_phase = single_support_phase

        self.zmp_planner = ZMPPlanner()

        self.control_dt = 0.01
        self.control_cost_Q = 1.0
        self.control_cost_R = 0.1
        self.footstep_height = 0.03
        self.foot_to_com_x = robot.data_dict["offsets"]["foot_to_com_x"]
        self.foot_to_com_y = robot.data_dict["offsets"]["foot_to_com_y"]

        self.init_joint_angles = robot.init_joint_angles
        self.joint_angles = self.init_joint_angles
        # (time, joint_angles)
        self.joint_angles_list = [self.joint_angles]

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
        #         [numpy.asarray(fs[1:4]) for fs in footsteps], dtype=numpy.float32
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

        desired_zmps = [step[:2] for step in footsteps for _ in range(2)]
        time_list = np.array(  # type: ignore
            [0, self.single_support_phase]
            + [
                self.double_support_phase,
                self.single_support_phase,
            ]
            * (len(footsteps) - 1),
            dtype=np.float32,
        )
        time_steps = np.cumsum(time_list)  # type: ignore

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

        traj = {
            "time": np.zeros(N, dtype=np.float32),  # type: ignore
            "x": np.zeros((N, 4), dtype=np.float32),  # type: ignore
            "u": np.zeros((N, 2), dtype=np.float32),  # type: ignore
            "cop": np.zeros((N, 2), dtype=np.float32),  # type: ignore
            "desired_zmp": np.zeros((N, 2), dtype=np.float32),  # type: ignore
            "nominal_com": np.zeros((N, 6), dtype=np.float32),  # type: ignore
        }

        for i in range(N):
            # Update traj["time"]
            traj["time"] = inplace_update(
                traj["time"], i, time_steps[0] + i * self.control_dt
            )
            if i == 0:
                # Set the initial state for traj["x"]
                traj["x"] = inplace_update(traj["x"], i, x0)
            else:
                # Compute the state derivative xd
                xd = np.hstack((traj["x"][i - 1, 2:], traj["u"][i - 1, :]))  # type: ignore
                traj["x"] = inplace_update(
                    traj["x"], i, traj["x"][i - 1, :] + xd * self.control_dt
                )

            # Update traj["u"] using the ZMP planner
            traj["u"] = inplace_update(
                traj["u"],
                i,
                self.zmp_planner.get_optim_com_acc(traj["time"][i], traj["x"][i, :]),
            )

        print()

        # zmp_ref_traj = self.zmp_planner.compute_zmp_ref_traj(self.foot_steps)
        # zmp_traj, self.com_ref_traj = self.zmp_planner.compute_com_traj(
        #     self.com_init, zmp_ref_traj
        # )

        # foot_steps_copy = copy.deepcopy(self.foot_steps)
        # com_ref_traj_copy = copy.deepcopy(self.com_ref_traj)

        # self.joint_angles_list.append((0.5, self.init_joint_angles))
        # while len(self.com_ref_traj) > 0:
        #     if self.idx == 0:
        #         t = self.config.squat_time + 0.5
        #     else:
        #         t = self.joint_angles_list[-1][0] + self.config.control_dt

        #     joint_angles = self.solve_joint_angles()
        #     self.joint_angles_list.append((t, joint_angles))

        # self.joint_angles_list.append((t + 0.5, self.init_joint_angles))

        # joint_angles_list = resample_trajectory(
        #     self.joint_angles_list,
        #     desired_interval=self.config.control_dt,
        #     interp_type="cubic",
        # )

        # return (
        #     path,
        #     foot_steps_copy,
        #     zmp_ref_traj,
        #     zmp_traj,
        #     com_ref_traj_copy,
        #     joint_angles_list,
        # )

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

    def _compute_foot_pos(self, com_pos):
        # Need to update here
        idx_curr = self.idx % self.num_footsteps
        up_start_idx = round(self.num_footsteps / 4)
        up_end_idx = round(self.num_footsteps / 2)
        up_period = up_end_idx - up_start_idx

        up_delta = self.config.foot_step_height / up_period
        support_leg = self.foot_steps[0].support_leg

        # Up or down foot movement
        if up_start_idx < idx_curr <= up_end_idx:
            if support_leg == "right":
                self.left_up += up_delta
            elif support_leg == "left":
                self.right_up += up_delta
        else:
            if support_leg == "right":
                self.left_up = max(self.left_up - up_delta, 0.0)
            elif support_leg == "left":
                self.right_up = max(self.right_up - up_delta, 0.0)

        # Move foot in the axes of x, y, theta
        if idx_curr > up_start_idx + up_period * 2:
            if support_leg == "right":
                self.left_pos = self.left_pos_target.copy()
            elif support_leg == "left":
                self.right_pos = self.right_pos_target.copy()
        elif idx_curr > up_start_idx:
            if support_leg == "right":
                self.left_pos += self.left_pos_delta
                self.theta_curr += self.theta_delta
            elif support_leg == "left":
                self.right_pos += self.right_pos_delta
                self.theta_curr += self.theta_delta

        left_hip_pos = [
            com_pos[0] - self.foot_to_com_x,
            com_pos[1] + self.foot_to_com_y,
        ]
        right_hip_pos = [
            com_pos[0] - self.foot_to_com_x,
            com_pos[1] - self.foot_to_com_y,
        ]

        if support_leg == "left":
            right_hip_pos[0] += self.foot_to_com_y * 2 * np.sin(self.theta_curr)
            right_hip_pos[1] += self.foot_to_com_y * 2 * (1 - np.cos(self.theta_curr))
        elif support_leg == "right":
            left_hip_pos[0] += self.foot_to_com_y * 2 * np.sin(self.theta_curr)
            left_hip_pos[1] += self.foot_to_com_y * 2 * (1 - np.cos(self.theta_curr))

        # target end effector positions in the hip frame
        left_offset_x = self.left_pos[0] - left_hip_pos[0]
        left_offset_y = self.left_pos[1] - left_hip_pos[1]
        right_offset_x = self.right_pos[0] - right_hip_pos[0]
        right_offset_y = self.right_pos[1] - right_hip_pos[1]

        left_foot_pos = [
            left_offset_x * np.cos(self.left_pos[2])
            + left_offset_y * np.sin(self.left_pos[2]),
            -left_offset_x * np.sin(self.left_pos[2])
            + left_offset_y * np.cos(self.left_pos[2]),
            self.config.squat_height + self.left_up,
        ]
        right_foot_pos = [
            right_offset_x * np.cos(self.right_pos[2])
            + right_offset_y * np.sin(self.right_pos[2]),
            -right_offset_x * np.sin(self.right_pos[2])
            + right_offset_y * np.cos(self.right_pos[2]),
            self.config.squat_height + self.right_up,
        ]

        left_foot_ori = [0, 0, self.theta_curr - self.left_pos[2]]
        right_foot_ori = [0, 0, self.theta_curr - self.right_pos[2]]

        return left_foot_pos, left_foot_ori, right_foot_pos, right_foot_ori

    def solve_joint_angles(self):
        if abs(self.idx * self.config.control_dt - self.foot_steps[1].time) < 1e-6:
            self.foot_steps.pop(0)

            fs_curr = self.foot_steps[0]
            fs_next = self.foot_steps[1]
            self.theta_delta = (fs_next.position[2] - fs_curr.position[2]) / (
                self.num_footsteps / 2
            )
            if fs_curr.support_leg == "left":
                if fs_next.support_leg == "right":
                    self.right_pos_target = fs_next.position
                else:
                    self.right_pos_target = np.array(
                        [
                            fs_next.position[0]
                            + np.sin(fs_next.position[2]) * self.foot_to_com_y,
                            fs_next.position[1]
                            - np.cos(fs_next.position[2]) * self.foot_to_com_y,
                            fs_next.position[2],
                        ]
                    )

                self.right_pos_delta = (self.right_pos_target - self.right_pos) / (
                    self.num_footsteps / 2
                )
            elif fs_curr.support_leg == "right":
                if fs_next.support_leg == "left":
                    self.left_pos_target = fs_next.position
                else:
                    self.left_pos_target = np.array(
                        [
                            fs_next.position[0]
                            - np.sin(fs_next.position[2]) * self.foot_to_com_y,
                            fs_next.position[1]
                            + np.cos(fs_next.position[2]) * self.foot_to_com_y,
                            fs_next.position[2],
                        ]
                    )

                self.left_pos_delta = (self.left_pos_target - self.left_pos) / (
                    self.num_footsteps / 2
                )

        com_pos_ref = self.com_ref_traj.pop(0)
        self.joint_angles = self.robot.solve_ik(
            *self._compute_foot_pos(com_pos_ref), self.joint_angles
        )
        self.idx += 1

        return self.joint_angles

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


if __name__ == "__main__":
    from toddlerbot.utils.math_utils import round_to_sig_digits

    robot = Robot("toddlerbot")
    walk_ref = WalkZMPReference(robot, max_knee_pitch=0.523599)
    left_leg_angles = walk_ref.calculate_leg_angles(np.ones(1, dtype=np.float32), True)
    left_ank_act = robot.ankle_ik([0.0, left_leg_angles["left_ank_pitch"].item()])
    print(left_leg_angles)
    print([round_to_sig_digits(x, 6) for x in left_ank_act])
