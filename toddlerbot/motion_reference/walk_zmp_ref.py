from typing import Any, List, Optional

from toddlerbot.algorithms.zmp.zmp_planner import ZMPPlanner
from toddlerbot.motion_reference.motion_ref import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np


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
        stride_size: float = 0.5,
        zmp_y_offset: float = 0.05,
        num_steps: int = 100,
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
        self.stride_size = stride_size
        self.zmp_y_offset = zmp_y_offset
        self.num_steps = num_steps

        self.zmp_planner = ZMPPlanner()

    def reset(self):
        pass

    def get_zmp_ref(
        self,
        path_pos: ArrayType,
        path_quat: ArrayType,
        phase: float,
        command: ArrayType,
        dt: float = 0.01,
    ):
        desired_zmps: List[ArrayType] = []
        for i in range(self.num_steps):
            footstep = np.array(  # type: ignore
                [i * self.stride_size, (-1) ** (i + 1) * self.zmp_y_offset]
            )
            if i == 0 or i == self.num_steps - 1:
                footstep = inplace_update(footstep, 1, 0)  # type: ignore

            desired_zmps.append(footstep)
            desired_zmps.append(footstep)

        time_list = np.array(  # type: ignore
            [0, self.single_support_phase]
            + [
                self.double_support_phase,
                self.single_support_phase,
            ]
            * (self.num_steps - 1),
            dtype=np.float32,
        )
        time_steps = np.cumsum(time_list)  # type: ignore

        self.zmp_planner.plan(
            time_steps,
            desired_zmps,
            np.array([0, 0, 0, 0], dtype=np.float32),  # type: ignore
            self.com_z,
            Qy=np.eye(2, dtype=np.float32),  # type: ignore
            R=np.eye(2, dtype=np.float32) * 0.1,  # type: ignore
        )

        x0 = np.array([path_pos[0], path_pos[1], command[0], command[1]])  # type: ignore
        x0 = np.array([0, 0, 0.2, -0.1])  # type: ignore

        N = int((time_steps[-1] - time_steps[0]) / dt)

        # time = time_steps[0] + np.arange(N) * dt  # type: ignore
        # com_pos = jax.vmap(self.zmp_planner.get_nominal_com)(time)

        # traj = {
        #     "time": np.zeros(N, dtype=np.float32),  # type: ignore
        #     "x": np.zeros((N, 4), dtype=np.float32),  # type: ignore
        #     "u": np.zeros((N, 2), dtype=np.float32),  # type: ignore
        #     "cop": np.zeros((N, 2), dtype=np.float32),  # type: ignore
        #     "desired_zmp": np.zeros((N, 2), dtype=np.float32),  # type: ignore
        #     "nominal_com": np.zeros((N, 6), dtype=np.float32),  # type: ignore
        # }

        # for i in range(3):
        #     traj["time"] = traj["time"].at[i].set(time_steps[0] + i * dt)
        #     if i == 0:
        #         traj["x"] = traj["x"].at[i, :].set(x0)
        #     else:
        #         xd = np.hstack((traj["x"][i - 1, 2:], traj["u"][i - 1, :]))  # type: ignore
        #         traj["x"] = traj["x"].at[i, :].set(traj["x"][i - 1, :] + xd * dt)

        #     traj["u"] = (
        #         traj["u"]
        #         .at[i, :]
        #         .set(
        #             self.zmp_planner.get_optim_com_acc(traj["time"][i], traj["x"][i, :])
        #         )
        #     )

        #     traj["cop"] = (
        #         traj["cop"]
        #         .at[i, :]
        #         .set(self.zmp_planner.com_acc_to_cop(traj["x"][i, :], traj["u"][i, :]))
        #     )

        #     traj["desired_zmp"] = (
        #         traj["desired_zmp"]
        #         .at[i, :]
        #         .set(self.zmp_planner.get_desired_zmp(traj["time"][i]))
        #     )

        #     traj["nominal_com"] = (
        #         traj["nominal_com"]
        #         .at[i, :2]
        #         .set(self.zmp_planner.get_nominal_com(traj["time"][i]))
        #     )
        #     traj["nominal_com"] = (
        #         traj["nominal_com"]
        #         .at[i, 2:4]
        #         .set(self.zmp_planner.get_nominal_com_vel(traj["time"][i]))
        #     )
        #     traj["nominal_com"] = (
        #         traj["nominal_com"]
        #         .at[i, 4:]
        #         .set(self.zmp_planner.get_nominal_com_acc(traj["time"][i]))
        #     )

        # with open("jax_results.txt", "w") as file:
        #     file.write("Time:\n" + str(traj["time"]) + "\n\n")
        #     file.write("Desired ZMP:\n" + str(traj["desired_zmp"]) + "\n\n")
        #     file.write("Nominal COM:\n" + str(traj["nominal_com"]) + "\n\n")
        #     file.write("Center of Pressure (COP):\n" + str(traj["cop"]) + "\n\n")
        #     file.write("State Vector (x):\n" + str(traj["x"]) + "\n\n")
        #     file.write("Control Inputs (u):\n" + str(traj["u"]) + "\n\n")

        # return traj

        if self.curr_pose is not None:
            curr_pose = self.curr_pose

        path, self.foot_steps = self.fsp.compute_steps(curr_pose, target_pose)
        zmp_ref_traj = self.zmp_planner.compute_zmp_ref_traj(self.foot_steps)
        zmp_traj, self.com_ref_traj = self.zmp_planner.compute_com_traj(
            self.com_init, zmp_ref_traj
        )

        foot_steps_copy = copy.deepcopy(self.foot_steps)
        com_ref_traj_copy = copy.deepcopy(self.com_ref_traj)

        self.joint_angles_traj.append((0.5, self.init_joint_angles))
        while len(self.com_ref_traj) > 0:
            if self.idx == 0:
                t = self.config.squat_time + 0.5
            else:
                t = self.joint_angles_traj[-1][0] + self.config.control_dt

            joint_angles = self.solve_joint_angles()
            self.joint_angles_traj.append((t, joint_angles))

        self.joint_angles_traj.append((t + 0.5, self.init_joint_angles))

        joint_angles_traj = resample_trajectory(
            self.joint_angles_traj,
            desired_interval=self.config.control_dt,
            interp_type="cubic",
        )

        return (
            path,
            foot_steps_copy,
            zmp_ref_traj,
            zmp_traj,
            com_ref_traj_copy,
            joint_angles_traj,
        )

    def _compute_foot_pos(self, com_pos):
        # Need to update here
        idx_curr = self.idx % self.fs_steps
        up_start_idx = round(self.fs_steps / 4)
        up_end_idx = round(self.fs_steps / 2)
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
                if self.config.rotate_torso:
                    self.theta_curr += self.theta_delta
            elif support_leg == "left":
                self.right_pos += self.right_pos_delta
                if self.config.rotate_torso:
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
                self.fs_steps / 2
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
                    self.fs_steps / 2
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
                    self.fs_steps / 2
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
