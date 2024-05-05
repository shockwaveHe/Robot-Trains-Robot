import copy

import numpy as np

from toddlerbot.planning.foot_step_planner import (
    FootStepPlanner,
    FootStepPlanParameters,
)
from toddlerbot.planning.zmp_feedback_planner import ZMPFeedbackPlanner
from toddlerbot.utils.math_utils import resample_trajectory


class Walking:
    """Class to handle the walking motion of a humanoid robot."""

    def __init__(self, robot, config):
        self.robot = robot
        self.config = config
        self.fs_steps = round(config.plan_t_step / config.control_dt)
        self.x_offset_com_to_foot = robot.com[0]
        self.y_offset_com_to_foot = robot.offsets["y_offset_com_to_foot"]

        zero_joint_angles, self.initial_joint_angles = robot.initialize_joint_angles()
        self.joint_angles = self.initial_joint_angles
        # (time, joint_angles)
        self.joint_angles_traj = [(0.0, zero_joint_angles)]

        plan_params = FootStepPlanParameters(
            max_stride=np.array(config.plan_max_stride),
            t_step=config.plan_t_step,
            y_offset_com_to_foot=self.y_offset_com_to_foot,
        )
        self.fsp = FootStepPlanner(plan_params)

        self.zmp_controller = ZMPFeedbackPlanner(
            com_z=robot.com[2] - config.squat_height,
            dt=config.control_dt,
            t_preview=config.control_t_preview,
            t_filter=config.control_t_filter,
            Q_val=config.zmp_control_cost_Q,
            R_val=config.zmp_control_cost_R,
            x_offset_com_to_foot=self.x_offset_com_to_foot,
            y_disp_zmp=config.y_offset_zmp - self.y_offset_com_to_foot,
        )
        # self.lqr_controller = LQRFullBodyController(
        #     nq=len(self.initial_joint_angles),
        #     dt=config.control_dt,
        #     Q_val=config.lqr_control_cost_Q,
        #     R_val=config.lqr_control_cost_R,
        # )

        self.curr_pose = None
        self.com_init = np.concatenate(
            [np.array(robot.com)[None, :2], np.zeros((2, 2))], axis=0
        )

        self.idx = 0

        self.foot_steps = []
        self.com_ref_traj = []

        self.left_up = self.right_up = 0.0
        # Assume the initial state is the canonical pose
        self.theta_curr = 0.0
        self.left_pos = np.array([0.0, self.y_offset_com_to_foot, 0.0])
        self.right_pos = np.array([0.0, -self.y_offset_com_to_foot, 0.0])

    def plan(self, curr_pose=None, target_pose=None):
        if self.curr_pose is not None:
            curr_pose = self.curr_pose

        path, self.foot_steps = self.fsp.compute_steps(curr_pose, target_pose)
        zmp_ref_traj = self.zmp_controller.compute_zmp_ref_traj(self.foot_steps)
        zmp_traj, self.com_ref_traj = self.zmp_controller.compute_com_traj(
            self.com_init, zmp_ref_traj
        )

        foot_steps_copy = copy.deepcopy(self.foot_steps)
        com_ref_traj_copy = copy.deepcopy(self.com_ref_traj)

        self.joint_angles_traj.append((0.5, self.initial_joint_angles))
        while len(self.com_ref_traj) > 0:
            if self.idx == 0:
                t = self.config.squat_time + 0.5
            else:
                t = self.joint_angles_traj[-1][0] + self.config.control_dt

            joint_angles = self.solve_joint_angles()
            self.joint_angles_traj.append((t, joint_angles))

        self.joint_angles_traj.append((t + 0.5, self.initial_joint_angles))

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
                            + np.sin(fs_next.position[2]) * self.y_offset_com_to_foot,
                            fs_next.position[1]
                            - np.cos(fs_next.position[2]) * self.y_offset_com_to_foot,
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
                            - np.sin(fs_next.position[2]) * self.y_offset_com_to_foot,
                            fs_next.position[1]
                            + np.cos(fs_next.position[2]) * self.y_offset_com_to_foot,
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
            com_pos[0] - self.x_offset_com_to_foot,
            com_pos[1] + self.y_offset_com_to_foot,
        ]
        right_hip_pos = [
            com_pos[0] - self.x_offset_com_to_foot,
            com_pos[1] - self.y_offset_com_to_foot,
        ]

        if support_leg == "left":
            right_hip_pos[0] += self.y_offset_com_to_foot * 2 * np.sin(self.theta_curr)
            right_hip_pos[1] += (
                self.y_offset_com_to_foot * 2 * (1 - np.cos(self.theta_curr))
            )
        elif support_leg == "right":
            left_hip_pos[0] += self.y_offset_com_to_foot * 2 * np.sin(self.theta_curr)
            left_hip_pos[1] += (
                self.y_offset_com_to_foot * 2 * (1 - np.cos(self.theta_curr))
            )

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
