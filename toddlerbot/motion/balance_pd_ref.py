from typing import List, Tuple

from toddlerbot.motion.motion_ref import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np


class BalancePDReference(MotionReference):
    def __init__(
        self,
        robot: Robot,
        dt: float,
        arm_playback_speed: float = 1.0,
        com_kp: List[float] = [1.0, 1.0],
    ):
        super().__init__("balance_pd", "perceptual", robot, dt)

        self.arm_playback_speed = arm_playback_speed
        if self.arm_playback_speed > 0.0:
            self.arm_time_ref /= arm_playback_speed

        self.com_kp = np.array(com_kp, dtype=np.float32)
        self.ik_iters = 1

    def get_vel(self, command: ArrayType) -> Tuple[ArrayType, ArrayType]:
        lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return lin_vel, ang_vel

    def get_state_ref(
        self, state_curr: ArrayType, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        path_state = self.integrate_path_state(state_curr, command)
        joint_pos_curr = state_curr[13 : 13 + self.robot.nu]

        joint_pos = self.default_joint_pos.copy()

        # neck yaw, neck pitch, arm, waist roll, waist yaw
        neck_joint_pos = np.clip(
            joint_pos_curr[self.neck_joint_indices] + self.dt * command[:2],
            self.neck_joint_limits[0],
            self.neck_joint_limits[1],
        )
        joint_pos = inplace_update(joint_pos, self.neck_joint_indices, neck_joint_pos)

        if self.arm_playback_speed > 0:
            command_ref_idx = (command[2] * (self.arm_ref_size - 2)).astype(int)
            time_ref = self.arm_time_ref[command_ref_idx] + time_curr
            ref_idx = np.minimum(
                np.searchsorted(self.arm_time_ref, time_ref, side="right") - 1,
                self.arm_ref_size - 2,
            )
            # Linearly interpolate between p_start and p_end
            arm_joint_pos_start = self.arm_joint_pos_ref[ref_idx]
            arm_joint_pos_end = self.arm_joint_pos_ref[ref_idx + 1]
            arm_duration = self.arm_time_ref[ref_idx + 1] - self.arm_time_ref[ref_idx]
            arm_joint_pos = arm_joint_pos_start + (
                arm_joint_pos_end - arm_joint_pos_start
            ) * ((time_ref - self.arm_time_ref[ref_idx]) / arm_duration)
            joint_pos = inplace_update(joint_pos, self.arm_joint_indices, arm_joint_pos)
        else:
            joint_pos = inplace_update(
                joint_pos,
                self.arm_joint_indices,
                joint_pos_curr[self.arm_joint_indices],
            )

        waist_joint_pos = np.clip(
            joint_pos_curr[self.waist_joint_indices] + self.dt * command[3:5],
            self.waist_joint_limits[0],
            self.waist_joint_limits[1],
        )
        joint_pos = inplace_update(joint_pos, self.waist_joint_indices, waist_joint_pos)

        com_z_target = np.interp(
            command[5],
            np.array([-1, 0]),
            np.array([self.com_z_limits[0], self.com_z_limits[1]]),
        )
        joint_pos = inplace_update(
            joint_pos, self.leg_joint_indices, self.com_ik(com_z_target)
        )

        # print("start ik")
        # for i in range(self.ik_iters):
        state_ref = np.concatenate((path_state, joint_pos, self.default_joint_vel))
        qpos = self.get_qpos_ref(state_ref)
        data = self.forward(qpos)

        # import cv2
        # import mujoco

        # renderer = mujoco.Renderer(self.model)
        # renderer.update_scene(data)
        # pixels = renderer.render()
        # pixels_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        # cv2.imshow("Simulation", pixels_bgr)
        # cv2.waitKey(1)  # This ensures the window updates without blocking

        com_pos = np.array(data.subtree_com[0], dtype=np.float32)
        # PD controller on CoM position
        com_pos_error = self.desired_com[:2] - com_pos[:2]
        com_ctrl = self.com_kp * com_pos_error

        # print(
        #     f"{i}: com_pos: {com_pos}, com_pos_error: {com_pos_error}, com_ctrl: {com_ctrl}"
        # )

        joint_pos = inplace_update(
            joint_pos,
            self.leg_joint_indices,
            self.com_ik(com_z_target, com_ctrl[0], com_ctrl[1]),
        )

        joint_vel = self.default_joint_vel.copy()

        stance_mask = np.ones(2, dtype=np.float32)

        return np.concatenate((path_state, joint_pos, joint_vel, stance_mask))

    def override_motor_target(
        self, motor_target: ArrayType, state_ref: ArrayType
    ) -> ArrayType:
        neck_joint_pos = state_ref[13 + self.neck_motor_indices]
        neck_motor_pos = self.neck_ik(neck_joint_pos)
        motor_target = inplace_update(
            motor_target,
            self.neck_motor_indices,
            neck_motor_pos,
        )
        arm_joint_pos = state_ref[13 + self.arm_motor_indices]
        arm_motor_pos = self.arm_ik(arm_joint_pos)
        motor_target = inplace_update(
            motor_target,
            self.arm_motor_indices,
            arm_motor_pos,
        )
        waist_joint_pos = state_ref[13 + self.waist_motor_indices]
        waist_motor_pos = self.waist_ik(waist_joint_pos)
        motor_target = inplace_update(
            motor_target,
            self.waist_motor_indices,
            waist_motor_pos,
        )

        return motor_target
