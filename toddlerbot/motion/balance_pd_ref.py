from typing import Tuple

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
    ):
        super().__init__("balance_pd", "perceptual", robot, dt)

        self.arm_playback_speed = arm_playback_speed
        if self.arm_playback_speed > 0.0:
            self.arm_time_ref /= arm_playback_speed

        self.last_com_z_target = 0.0

    def get_vel(self, command: ArrayType) -> Tuple[ArrayType, ArrayType]:
        lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return lin_vel, ang_vel

    def get_state_ref(
        self, state_curr: ArrayType, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        path_state = self.integrate_path_state(state_curr, command)

        joint_pos_curr = state_curr[13 + self.robot.nu : 13 + 2 * self.robot.nu]

        joint_pos = self.default_joint_pos.copy()
        motor_pos = self.default_motor_pos.copy()
        # neck yaw, neck pitch, arm, waist roll, waist yaw
        neck_joint_pos = np.clip(
            joint_pos_curr[self.neck_joint_indices] + self.dt * command[:2],
            self.neck_joint_limits[0],
            self.neck_joint_limits[1],
        )
        joint_pos = inplace_update(joint_pos, self.neck_joint_indices, neck_joint_pos)
        motor_pos = inplace_update(
            motor_pos, self.neck_motor_indices, self.neck_ik(neck_joint_pos)
        )

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
        else:
            arm_joint_pos = joint_pos_curr[self.arm_joint_indices]

        joint_pos = inplace_update(joint_pos, self.arm_joint_indices, arm_joint_pos)
        motor_pos = inplace_update(
            motor_pos, self.arm_motor_indices, self.arm_ik(arm_joint_pos)
        )

        waist_joint_pos = np.clip(
            joint_pos_curr[self.waist_joint_indices] + self.dt * command[3:5],
            self.waist_joint_limits[0],
            self.waist_joint_limits[1],
        )
        joint_pos = inplace_update(joint_pos, self.waist_joint_indices, waist_joint_pos)
        motor_pos = inplace_update(
            motor_pos, self.waist_motor_indices, self.waist_ik(waist_joint_pos)
        )

        # com_curr = self.com_fk(np.abs(joint_pos_curr[self.left_knee_idx]))
        com_z_target = np.clip(
            self.last_com_z_target + self.dt * command[5],
            self.com_z_limits[0],
            self.com_z_limits[1],
        )
        self.last_com_z_target = com_z_target
        leg_joint_pos = self.com_ik(com_z_target)
        joint_pos = inplace_update(joint_pos, self.leg_joint_indices, leg_joint_pos)
        motor_pos = inplace_update(
            motor_pos, self.leg_motor_indices, self.leg_ik(leg_joint_pos)
        )

        state_ref = np.concatenate((path_state, motor_pos, joint_pos))
        qpos = self.get_qpos_ref(state_ref)
        data = self.forward(qpos)

        com_pos = np.array(data.subtree_com[0], dtype=np.float32)
        # PD controller on CoM position
        com_pos_error = self.desired_com[:2] - com_pos[:2]
        com_ctrl = self.com_kp * com_pos_error

        # print(
        #     f"com_pos: {com_pos}, com_pos_error: {com_pos_error}, com_ctrl: {com_ctrl}"
        # )

        leg_joint_pos = self.com_ik(com_z_target, com_ctrl[0], com_ctrl[1])
        joint_pos = inplace_update(joint_pos, self.leg_joint_indices, leg_joint_pos)
        motor_pos = inplace_update(
            motor_pos, self.leg_motor_indices, self.leg_ik(leg_joint_pos)
        )

        stance_mask = np.ones(2, dtype=np.float32)

        return np.concatenate((path_state, motor_pos, joint_pos, stance_mask))
