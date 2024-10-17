from typing import Tuple

from toddlerbot.motion.motion_ref import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np


class SquatReference(MotionReference):
    def __init__(self, robot: Robot, dt: float):
        super().__init__("squat", "perceptual", robot, dt)

    def get_vel(self, command: ArrayType) -> Tuple[ArrayType, ArrayType]:
        lin_vel = np.array([0.0, 0.0, command[-1]], dtype=np.float32)
        ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return lin_vel, ang_vel

    def get_state_ref(
        self, state_curr: ArrayType, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        torso_state = self.integrate_torso_state(
            state_curr[:3], state_curr[3:7], command
        )
        joint_pos_curr = state_curr[13 : 13 + self.robot.nu]

        neck_yaw_pos = np.interp(
            command[0],
            np.array([-1, 0, 1]),
            np.array([self.neck_joint_limits[0, 0], 0.0, self.neck_joint_limits[1, 0]]),
        )
        neck_pitch_pos = np.interp(
            command[1],
            np.array([-1, 0, 1]),
            np.array([self.neck_joint_limits[0, 1], 0.0, self.neck_joint_limits[1, 1]]),
        )
        neck_joint_pos = np.array([neck_yaw_pos, neck_pitch_pos])

        ref_idx = (command[2] * (self.arm_ref_size - 2)).astype(int)
        # Linearly interpolate between p_start and p_end
        arm_joint_pos = self.arm_joint_pos_ref[ref_idx]

        waist_roll_pos = np.interp(
            command[3],
            np.array([-1, 0, 1]),
            np.array(
                [self.waist_joint_limits[0, 0], 0.0, self.waist_joint_limits[1, 0]]
            ),
        )
        waist_yaw_pos = np.interp(
            command[4],
            np.array([-1, 0, 1]),
            np.array(
                [self.waist_joint_limits[0, 1], 0.0, self.waist_joint_limits[1, 1]]
            ),
        )
        waist_joint_pos = np.array([waist_roll_pos, waist_yaw_pos])

        com_curr = self.com_fk(joint_pos_curr[self.left_knee_pitch_idx])
        com_z_target = np.clip(
            com_curr[2] + self.dt * command[5],
            self.com_z_limits[0],
            self.com_z_limits[1],
        )
        leg_joint_pos = self.com_ik(com_z_target)

        joint_pos = self.default_joint_pos.copy()
        joint_pos = inplace_update(joint_pos, self.neck_joint_indices, neck_joint_pos)
        joint_pos = inplace_update(joint_pos, self.arm_joint_indices, arm_joint_pos)
        joint_pos = inplace_update(joint_pos, self.waist_joint_indices, waist_joint_pos)
        joint_pos = inplace_update(joint_pos, self.leg_joint_indices, leg_joint_pos)

        joint_vel = self.default_joint_vel.copy()

        stance_mask = np.ones(2, dtype=np.float32)

        return np.concatenate((torso_state, joint_pos, joint_vel, stance_mask))

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
        waist_joint_pos = state_ref[13 + self.waist_motor_indices]
        waist_motor_pos = self.waist_ik(waist_joint_pos)
        motor_target = inplace_update(
            motor_target,
            self.waist_motor_indices,
            waist_motor_pos,
        )
        leg_joint_pos = state_ref[13 + self.leg_motor_indices]
        leg_motor_pos = self.leg_ik(leg_joint_pos)
        motor_target = inplace_update(
            motor_target,
            self.leg_motor_indices,
            leg_motor_pos,
        )

        return motor_target
