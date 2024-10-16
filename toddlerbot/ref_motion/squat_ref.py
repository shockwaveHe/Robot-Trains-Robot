from typing import Tuple

from toddlerbot.ref_motion import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np


class SquatReference(MotionReference):
    def __init__(self, robot: Robot, dt: float, com_z_lower_limit_offset: float = 0.01):
        super().__init__("squat", "perceptual", robot, dt)

        self._setup_com_ik(com_z_lower_limit_offset)

    def _setup_com_ik(self, com_z_lower_limit_offset: float):
        self.knee_pitch_default = self.default_joint_pos[
            self.robot.joint_ordering.index("left_knee_pitch")
        ]
        self.hip_pitch_to_knee_z = self.robot.data_dict["offsets"][
            "hip_pitch_to_knee_z"
        ]
        self.knee_to_ank_pitch_z = self.robot.data_dict["offsets"][
            "knee_to_ank_pitch_z"
        ]
        self.hip_pitch_to_ank_pitch_z = np.sqrt(
            self.hip_pitch_to_knee_z**2
            + self.knee_to_ank_pitch_z**2
            - 2
            * self.hip_pitch_to_knee_z
            * self.knee_to_ank_pitch_z
            * np.cos(np.pi - self.knee_pitch_default)
        )
        self.shin_thigh_ratio = self.knee_to_ank_pitch_z / self.hip_pitch_to_knee_z

        self.com_z_limits = np.array(
            [
                self.get_com_z(self.robot.joint_limits["left_knee_pitch"][1]).item()
                + com_z_lower_limit_offset,
                0.0,
            ],
            dtype=np.float32,
        )
        self.left_knee_pitch_idx = self.robot.joint_ordering.index("left_knee_pitch")
        self.leg_pitch_joint_indicies = np.array(
            [
                self.robot.joint_ordering.index(joint_name)
                for joint_name in [
                    "left_hip_pitch",
                    "left_knee_pitch",
                    "left_ank_pitch",
                    "right_hip_pitch",
                    "right_knee_pitch",
                    "right_ank_pitch",
                ]
            ]
        )
        self.leg_roll_joint_indicies = np.array(
            [
                self.robot.joint_ordering.index(joint_name)
                for joint_name in [
                    "left_hip_roll",
                    "left_ank_roll",
                    "right_hip_roll",
                    "right_ank_roll",
                ]
            ]
        )

    def get_phase_signal(self, time_curr: float | ArrayType) -> ArrayType:
        return np.zeros(1, dtype=np.float32)

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

        # neck yaw, neck pitch, arm, waist roll, waist yaw, squat
        neck_joint_pos = self.neck_joint_limits[0] + command[:2] * (
            self.neck_joint_limits[1] - self.neck_joint_limits[0]
        )

        ref_idx = (command[2] * (self.arm_ref_size - 2)).astype(int)
        # Linearly interpolate between p_start and p_end
        arm_joint_pos = self.arm_joint_pos_ref[ref_idx]

        waist_joint_pos = self.waist_joint_limits[0] + command[3:5] * (
            self.waist_joint_limits[1] - self.waist_joint_limits[0]
        )

        com_z_curr = self.get_com_z(joint_pos_curr[self.left_knee_pitch_idx])
        com_z_target = np.clip(
            com_z_curr + self.dt * command[5],
            self.com_z_limits[0],
            self.com_z_limits[1],
        )
        leg_pitch_joint_pos = self.get_leg_pitch_pos(com_z_target)

        joint_pos = self.default_joint_pos.copy()
        joint_pos = inplace_update(joint_pos, self.neck_joint_indices, neck_joint_pos)
        joint_pos = inplace_update(joint_pos, self.arm_joint_indices, arm_joint_pos)
        joint_pos = inplace_update(joint_pos, self.waist_joint_indices, waist_joint_pos)
        joint_pos = inplace_update(
            joint_pos, self.leg_pitch_joint_indicies, leg_pitch_joint_pos
        )

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

    def get_com_z(self, knee_angle: float | ArrayType) -> ArrayType:
        # Compute the length from hip pitch to ankle pitch along the z-axis
        com_z = np.array(
            np.sqrt(
                self.hip_pitch_to_knee_z**2
                + self.knee_to_ank_pitch_z**2
                - 2
                * self.hip_pitch_to_knee_z
                * self.knee_to_ank_pitch_z
                * np.cos(np.pi - knee_angle)
            )
            - self.hip_pitch_to_ank_pitch_z,
            dtype=np.float32,
        )
        return com_z

    def get_leg_pitch_pos(self, com_z_target: ArrayType) -> ArrayType:
        knee_angle_cos = (
            self.hip_pitch_to_knee_z**2
            + self.knee_to_ank_pitch_z**2
            - (self.hip_pitch_to_ank_pitch_z + com_z_target) ** 2
        ) / (2 * self.hip_pitch_to_knee_z * self.knee_to_ank_pitch_z)
        knee_angle_cos = np.clip(knee_angle_cos, -1.0, 1.0)
        knee_angle = np.abs(np.pi - np.arccos(knee_angle_cos))

        ank_pitch_angle = np.arctan2(
            np.sin(knee_angle),
            np.cos(knee_angle) + self.shin_thigh_ratio,
        )
        hip_pitch_angle = knee_angle - ank_pitch_angle

        return np.array(
            [
                -hip_pitch_angle,
                knee_angle,
                -ank_pitch_angle,
                hip_pitch_angle,
                -knee_angle,
                -ank_pitch_angle,
            ],
            dtype=np.float32,
        )
