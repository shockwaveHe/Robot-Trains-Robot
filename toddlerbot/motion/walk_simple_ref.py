from typing import Tuple

from toddlerbot.motion.motion_ref import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np


class WalkSimpleReference(MotionReference):
    def __init__(
        self,
        robot: Robot,
        dt: float,
        cycle_time: float,
        max_knee_pitch: float = np.pi / 3,
        double_support_phase: float = 0.1,
    ):
        super().__init__("walk_simple", "periodic", robot, dt)

        self.cycle_time = cycle_time

        self.knee_pitch_default = self.default_joint_pos[
            self.robot.joint_ordering.index("left_knee_pitch")
        ]
        self.max_knee_pitch = max_knee_pitch
        self.double_support_phase = double_support_phase

        self.num_joints = len(self.robot.joint_ordering)
        self.shin_thigh_ratio = (
            self.robot.data_dict["offsets"]["knee_to_ank_pitch_z"]
            / self.robot.data_dict["offsets"]["hip_pitch_to_knee_z"]
        )

        self.left_pitch_joint_indices = np.array(
            [
                self.robot.joint_ordering.index("left_hip_pitch"),
                self.robot.joint_ordering.index("left_knee_pitch"),
                self.robot.joint_ordering.index("left_ank_pitch"),
            ]
        )
        self.right_pitch_joint_indices = np.array(
            [
                self.robot.joint_ordering.index("right_hip_pitch"),
                self.robot.joint_ordering.index("right_knee_pitch"),
                self.robot.joint_ordering.index("right_ank_pitch"),
            ]
        )

    def get_phase_signal(self, time_curr: float | ArrayType) -> ArrayType:
        phase_signal = np.array(
            [
                np.sin(2 * np.pi * time_curr / self.cycle_time),
                np.cos(2 * np.pi * time_curr / self.cycle_time),
            ],
            dtype=np.float32,
        )
        return phase_signal

    def get_vel(self, command: ArrayType) -> Tuple[ArrayType, ArrayType]:
        lin_vel = np.array([command[0], command[1], 0.0], dtype=np.float32)
        ang_vel = np.array([0.0, 0.0, command[2]], dtype=np.float32)
        return lin_vel, ang_vel

    def get_state_ref(
        self, state_curr: ArrayType, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        torso_state = self.integrate_torso_state(
            state_curr[:3], state_curr[3:7], command
        )

        sin_phase_signal = np.sin(2 * np.pi * time_curr / self.cycle_time)
        signal_left = np.clip(sin_phase_signal, 0, None)
        signal_right = np.clip(sin_phase_signal, None, 0)

        left_leg_pitch_pos = self.get_leg_pitch_pos(signal_left, True)
        right_leg_pitch_pos = self.get_leg_pitch_pos(signal_right, False)

        joint_pos = self.default_joint_pos.copy()
        joint_pos = inplace_update(
            joint_pos, self.left_pitch_joint_indices, left_leg_pitch_pos
        )
        joint_pos = inplace_update(
            joint_pos, self.right_pitch_joint_indices, right_leg_pitch_pos
        )
        double_support_mask = np.abs(sin_phase_signal) < self.double_support_phase
        joint_pos = np.where(double_support_mask, self.default_joint_pos, joint_pos)

        joint_vel = self.default_joint_vel.copy()

        stance_mask = np.zeros(2, dtype=np.float32)
        stance_mask = inplace_update(stance_mask, 0, np.any(sin_phase_signal >= 0))
        stance_mask = inplace_update(stance_mask, 1, np.any(sin_phase_signal < 0))
        stance_mask = np.where(double_support_mask, 1, stance_mask)

        return np.concatenate((torso_state, joint_pos, joint_vel, stance_mask))

    def get_leg_pitch_pos(self, signal: ArrayType, is_left: bool):
        knee_angle = np.abs(
            signal * (self.max_knee_pitch - self.knee_pitch_default)
            + (2 * int(is_left) - 1) * self.knee_pitch_default
        )
        ank_pitch_angle = np.arctan2(
            np.sin(knee_angle),
            np.cos(knee_angle) + self.shin_thigh_ratio,
        )
        hip_pitch_angle = knee_angle - ank_pitch_angle

        if is_left:
            return np.array(
                [-hip_pitch_angle, knee_angle, -ank_pitch_angle], dtype=np.float32
            )
        else:
            return np.array(
                [hip_pitch_angle, -knee_angle, -ank_pitch_angle], dtype=np.float32
            )

    def override_motor_target(
        self, motor_target: ArrayType, state_ref: ArrayType
    ) -> ArrayType:
        motor_target = inplace_update(
            motor_target,
            self.neck_motor_indices,
            self.default_motor_pos[self.neck_motor_indices],
        )
        motor_target = inplace_update(
            motor_target,
            self.arm_motor_indices,
            self.default_motor_pos[self.arm_motor_indices],
        )

        return motor_target
