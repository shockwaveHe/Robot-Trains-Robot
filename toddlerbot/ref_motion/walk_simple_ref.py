from typing import Optional

from toddlerbot.ref_motion import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np


class WalkSimpleReference(MotionReference):
    def __init__(
        self,
        robot: Robot,
        cycle_time: float,
        max_knee_pitch: float = np.pi / 3,
        double_support_phase: float = 0.1,
    ):
        super().__init__("walk_simple", "periodic", robot)

        self.cycle_time = cycle_time
        self.default_joint_pos = np.array(  # type: ignore
            list(robot.default_joint_angles.values()), dtype=np.float32
        )
        self.default_joint_vel = np.zeros_like(self.default_joint_pos)  # type: ignore

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

        self.left_hip_pitch_idx = self.robot.joint_ordering.index("left_hip_pitch")
        self.left_knee_pitch_idx = self.robot.joint_ordering.index("left_knee_pitch")
        self.left_ank_pitch_idx = self.robot.joint_ordering.index("left_ank_pitch")
        self.right_hip_pitch_idx = self.robot.joint_ordering.index("right_hip_pitch")
        self.right_knee_pitch_idx = self.robot.joint_ordering.index("right_knee_pitch")
        self.right_ank_pitch_idx = self.robot.joint_ordering.index("right_ank_pitch")

    def get_phase_signal(
        self, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        phase_signal = np.array(  # type:ignore
            [
                np.sin(2 * np.pi * time_curr / self.cycle_time),  # type:ignore
                np.cos(2 * np.pi * time_curr / self.cycle_time),  # type:ignore
            ],
            dtype=np.float32,
        )
        return phase_signal

    def get_state_ref(
        self,
        path_pos: ArrayType,
        path_quat: ArrayType,
        time_curr: Optional[float | ArrayType] = None,
        command: Optional[ArrayType] = None,
    ) -> ArrayType:
        if time_curr is None:
            raise ValueError(f"time_curr is required for {self.name}")

        if command is None:
            raise ValueError(f"command is required for {self.name}")

        sin_phase_signal = np.sin(2 * np.pi * time_curr / self.cycle_time)  # type: ignore
        signal_left = np.clip(sin_phase_signal, 0, None)  # type: ignore
        signal_right = np.clip(sin_phase_signal, None, 0)  # type: ignore

        linear_vel = np.array([command[0], command[1], 0.0], dtype=np.float32)  # type: ignore
        angular_vel = np.array([0.0, 0.0, command[2]], dtype=np.float32)  # type: ignore

        joint_pos = self.default_joint_pos.copy()  # type: ignore

        left_leg_angles = self.calculate_leg_angles(signal_left, True)  # type: ignore
        right_leg_angles = self.calculate_leg_angles(signal_right, False)  # type: ignore
        leg_angles = {**left_leg_angles, **right_leg_angles}

        for idx, angle in leg_angles.items():
            joint_pos = inplace_update(joint_pos, idx, angle)

        double_support_mask = np.abs(sin_phase_signal) < self.double_support_phase  # type: ignore
        joint_pos = np.where(double_support_mask, self.default_joint_pos, joint_pos)  # type: ignore

        joint_vel = self.default_joint_vel.copy()  # type: ignore

        stance_mask = np.zeros(2, dtype=np.float32)  # type: ignore
        stance_mask = inplace_update(stance_mask, 0, np.any(sin_phase_signal >= 0))  # type: ignore
        stance_mask = inplace_update(stance_mask, 1, np.any(sin_phase_signal < 0))  # type: ignore
        stance_mask = np.where(double_support_mask, 1, stance_mask)  # type: ignore

        return np.concatenate(  # type: ignore
            (
                path_pos,
                path_quat,
                linear_vel,
                angular_vel,
                joint_pos,
                joint_vel,
                stance_mask,
            )  # type: ignore
        )

    def calculate_leg_angles(self, signal: ArrayType, is_left: bool):
        knee_angle = np.abs(  # type: ignore
            signal * (self.max_knee_pitch - self.knee_pitch_default)
            + (2 * int(is_left) - 1) * self.knee_pitch_default
        )
        ank_pitch_angle = np.arctan2(  # type: ignore
            np.sin(knee_angle),  # type: ignore
            np.cos(knee_angle) + self.shin_thigh_ratio,  # type: ignore
        )
        hip_pitch_angle = knee_angle - ank_pitch_angle

        if is_left:
            return {
                self.left_hip_pitch_idx: -hip_pitch_angle,
                self.left_knee_pitch_idx: knee_angle,
                self.left_ank_pitch_idx: -ank_pitch_angle,
            }
        else:
            return {
                self.right_hip_pitch_idx: hip_pitch_angle,
                self.right_knee_pitch_idx: -knee_angle,
                self.right_ank_pitch_idx: -ank_pitch_angle,
            }

    def override_motor_target(
        self, motor_target: ArrayType, state_ref: ArrayType
    ) -> ArrayType:
        motor_target = inplace_update(
            motor_target,
            self.neck_actuator_indices,  # type: ignore
            self.default_motor_pos[self.neck_actuator_indices],
        )
        motor_target = inplace_update(
            motor_target,
            self.arm_actuator_indices,  # type: ignore
            self.default_motor_pos[self.arm_actuator_indices],
        )

        return motor_target
