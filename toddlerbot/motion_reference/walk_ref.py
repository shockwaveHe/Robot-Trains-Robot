from typing import Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.motion_reference.motion_ref import MotionReference
from toddlerbot.sim.robot import Robot


class WalkReference(MotionReference):
    def __init__(
        self,
        robot: Robot,
        joint_pos_ref_scale: float = 1.0,
        double_support_phase: float = 0.1,
    ):
        super().__init__("periodic", robot)

        self.joint_pos_ref_scale = joint_pos_ref_scale
        self.double_support_phase = double_support_phase
        self.shin_thigh_ratio = (
            self.robot.data_dict["offsets"]["knee_to_ank_pitch_z"]
            / self.robot.data_dict["offsets"]["hip_pitch_to_knee_z"]
        )

    def get_state(
        self,
        path_frame: npt.NDArray[np.float32],
        phase: Optional[float] = None,
        command: Optional[npt.NDArray[np.float32]] = None,
    ) -> npt.NDArray[np.float32]:
        if phase is None:
            raise ValueError(f"phase is required for {self.motion_type} motion")

        if command is None:
            raise ValueError(f"command is required for {self.motion_type} motion")

        pos = path_frame[:3]
        quat = path_frame[3:]

        linear_vel = command[:3]
        angular_vel = command[3:]

        sin_phase_signal = np.sin(2 * np.pi * phase)
        # When sin_phase_signal < 0, left foot is in stance phase
        # When sin_phase_signal > 0, right foot is in stance phase
        signal_left = np.clip(sin_phase_signal, 0, None)  # type: ignore
        signal_right = np.clip(sin_phase_signal, None, 0)  # type: ignore

        # Initialize joint positions and velocities
        num_joints = len(self.robot.joint_ordering)
        joint_pos = np.zeros(num_joints, dtype=np.float32)
        joint_vel = np.zeros(num_joints, dtype=np.float32)

        # Define stance angles for each leg

        # Calculate joint angles for both legs
        left_hip_pitch, left_knee_pitch, left_ank_pitch = self.calculate_leg_angles(
            signal_left, is_left=True
        )
        right_hip_pitch, right_knee_pitch, right_ank_pitch = self.calculate_leg_angles(
            signal_right, is_left=False
        )

        # Set joint positions
        joint_pos[self.get_joint_idx("left_hip_pitch")] = left_hip_pitch
        joint_pos[self.get_joint_idx("left_knee_pitch")] = left_knee_pitch
        joint_pos[self.get_joint_idx("left_ank_pitch")] = left_ank_pitch

        joint_pos[self.get_joint_idx("right_hip_pitch")] = right_hip_pitch
        joint_pos[self.get_joint_idx("right_knee_pitch")] = right_knee_pitch
        joint_pos[self.get_joint_idx("right_ank_pitch")] = right_ank_pitch

        # Double support phase handling
        double_support_mask = np.abs(sin_phase_signal) < self.double_support_phase
        joint_pos[double_support_mask] = 0

        # Stance mask for determining support leg
        stance_mask = np.zeros(2, dtype=np.float32)
        stance_mask[0] = np.any(sin_phase_signal >= 0)  # type: ignore
        stance_mask[1] = np.any(sin_phase_signal < 0)  # type: ignore
        stance_mask[double_support_mask] = 1

        return np.concatenate(  # type: ignore
            (pos, quat, linear_vel, angular_vel, joint_pos, joint_vel, stance_mask)
        )

    def calculate_leg_angles(
        self, signal: npt.NDArray[np.float32], is_left: bool = True
    ):
        knee_angle = np.abs(signal * self.joint_pos_ref_scale)
        ank_pitch_angle = np.arctan2(
            np.sin(knee_angle), np.cos(knee_angle) + self.shin_thigh_ratio
        )
        hip_pitch_angle = knee_angle - ank_pitch_angle

        if is_left:
            return -hip_pitch_angle, knee_angle, -ank_pitch_angle
        else:
            return hip_pitch_angle, -knee_angle, -ank_pitch_angle
