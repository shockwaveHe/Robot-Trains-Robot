from typing import Optional, Tuple

from toddlerbot.ref_motion import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np


class WalkSimpleReference(MotionReference):
    def __init__(
        self,
        robot: Robot,
        cycle_time: float,
        default_joint_pos: Optional[ArrayType] = None,
        default_joint_vel: Optional[ArrayType] = None,
        max_knee_pitch: float = np.pi / 3,
        double_support_phase: float = 0.1,
    ):
        super().__init__("walk_simple", "periodic", robot)

        self.cycle_time = cycle_time

        self.default_joint_pos = default_joint_pos
        self.default_joint_vel = default_joint_vel
        if self.default_joint_pos is None:
            self.default_joint_pos = np.zeros(  # type: ignore
                len(self.robot.joint_ordering), dtype=np.float32
            )
            self.knee_pitch_default = 0.0
        else:
            self.knee_pitch_default = self.default_joint_pos[
                self.get_joint_idx("left_knee_pitch")
            ]
        self.max_knee_pitch = max_knee_pitch
        self.double_support_phase = double_support_phase

        self.num_joints = len(self.robot.joint_ordering)
        self.shin_thigh_ratio = (
            self.robot.data_dict["offsets"]["knee_to_ank_pitch_z"]
            / self.robot.data_dict["offsets"]["hip_pitch_to_knee_z"]
        )

    def get_state_ref(
        self,
        path_pos: ArrayType,
        path_quat: ArrayType,
        time_curr: Optional[float | ArrayType] = None,
        command: Optional[ArrayType] = None,
    ) -> Tuple[ArrayType, ArrayType]:
        if time_curr is None:
            raise ValueError(f"time_curr is required for {self.name}")

        if command is None:
            raise ValueError(f"command is required for {self.name}")

        phase_signal = np.array(  # type:ignore
            [
                np.sin(2 * np.pi * time_curr / self.cycle_time),  # type:ignore
                np.cos(2 * np.pi * time_curr / self.cycle_time),  # type:ignore
            ],
            dtype=np.float32,
        )
        sin_phase_signal = np.sin(2 * np.pi * time_curr / self.cycle_time)  # type: ignore
        signal_left = np.clip(sin_phase_signal, 0, None)  # type: ignore
        signal_right = np.clip(sin_phase_signal, None, 0)  # type: ignore

        linear_vel = np.array([command[0], command[1], 0.0], dtype=np.float32)  # type: ignore
        angular_vel = np.array([0.0, 0.0, command[2]], dtype=np.float32)  # type: ignore

        assert self.default_joint_pos is not None
        joint_pos = self.default_joint_pos.copy()  # type: ignore

        if self.default_joint_vel is None:
            joint_vel = np.zeros(self.num_joints, dtype=np.float32)  # type: ignore
        else:
            joint_vel = self.default_joint_vel.copy()  # type: ignore

        left_leg_angles = self.calculate_leg_angles(signal_left, True)
        right_leg_angles = self.calculate_leg_angles(signal_right, False)

        leg_angles = {**left_leg_angles, **right_leg_angles}

        for name, angle in leg_angles.items():
            joint_pos = inplace_update(joint_pos, self.get_joint_idx(name), angle)

        double_support_mask = np.abs(sin_phase_signal) < self.double_support_phase  # type: ignore
        joint_pos = np.where(  # type: ignore
            double_support_mask, self.default_joint_pos, joint_pos
        )

        stance_mask = np.zeros(2, dtype=np.float32)  # type: ignore
        stance_mask = inplace_update(stance_mask, 0, np.any(sin_phase_signal >= 0))  # type: ignore
        stance_mask = inplace_update(stance_mask, 1, np.any(sin_phase_signal < 0))  # type: ignore
        stance_mask = np.where(double_support_mask, 1, stance_mask)  # type: ignore

        return phase_signal, np.concatenate(  # type: ignore
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
