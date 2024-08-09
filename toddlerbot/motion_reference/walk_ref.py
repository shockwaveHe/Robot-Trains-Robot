from typing import Any, Optional

import jax
import numpy as np
import numpy.typing as npt
from jax import numpy as jnp

from toddlerbot.motion_reference.motion_ref import MotionReference
from toddlerbot.sim.robot import Robot


class WalkReference(MotionReference):
    def __init__(
        self,
        robot: Robot,
        use_jax: bool = False,
        joint_pos_ref_scale: float = 1.0,
        double_support_phase: float = 0.1,
    ):
        super().__init__("periodic", robot, use_jax)

        self.num_joints = len(self.robot.joint_ordering)
        self.joint_pos_ref_scale = joint_pos_ref_scale
        self.double_support_phase = double_support_phase
        self.shin_thigh_ratio = (
            self.robot.data_dict["offsets"]["knee_to_ank_pitch_z"]
            / self.robot.data_dict["offsets"]["hip_pitch_to_knee_z"]
        )

    def get_state_ref(
        self,
        path_frame: npt.NDArray[np.float32] | jax.Array,
        phase: Optional[float | npt.NDArray[np.float32] | jax.Array] = None,
        command: Optional[npt.NDArray[np.float32] | jax.Array] = None,
    ) -> npt.NDArray[np.float32] | jax.Array:
        if phase is None:
            raise ValueError(f"phase is required for {self.motion_type} motion")

        if command is None:
            raise ValueError(f"command is required for {self.motion_type} motion")

        backend = jnp if self.use_jax else np

        pos = path_frame[:3]
        quat = path_frame[3:]

        linear_vel = backend.array([command[0], command[1], 0.0], dtype=backend.float32)  # type: ignore
        angular_vel = backend.array([0.0, 0.0, command[2]], dtype=backend.float32)  # type: ignore

        sin_phase_signal = backend.sin(2 * backend.pi * phase)  # type: ignore
        signal_left = backend.clip(sin_phase_signal, 0, None)  # type: ignore
        signal_right = backend.clip(sin_phase_signal, None, 0)  # type: ignore

        joint_pos = backend.zeros(self.num_joints, dtype=backend.float32)  # type: ignore
        joint_vel = backend.zeros(self.num_joints, dtype=backend.float32)  # type: ignore

        left_leg_angles = self.calculate_leg_angles(signal_left, True, backend)  # type: ignore
        right_leg_angles = self.calculate_leg_angles(signal_right, False, backend)  # type: ignore

        leg_angles = {**left_leg_angles, **right_leg_angles}

        for name, angle in leg_angles.items():
            joint_idx = self.get_joint_idx(name)
            if self.use_jax:
                joint_pos = joint_pos.at[joint_idx].set(angle)  # type: ignore
            else:
                joint_pos[joint_idx] = angle

        double_support_mask = backend.abs(sin_phase_signal) < self.double_support_phase  # type: ignore
        stance_mask = backend.zeros(2, dtype=backend.float32)  # type: ignore

        if self.use_jax:
            joint_pos = backend.where(double_support_mask, 0, joint_pos)  # type: ignore
            stance_mask = stance_mask.at[0].set(backend.any(sin_phase_signal >= 0))  # type: ignore
            stance_mask = stance_mask.at[1].set(backend.any(sin_phase_signal < 0))  # type: ignore
            stance_mask = backend.where(double_support_mask, 1, stance_mask)  # type: ignore
        else:
            joint_pos[double_support_mask] = 0
            stance_mask[0] = backend.any(sin_phase_signal >= 0)  # type: ignore
            stance_mask[1] = backend.any(sin_phase_signal < 0)  # type: ignore
            stance_mask[double_support_mask] = 1

        return backend.concatenate(  # type: ignore
            (pos, quat, linear_vel, angular_vel, joint_pos, joint_vel, stance_mask)  # type: ignore
        )

    def calculate_leg_angles(
        self, signal: npt.NDArray[np.float32] | jax.Array, is_left: bool, backend: Any
    ):
        knee_angle = backend.abs(signal * self.joint_pos_ref_scale)
        ank_pitch_angle = backend.arctan2(
            backend.sin(knee_angle), backend.cos(knee_angle) + self.shin_thigh_ratio
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
