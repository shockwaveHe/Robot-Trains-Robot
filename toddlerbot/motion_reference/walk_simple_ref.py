from typing import Any, Optional

import jax
import numpy as np
import numpy.typing as npt
from jax import numpy as jnp

from toddlerbot.motion_reference.motion_ref import MotionReference
from toddlerbot.sim.robot import Robot


class WalkSimpleReference(MotionReference):
    def __init__(
        self,
        robot: Robot,
        use_jax: bool = False,
        default_joint_pos: Optional[npt.NDArray[np.float32] | jax.Array] = None,
        default_joint_vel: Optional[npt.NDArray[np.float32] | jax.Array] = None,
        max_knee_pitch: float = np.pi / 3,
        double_support_phase: float = 0.1,
    ):
        super().__init__("walk", "periodic", robot, use_jax)

        self.default_joint_pos = default_joint_pos
        self.default_joint_vel = default_joint_vel
        if self.default_joint_pos is None:
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
        path_pos: npt.NDArray[np.float32] | jax.Array,
        path_quat: npt.NDArray[np.float32] | jax.Array,
        phase: Optional[float | npt.NDArray[np.float32] | jax.Array] = None,
        command: Optional[npt.NDArray[np.float32] | jax.Array] = None,
    ) -> npt.NDArray[np.float32] | jax.Array:
        if phase is None:
            raise ValueError(f"phase is required for {self.motion_type} motion")

        if command is None:
            raise ValueError(f"command is required for {self.motion_type} motion")

        backend = jnp if self.use_jax else np

        linear_vel = backend.array([command[0], command[1], 0.0], dtype=backend.float32)  # type: ignore
        angular_vel = backend.array([0.0, 0.0, command[2]], dtype=backend.float32)  # type: ignore

        sin_phase_signal = backend.sin(2 * backend.pi * phase)  # type: ignore
        signal_left = backend.clip(sin_phase_signal, 0, None)  # type: ignore
        signal_right = backend.clip(sin_phase_signal, None, 0)  # type: ignore

        if self.default_joint_pos is None:
            joint_pos = backend.zeros(self.num_joints, dtype=backend.float32)  # type: ignore
        else:
            joint_pos = self.default_joint_pos.copy()  # type: ignore

        if self.default_joint_vel is None:
            joint_vel = backend.zeros(self.num_joints, dtype=backend.float32)  # type: ignore
        else:
            joint_vel = self.default_joint_vel.copy()  # type: ignore

        left_leg_angles = self.calculate_leg_angles(signal_left, True, backend)  # type: ignore
        right_leg_angles = self.calculate_leg_angles(signal_right, False, backend)  # type: ignore

        leg_angles = {**left_leg_angles, **right_leg_angles}

        if self.use_jax:
            indices = jnp.array([self.get_joint_idx(name) for name in leg_angles])  # type: ignore
            angles = jnp.array(list(leg_angles.values()))  # type: ignore
            joint_pos = joint_pos.at[indices].set(angles)  # type: ignore
        else:
            for name, angle in leg_angles.items():
                joint_pos[self.get_joint_idx(name)] = angle

        double_support_mask = backend.abs(sin_phase_signal) < self.double_support_phase  # type: ignore
        joint_pos = backend.where(  # type: ignore
            double_support_mask, self.default_joint_pos, joint_pos
        )

        stance_mask = backend.zeros(2, dtype=backend.float32)  # type: ignore
        if self.use_jax:
            stance_mask = stance_mask.at[0].set(backend.any(sin_phase_signal >= 0))  # type: ignore
            stance_mask = stance_mask.at[1].set(backend.any(sin_phase_signal < 0))  # type: ignore
            stance_mask = backend.where(double_support_mask, 1, stance_mask)  # type: ignore
        else:
            stance_mask[0] = backend.any(sin_phase_signal >= 0)  # type: ignore
            stance_mask[1] = backend.any(sin_phase_signal < 0)  # type: ignore
            stance_mask[double_support_mask] = 1

        return backend.concatenate(  # type: ignore
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

    def calculate_leg_angles(
        self, signal: npt.NDArray[np.float32] | jax.Array, is_left: bool, backend: Any
    ):
        knee_angle = backend.abs(
            signal * (self.max_knee_pitch - self.knee_pitch_default)
            + (2 * int(is_left) - 1) * self.knee_pitch_default
        )
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


if __name__ == "__main__":
    from toddlerbot.utils.math_utils import round_to_sig_digits

    robot = Robot("toddlerbot")
    walk_ref = WalkSimpleReference(robot, max_knee_pitch=0.523599)
    left_leg_angles = walk_ref.calculate_leg_angles(
        np.ones(1, dtype=np.float32), True, np
    )
    left_ank_act = robot.ankle_ik([0.0, left_leg_angles["left_ank_pitch"].item()])
    print(left_leg_angles)
    print([round_to_sig_digits(x, 6) for x in left_ank_act])
