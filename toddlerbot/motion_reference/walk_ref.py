from typing import Optional

import jax.numpy as jnp
from jax import jit  # type: ignore

from toddlerbot.motion_reference.motion_ref import MotionReference
from toddlerbot.sim.robot import Robot


class WalkReference(MotionReference):
    def __init__(self, robot: Robot, joint_pos_ref_scale: float):
        super().__init__("periodic", robot)

        self.joint_pos_ref_scale = joint_pos_ref_scale
        self.shin_thigh_ratio = (
            self.robot.data_dict["offsets"]["knee_to_ank_pitch_z"]
            / self.robot.data_dict["offsets"]["hip_pitch_to_knee_z"]
        )
        self.double_support_phase = 0.1

    @jit
    def get_state(
        self,
        path_frame: jnp.ndarray,
        phase: Optional[float] = None,
        command: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        if phase is None:
            raise ValueError(f"phase is required for {self.motion_type} motion")

        if command is None:
            raise ValueError(f"command is required for {self.motion_type} motion")

        pos = path_frame[:3]
        quat = path_frame[3:]

        linear_vel = command[:3]
        angular_vel = command[3:]

        sin_phase_signal = jnp.sin(2 * jnp.pi * phase)  # type: ignore
        # When sin_phase_signal < 0, left foot is in stance phase
        # When sin_phase_signal > 0, right foot is in stance phase
        signal_left = jnp.clip(sin_phase_signal, 0, None)  # type: ignore
        signal_right = jnp.clip(sin_phase_signal, None, 0)  # type: ignore

        num_joints = len(self.robot.joint_ordering)
        joint_pos = jnp.zeros(num_joints)  # type: ignore
        joint_vel = jnp.zeros(num_joints)  # type: ignore

        # Calculate joint angles
        left_hip_pitch, left_knee_pitch, left_ank_pitch = self.calculate_leg_angles(
            signal_left, is_left=True
        )
        right_hip_pitch, right_knee_pitch, right_ank_pitch = self.calculate_leg_angles(
            signal_right, is_left=False
        )
        joint_pos = joint_pos.at[self.get_joint_idx("left_hip_pitch")].set(  # type: ignore
            left_hip_pitch
        )
        joint_pos = joint_pos.at[self.get_joint_idx("left_knee_pitch")].set(  # type: ignore
            left_knee_pitch
        )
        joint_pos = joint_pos.at[self.get_joint_idx("left_ank_pitch")].set(  # type: ignore
            left_ank_pitch
        )
        joint_pos = joint_pos.at[self.get_joint_idx("right_hip_pitch")].set(  # type: ignore
            right_hip_pitch
        )
        joint_pos = joint_pos.at[self.get_joint_idx("right_knee_pitch")].set(  # type: ignore
            right_knee_pitch
        )
        joint_pos = joint_pos.at[self.get_joint_idx("right_ank_pitch")].set(  # type: ignore
            right_ank_pitch
        )

        # Double support phase handling
        double_support_mask = jnp.abs(sin_phase_signal) < self.double_support_phase  # type: ignore
        joint_pos = jnp.where(double_support_mask, 0, joint_pos)  # type: ignore

        # Stance mask for determining support leg
        stance_mask = jnp.zeros(2, dtype=jnp.float32)  # type: ignore
        stance_mask = stance_mask.at[0].set(jnp.any(sin_phase_signal >= 0))  # type: ignore
        stance_mask = stance_mask.at[1].set(jnp.any(sin_phase_signal < 0))  # type: ignore
        stance_mask = jnp.where(double_support_mask, 1, stance_mask)  # type: ignore

        return jnp.concatenate(  # type: ignore
            (pos, quat, linear_vel, angular_vel, joint_pos, joint_vel, stance_mask)
        )

    def calculate_leg_angles(self, signal: jnp.ndarray, is_left: bool = True):
        knee_angle = jnp.abs(signal * self.joint_pos_ref_scale)  # type: ignore
        ank_pitch_angle = jnp.arctan2(  # type: ignore
            jnp.sin(knee_angle),  # type: ignore
            jnp.cos(knee_angle) + self.shin_thigh_ratio,  # type: ignore
        )
        hip_pitch_angle = knee_angle - ank_pitch_angle

        if is_left:
            return -hip_pitch_angle, knee_angle, -ank_pitch_angle
        else:
            return hip_pitch_angle, -knee_angle, -ank_pitch_angle
