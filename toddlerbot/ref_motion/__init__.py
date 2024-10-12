import os
from abc import ABC, abstractmethod
from typing import Tuple

import numpy

from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType
from toddlerbot.utils.array_utils import array_lib as np
from toddlerbot.utils.math_utils import quat_mult


class MotionReference(ABC):
    def __init__(self, name: str, motion_type: str, robot: Robot, dt: float):
        self.name = name
        self.motion_type = motion_type
        self.robot = robot
        self.dt = dt
        self.use_jax = os.environ.get("USE_JAX", "false") == "true"

        self.default_joint_pos = np.array(
            list(robot.default_joint_angles.values()), dtype=np.float32
        )
        self.default_joint_vel = np.zeros_like(self.default_joint_pos)

        self.default_motor_pos = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )

        indices = np.arange(robot.nu)
        motor_groups = numpy.array(
            [robot.joint_groups[name] for name in robot.motor_ordering]
        )
        joint_groups = numpy.array(
            [robot.joint_groups[name] for name in robot.joint_ordering]
        )
        self.leg_actuator_indices = indices[motor_groups == "leg"]
        self.leg_joint_indices = indices[joint_groups == "leg"]
        self.arm_actuator_indices = indices[motor_groups == "arm"]
        self.arm_joint_indices = indices[joint_groups == "arm"]
        self.neck_actuator_indices = indices[motor_groups == "neck"]
        self.neck_joint_indices = indices[joint_groups == "neck"]
        self.waist_actuator_indices = indices[motor_groups == "waist"]
        self.waist_joint_indices = indices[joint_groups == "waist"]

    @abstractmethod
    def get_phase_signal(self, time_curr: float | ArrayType) -> ArrayType:
        pass

    @abstractmethod
    def get_vel(self, command: ArrayType) -> Tuple[ArrayType, ArrayType]:
        pass

    def integrate_torso_state(
        self, torso_pos: ArrayType, torso_quat: ArrayType, command: ArrayType
    ) -> Tuple[ArrayType, ArrayType]:
        lin_vel, ang_vel = self.get_vel(command)

        # Update position
        torso_pos += lin_vel * self.dt

        # Compute the angle of rotation for each axis
        theta_roll = ang_vel[0] * self.dt / 2.0
        theta_pitch = ang_vel[1] * self.dt / 2.0
        theta_yaw = ang_vel[2] * self.dt / 2.0

        # Compute the quaternion for each rotational axis
        roll_quat = np.array([np.cos(theta_roll), np.sin(theta_roll), 0.0, 0.0])
        pitch_quat = np.array([np.cos(theta_pitch), 0.0, np.sin(theta_pitch), 0.0])
        yaw_quat = np.array([np.cos(theta_yaw), 0.0, 0.0, np.sin(theta_yaw)])

        # Normalize each quaternion
        roll_quat /= np.linalg.norm(roll_quat)
        pitch_quat /= np.linalg.norm(pitch_quat)
        yaw_quat /= np.linalg.norm(yaw_quat)

        # Combine the quaternions to get the full rotation (roll * pitch * yaw)
        full_quat = quat_mult(quat_mult(roll_quat, pitch_quat), yaw_quat)

        # Update the current quaternion by applying the new rotation
        torso_quat = quat_mult(torso_quat, full_quat)
        torso_quat /= np.linalg.norm(torso_quat)

        return np.concatenate([torso_pos, torso_quat, lin_vel, ang_vel])

    @abstractmethod
    def get_state_ref(
        self, state_curr: ArrayType, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        pass

    @abstractmethod
    def override_motor_target(
        self, motor_target: ArrayType, state_ref: ArrayType
    ) -> ArrayType:
        pass

    def neck_ik(self, neck_joint_pos: ArrayType) -> ArrayType:
        neck_motor_pos = neck_joint_pos / self.neck_gear_ratio
        return neck_motor_pos

    def arm_fk(self, arm_motor_pos: ArrayType) -> ArrayType:
        arm_joint_pos = arm_motor_pos * self.arm_gear_ratio
        return arm_joint_pos

    def arm_ik(self, arm_joint_pos: ArrayType) -> ArrayType:
        arm_motor_pos = arm_joint_pos / self.arm_gear_ratio
        return arm_motor_pos

    def waist_ik(self, waist_joint_pos: ArrayType) -> ArrayType:
        waist_roll, waist_yaw = waist_joint_pos / self.waist_coef
        waist_act_1 = (-waist_roll + waist_yaw) / 2
        waist_act_2 = (waist_roll + waist_yaw) / 2
        return np.array([waist_act_1, waist_act_2], dtype=np.float32)

    def leg_fk(self, knee_angle: float | ArrayType) -> ArrayType:
        # Compute the length from hip pitch to ankle pitch along the z-axis
        com_z_target = np.array(
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
        return com_z_target

    def leg_ik(self, com_z_target: ArrayType) -> ArrayType:
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
