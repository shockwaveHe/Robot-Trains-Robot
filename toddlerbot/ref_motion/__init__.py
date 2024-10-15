import os
from abc import ABC, abstractmethod
from typing import List, Tuple

import joblib
import numpy

from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
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

        self._setup_neck()
        self._setup_arm()
        self._setup_waist()

    def _get_gear_ratios(self, motor_names: List[str]) -> ArrayType:
        gear_ratios = np.ones(len(motor_names), dtype=np.float32)
        for i, motor_name in enumerate(motor_names):
            motor_config = self.robot.config["joints"][motor_name]
            if motor_config["transmission"] in ["gear", "rack_and_pinion"]:
                gear_ratios = inplace_update(
                    gear_ratios, i, -motor_config["gear_ratio"]
                )
        return gear_ratios

    def _setup_neck(self):
        neck_motor_names = [
            self.robot.motor_ordering[i] for i in self.neck_actuator_indices
        ]
        self.neck_gear_ratio = self._get_gear_ratios(neck_motor_names)
        self.neck_joint_limits = np.array(
            [
                self.robot.joint_limits["neck_yaw_driven"],
                self.robot.joint_limits["neck_pitch_driven"],
            ],
            dtype=np.float32,
        ).T

    def _setup_arm(self):
        arm_motor_names = [
            self.robot.motor_ordering[i] for i in self.arm_actuator_indices
        ]
        self.arm_gear_ratio = self._get_gear_ratios(arm_motor_names)

        # Load the balance dataset
        data_path = os.path.join("toddlerbot", "ref_motion", "balance_dataset.lz4")
        data_dict = joblib.load(data_path)
        # state_array: [time(1), motor_pos(14), fsrL(1), fsrR(1), camera_frame_idx(1)]
        state_arr = data_dict["state_array"]
        self.arm_time_ref = np.array(
            state_arr[:, 0] - state_arr[0, 0], dtype=np.float32
        )
        self.arm_joint_pos_ref = np.array(
            [
                self.arm_fk(arm_motor_pos)
                for arm_motor_pos in state_arr[:, 1 + self.arm_actuator_indices]
            ],
            dtype=np.float32,
        )
        self.arm_ref_size = len(self.arm_time_ref)

    def _setup_waist(self):
        self.waist_coef = np.array(
            [
                self.robot.config["general"]["offsets"]["waist_roll_coef"],
                self.robot.config["general"]["offsets"]["waist_yaw_coef"],
            ],
            dtype=np.float32,
        )
        self.waist_joint_limits = np.array(
            [
                self.robot.joint_limits["waist_roll"],
                self.robot.joint_limits["waist_yaw"],
            ],
            dtype=np.float32,
        ).T

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
