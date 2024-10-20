import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import joblib
import mujoco
import numpy
from mujoco import mjx

from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.math_utils import euler2quat, quat_mult, rotate_vec


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
        self.leg_motor_indices = indices[motor_groups == "leg"]
        self.leg_joint_indices = indices[joint_groups == "leg"]
        self.arm_motor_indices = indices[motor_groups == "arm"]
        self.arm_joint_indices = indices[joint_groups == "arm"]
        self.neck_motor_indices = indices[motor_groups == "neck"]
        self.neck_joint_indices = indices[joint_groups == "neck"]
        self.waist_motor_indices = indices[motor_groups == "waist"]
        self.waist_joint_indices = indices[joint_groups == "waist"]

        self._setup_neck()
        self._setup_arm()
        self._setup_waist()
        self._setup_leg()
        self._setup_com()
        self._setup_mjx()

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
            self.robot.motor_ordering[i] for i in self.neck_motor_indices
        ]
        self.neck_pitch_idx = self.robot.joint_ordering.index("neck_pitch")
        self.neck_gear_ratio = self._get_gear_ratios(neck_motor_names)
        self.neck_joint_limits = np.array(
            [
                self.robot.joint_limits["neck_yaw_driven"],
                self.robot.joint_limits["neck_pitch"],
            ],
            dtype=np.float32,
        ).T

    def _setup_arm(self):
        arm_motor_names = [self.robot.motor_ordering[i] for i in self.arm_motor_indices]
        self.arm_gear_ratio = self._get_gear_ratios(arm_motor_names)

        # Load the balance dataset
        data_path = os.path.join("toddlerbot", "motion", "balance_dataset.lz4")
        data_dict = joblib.load(data_path)
        time_arr = data_dict["time"]
        motor_pos_arr = data_dict["motor_pos"]
        self.arm_time_ref = np.array(time_arr - time_arr[0], dtype=np.float32)
        self.arm_joint_pos_ref = np.array(
            [
                self.arm_fk(arm_motor_pos)
                for arm_motor_pos in motor_pos_arr[:, self.arm_motor_indices]
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

    def _setup_leg(self):
        leg_motor_names = [self.robot.motor_ordering[i] for i in self.leg_motor_indices]
        self.leg_gear_ratio = self._get_gear_ratios(leg_motor_names)
        self.left_knee_idx = self.robot.joint_ordering.index("left_knee")
        self.left_hip_pitch_idx = self.robot.joint_ordering.index("left_hip_pitch")
        self.left_hip_roll_idx = self.robot.joint_ordering.index("left_hip_roll")
        self.right_knee_idx = self.robot.joint_ordering.index("right_knee")
        self.right_hip_pitch_idx = self.robot.joint_ordering.index("right_hip_pitch")
        self.right_hip_roll_idx = self.robot.joint_ordering.index("right_hip_roll")

    def _setup_com(self, com_z_lower_limit_offset: float = 0.01):
        self.knee_default = self.default_joint_pos[self.left_knee_idx]
        self.hip_pitch_to_knee = self.robot.data_dict["offsets"]["hip_pitch_to_knee_z"]
        self.knee_to_ank_pitch = self.robot.data_dict["offsets"]["knee_to_ank_pitch_z"]
        self.hip_pitch_to_ank_pitch_z = np.sqrt(
            self.hip_pitch_to_knee**2
            + self.knee_to_ank_pitch**2
            - 2
            * self.hip_pitch_to_knee
            * self.knee_to_ank_pitch
            * np.cos(np.pi - self.knee_default)
        )
        self.shin_thigh_ratio = self.knee_to_ank_pitch / self.hip_pitch_to_knee

        self.com_z_limits = np.array(
            [
                self.com_fk(self.robot.joint_limits["left_knee"][1])[2]
                + com_z_lower_limit_offset,
                0.0,
            ],
            dtype=np.float32,
        )

    def _setup_mjx(self):
        xml_path = find_robot_file_path(self.robot.name, suffix="_scene.xml")
        model = mujoco.MjModel.from_xml_path(xml_path)
        # self.renderer = mujoco.Renderer(model)
        self.default_qpos = np.array(model.keyframe("home").qpos)
        self.mj_joint_indices = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.joint_ordering
            ]
        )
        self.mj_motor_indices = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.motor_ordering
            ]
        )
        self.mj_passive_indices = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.passive_joint_names
            ]
        )
        # Account for the free joint
        self.mj_joint_indices -= 1
        self.mj_motor_indices -= 1
        self.mj_passive_indices -= 1

        self.left_foot_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "left_foot_center"
        )
        self.right_foot_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "right_foot_center"
        )
        self.passive_joint_indices = np.array(
            [self.neck_pitch_idx, self.left_knee_idx, self.right_knee_idx]
        )

        if self.use_jax:
            self.model = mjx.put_model(model)

            def forward(qpos):
                data = mjx.make_data(self.model)
                data = data.replace(qpos=qpos)
                return mjx.forward(self.model, data)

        else:
            self.model = model

            def forward(qpos):
                data = mujoco.MjData(self.model)
                data.qpos = qpos
                mujoco.mj_forward(self.model, data)
                return data

        self.forward = forward

        data = self.forward(self.default_qpos)
        self.feet_center_init = (
            data.site_xpos[self.left_foot_site_id]
            + data.site_xpos[self.right_foot_site_id]
        ) / 2.0
        self.desired_com = (
            self.feet_center_init
        )  # np.array(data.subtree_com[0], dtype=np.float32)

    def get_phase_signal(self, time_curr: float | ArrayType) -> ArrayType:
        return np.zeros(1, dtype=np.float32)

    @abstractmethod
    def get_vel(self, command: ArrayType) -> Tuple[ArrayType, ArrayType]:
        pass

    def integrate_torso_state(
        self, state_curr: ArrayType, command: ArrayType
    ) -> ArrayType:
        lin_vel, ang_vel = self.get_vel(command)

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
        torso_quat = quat_mult(state_curr[3:7], full_quat)
        torso_quat /= np.linalg.norm(torso_quat)

        waist_joint_pos = state_curr[13 + self.waist_joint_indices]
        waist_euler_inv = np.array([waist_joint_pos[0], 0.0, waist_joint_pos[1]])
        waist_quat_inv = euler2quat(waist_euler_inv)
        path_quat = quat_mult(torso_quat, waist_quat_inv)

        # Update position
        torso_pos = state_curr[:3] + rotate_vec(lin_vel, path_quat) * self.dt

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

    def leg_ik(self, leg_joint_pos: ArrayType) -> ArrayType:
        leg_motor_pos = leg_joint_pos / self.leg_gear_ratio
        return leg_motor_pos

    def com_fk(
        self,
        knee_angle: float | ArrayType,
        hip_pitch_angle: Optional[float | ArrayType] = None,
        hip_roll_angle: Optional[float | ArrayType] = None,
    ) -> ArrayType:
        # Compute the length from hip pitch to ankle pitch along the z-axis
        hip_pitch_to_ank_pitch = np.sqrt(
            self.hip_pitch_to_knee**2
            + self.knee_to_ank_pitch**2
            - 2
            * self.hip_pitch_to_knee
            * self.knee_to_ank_pitch
            * np.cos(np.pi - knee_angle)
        )

        if hip_pitch_angle is None:
            alpha = 0.0
        else:
            alpha = (
                np.arcsin(
                    self.knee_to_ank_pitch / hip_pitch_to_ank_pitch * np.sin(knee_angle)
                )
                + hip_pitch_angle
            )

        if hip_roll_angle is None:
            hip_roll_angle = 0.0

        com_x = hip_pitch_to_ank_pitch * np.sin(alpha) * np.cos(hip_roll_angle)
        com_y = hip_pitch_to_ank_pitch * np.cos(alpha) * np.sin(hip_roll_angle)
        com_z = (
            hip_pitch_to_ank_pitch * np.cos(alpha) * np.cos(hip_roll_angle)
            - self.hip_pitch_to_ank_pitch_z
        )
        return np.array([com_x, com_y, com_z], dtype=np.float32)

    def com_ik(
        self,
        com_z: float | ArrayType,
        com_x: Optional[float | ArrayType] = None,
        com_y: Optional[float | ArrayType] = None,
    ) -> ArrayType:
        if com_x is None:
            com_x = 0.0
        if com_y is None:
            com_y = 0.0

        hip_pitch_to_ank_pitch = np.sqrt(
            com_x**2 + com_y**2 + (self.hip_pitch_to_ank_pitch_z + com_z) ** 2
        )

        knee_angle_cos = (
            self.hip_pitch_to_knee**2
            + self.knee_to_ank_pitch**2
            - hip_pitch_to_ank_pitch**2
        ) / (2 * self.hip_pitch_to_knee * self.knee_to_ank_pitch)
        knee_angle_cos = np.clip(knee_angle_cos, -1.0, 1.0)
        knee_angle = np.abs(np.pi - np.arccos(knee_angle_cos))

        alpha = np.arctan2(com_x, self.hip_pitch_to_ank_pitch_z + com_z)
        ank_pitch_angle = (
            np.arctan2(np.sin(knee_angle), np.cos(knee_angle) + self.shin_thigh_ratio)
            + alpha
        )
        hip_pitch_angle = knee_angle - ank_pitch_angle

        beta = np.arctan2(com_y, self.hip_pitch_to_ank_pitch_z + com_z)

        leg_joint_pos = np.array(
            [
                0.0,
                -beta,
                -hip_pitch_angle,
                knee_angle,
                -beta,
                -ank_pitch_angle,
                0.0,
                beta,
                hip_pitch_angle,
                -knee_angle,
                -beta,
                -ank_pitch_angle,
            ],
            dtype=np.float32,
        )

        return leg_joint_pos

    def get_qpos_ref(self, state_ref: ArrayType) -> ArrayType:
        qpos = self.default_qpos.copy()

        joint_pos_ref = state_ref[13 : 13 + self.robot.nu]
        qpos = inplace_update(qpos, 7 + self.mj_joint_indices, joint_pos_ref)

        neck_motor_pos = self.neck_ik(state_ref[13 + self.neck_joint_indices])
        waist_motor_pos = self.waist_ik(state_ref[13 + self.waist_joint_indices])
        leg_motor_pos = self.leg_ik(state_ref[13 + self.leg_joint_indices])
        arm_motor_pos = self.arm_ik(state_ref[13 + self.arm_joint_indices])
        motor_pos_ref = np.concatenate(
            [neck_motor_pos, waist_motor_pos, leg_motor_pos, arm_motor_pos]
        )
        qpos = inplace_update(qpos, 7 + self.mj_motor_indices, motor_pos_ref)

        passive_pos_ref = np.repeat(-state_ref[13 + self.passive_joint_indices], 4)
        qpos = inplace_update(qpos, 7 + self.mj_passive_indices, passive_pos_ref)

        waist_joint_pos = state_ref[13 + self.waist_joint_indices]
        waist_euler = np.array([-waist_joint_pos[0], 0.0, -waist_joint_pos[1]])
        waist_quat = euler2quat(waist_euler)
        torso_quat = quat_mult(state_ref[3:7], waist_quat)

        data = self.forward(qpos)

        feet_center = (
            data.site_xpos[self.left_foot_site_id]
            + data.site_xpos[self.right_foot_site_id]
        ) / 2.0
        torso_pos = np.asarray(state_ref[:3]) + self.feet_center_init - feet_center

        qpos = inplace_update(qpos, slice(0, 3), torso_pos)
        qpos = inplace_update(qpos, slice(3, 7), torso_quat)

        return qpos
