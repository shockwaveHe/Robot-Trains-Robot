import os
from typing import List, Tuple

import joblib

from toddlerbot.ref_motion import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np


class SquatReference(MotionReference):
    def __init__(self, robot: Robot, dt: float, com_z_lower_limit_offset: float = 0.01):
        super().__init__("squat", "perceptual", robot, dt)

        self._setup_neck()
        self._setup_arm()
        self._setup_waist()
        self._setup_leg(com_z_lower_limit_offset)

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

    def _setup_leg(self, com_z_lower_limit_offset: float):
        self.knee_pitch_default = self.default_joint_pos[
            self.robot.joint_ordering.index("left_knee_pitch")
        ]
        self.hip_pitch_to_knee_z = self.robot.data_dict["offsets"][
            "hip_pitch_to_knee_z"
        ]
        self.knee_to_ank_pitch_z = self.robot.data_dict["offsets"][
            "knee_to_ank_pitch_z"
        ]
        self.hip_pitch_to_ank_pitch_z = np.sqrt(
            self.hip_pitch_to_knee_z**2
            + self.knee_to_ank_pitch_z**2
            - 2
            * self.hip_pitch_to_knee_z
            * self.knee_to_ank_pitch_z
            * np.cos(np.pi - self.knee_pitch_default)
        )
        self.shin_thigh_ratio = self.knee_to_ank_pitch_z / self.hip_pitch_to_knee_z

        self.com_z_limits = np.array(
            [
                self.leg_fk(self.robot.joint_limits["left_knee_pitch"][1]).item()
                + com_z_lower_limit_offset,
                0.0,
            ],
            dtype=np.float32,
        )
        self.left_knee_pitch_idx = self.robot.joint_ordering.index("left_knee_pitch")
        self.leg_pitch_joint_indicies = np.array(
            [
                self.robot.joint_ordering.index(joint_name)
                for joint_name in [
                    "left_hip_pitch",
                    "left_knee_pitch",
                    "left_ank_pitch",
                    "right_hip_pitch",
                    "right_knee_pitch",
                    "right_ank_pitch",
                ]
            ]
        )
        self.leg_roll_joint_indicies = np.array(
            [
                self.robot.joint_ordering.index(joint_name)
                for joint_name in [
                    "left_hip_roll",
                    "left_ank_roll",
                    "right_hip_roll",
                    "right_ank_roll",
                ]
            ]
        )

    def get_phase_signal(self, time_curr: float | ArrayType) -> ArrayType:
        return np.zeros(1, dtype=np.float32)

    def get_vel(self, command: ArrayType) -> Tuple[ArrayType, ArrayType]:
        lin_vel = np.array([0.0, 0.0, command[-1]], dtype=np.float32)
        ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return lin_vel, ang_vel

    def get_state_ref(
        self, state_curr: ArrayType, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        torso_state = self.integrate_torso_state(
            state_curr[:3], state_curr[3:7], command
        )
        joint_pos_curr = state_curr[13 : 13 + self.robot.nu]

        # neck yaw, neck pitch, arm, waist roll, waist yaw, squat
        neck_joint_pos = self.neck_joint_limits[0] + command[:2] * (
            self.neck_joint_limits[1] - self.neck_joint_limits[0]
        )

        ref_idx = (command[2] * (self.arm_ref_size - 2)).astype(int)
        # Linearly interpolate between p_start and p_end
        arm_joint_pos = self.arm_joint_pos_ref[ref_idx]

        waist_joint_pos = self.waist_joint_limits[0] + command[3:5] * (
            self.waist_joint_limits[1] - self.waist_joint_limits[0]
        )

        com_z_curr = self.leg_fk(joint_pos_curr[self.left_knee_pitch_idx])
        com_z_target = np.clip(
            com_z_curr + self.dt * command[5],
            self.com_z_limits[0],
            self.com_z_limits[1],
        )
        leg_pitch_joint_pos = self.leg_ik(com_z_target)

        joint_pos = self.default_joint_pos.copy()
        joint_pos = inplace_update(joint_pos, self.neck_joint_indices, neck_joint_pos)
        joint_pos = inplace_update(joint_pos, self.arm_joint_indices, arm_joint_pos)
        joint_pos = inplace_update(joint_pos, self.waist_joint_indices, waist_joint_pos)
        joint_pos = inplace_update(
            joint_pos, self.leg_pitch_joint_indicies, leg_pitch_joint_pos
        )

        joint_vel = self.default_joint_vel.copy()

        stance_mask = np.ones(2, dtype=np.float32)

        return np.concatenate((torso_state, joint_pos, joint_vel, stance_mask))

    def override_motor_target(
        self, motor_target: ArrayType, state_ref: ArrayType
    ) -> ArrayType:
        neck_joint_pos = state_ref[13 + self.neck_actuator_indices]
        neck_motor_pos = self.neck_ik(neck_joint_pos)
        motor_target = inplace_update(
            motor_target,
            self.neck_actuator_indices,
            neck_motor_pos,
        )
        arm_joint_pos = state_ref[13 + self.arm_actuator_indices]
        arm_motor_pos = self.arm_ik(arm_joint_pos)
        motor_target = inplace_update(
            motor_target,
            self.arm_actuator_indices,
            arm_motor_pos,
        )
        waist_joint_pos = state_ref[13 + self.waist_actuator_indices]
        waist_motor_pos = self.waist_ik(waist_joint_pos)
        motor_target = inplace_update(
            motor_target,
            self.waist_actuator_indices,
            waist_motor_pos,
        )

        return motor_target
