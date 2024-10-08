import os
from typing import List, Optional

import joblib

from toddlerbot.ref_motion import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np


class BalanceReference(MotionReference):
    def __init__(
        self,
        robot: Robot,
        control_dt: float,
        arm_playback_speed: float = 1.0,
        com_z_lower_limit_offset: float = 0.01,
    ):
        super().__init__("balance", "perceptual", robot)

        self.control_dt = control_dt
        self.default_joint_pos = np.array(
            list(robot.default_joint_angles.values()), dtype=np.float32
        )
        self.default_joint_vel = np.zeros_like(self.default_joint_pos)

        ########## Neck Reference ##########
        neck_motor_names: List[str] = [
            robot.motor_ordering[i] for i in self.neck_actuator_indices
        ]
        self.neck_gear_ratio = np.ones(len(neck_motor_names), dtype=np.float32)  # type: ignore
        for i, motor_name in enumerate(neck_motor_names):
            motor_config = robot.config["joints"][motor_name]
            if motor_config["transmission"] == "gear":
                self.neck_gear_ratio = inplace_update(
                    self.neck_gear_ratio, i, -motor_config["gear_ratio"]
                )

        self.neck_yaw_idx = robot.joint_ordering.index("neck_yaw_driven")
        self.neck_pitch_idx = robot.joint_ordering.index("neck_pitch_driven")
        self.neck_yaw_limits = robot.joint_limits["neck_yaw_driven"]
        self.neck_pitch_limits = robot.joint_limits["neck_pitch_driven"]
        self.neck_yaw_target = 0.0
        self.neck_pitch_target = 0.0

        ########## Arm Reference ##########
        arm_motor_names: List[str] = [
            robot.motor_ordering[i] for i in self.arm_actuator_indices
        ]
        self.arm_gear_ratio = np.ones(len(arm_motor_names), dtype=np.float32)  # type: ignore
        for i, motor_name in enumerate(arm_motor_names):
            motor_config = robot.config["joints"][motor_name]
            if (
                motor_config["transmission"] == "gear"
                or motor_config["transmission"] == "rack_and_pinion"
            ):
                self.arm_gear_ratio = inplace_update(
                    self.arm_gear_ratio, i, -motor_config["gear_ratio"]
                )

        # Load the balance dataset
        data_path = os.path.join("toddlerbot", "ref_motion", "balance_dataset.lz4")
        data_dict = joblib.load(data_path)
        # state_array: [time(1), motor_pos(14), fsrL(1), fsrR(1), camera_frame_idx(1)]
        state_arr = data_dict["state_array"]
        self.arm_time_ref = (
            np.array(state_arr[:, 0] - state_arr[0, 0], dtype=np.float32)
            / arm_playback_speed
        )
        self.arm_joint_pos_ref = np.array(
            [
                self.arm_fk(arm_motor_pos)
                for arm_motor_pos in state_arr[:, 1 + self.arm_actuator_indices]
            ],
            dtype=np.float32,
        )
        self.arm_ref_size = len(self.arm_time_ref)

        # TODO: Add waist reference

        ########## Leg Reference ##########
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

        knee_limits = np.array(
            self.robot.joint_limits["left_knee_pitch"], dtype=np.float32
        )
        self.com_z_limits = np.array(
            [
                np.sqrt(
                    self.hip_pitch_to_knee_z**2
                    + self.knee_to_ank_pitch_z**2
                    - 2
                    * self.hip_pitch_to_knee_z
                    * self.knee_to_ank_pitch_z
                    * np.cos(np.pi - knee_limits[1])
                )
                - self.hip_pitch_to_ank_pitch_z
                + com_z_lower_limit_offset,
                0.0,
            ],
            dtype=np.float32,
        )
        self.leg_pitch_joint_indicies = np.array(
            [
                robot.joint_ordering.index("left_hip_pitch"),
                robot.joint_ordering.index("left_knee_pitch"),
                robot.joint_ordering.index("left_ank_pitch"),
                robot.joint_ordering.index("right_hip_pitch"),
                robot.joint_ordering.index("right_knee_pitch"),
                robot.joint_ordering.index("right_ank_pitch"),
            ]
        )

        self.com_z_target = np.zeros(1, dtype=np.float32)

    def get_phase_signal(
        self, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        return np.zeros(1, dtype=np.float32)

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

        command_neck_yaw = command[0]
        command_neck_pitch = command[1]
        command_arm = command[2]
        command_squat = command[3]

        # command: [neck_yaw, neck_pitch, arm, squat]
        linear_vel = np.array([0.0, 0.0, command_squat], dtype=np.float32)
        angular_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.neck_yaw_target = np.clip(
            self.neck_yaw_target + command_neck_yaw * self.control_dt,
            self.neck_yaw_limits[0],
            self.neck_yaw_limits[1],
        )
        self.neck_pitch_target = np.clip(
            self.neck_pitch_target + command_neck_pitch * self.control_dt,
            self.neck_pitch_limits[0],
            self.neck_pitch_limits[1],
        )

        command_ref_idx = (command_arm * (self.arm_ref_size - 2)).astype(int)
        time_start = self.arm_time_ref[command_ref_idx] + time_curr
        ref_idx = np.minimum(
            np.searchsorted(self.arm_time_ref, time_start, side="right") - 1,
            self.arm_ref_size - 2,
        )
        # Linearly interpolate between p_start and p_end
        arm_joint_pos_start = self.arm_joint_pos_ref[ref_idx]
        arm_joint_pos_end = self.arm_joint_pos_ref[ref_idx + 1]
        arm_duration = self.arm_time_ref[ref_idx + 1] - self.arm_time_ref[ref_idx]
        arm_joint_pos = arm_joint_pos_start + (
            arm_joint_pos_end - arm_joint_pos_start
        ) * ((time_start - self.arm_time_ref[ref_idx]) / arm_duration)

        self.com_z_target = np.clip(
            self.com_z_target + self.control_dt * command_squat,
            self.com_z_limits[0],
            self.com_z_limits[1],
        )
        leg_pitch_joint_pos = self.leg_ik(np.array(self.com_z_target, dtype=np.float32))

        joint_pos = self.default_joint_pos.copy()
        joint_pos = inplace_update(joint_pos, self.neck_yaw_idx, self.neck_yaw_target)
        joint_pos = inplace_update(
            joint_pos, self.neck_pitch_idx, self.neck_pitch_target
        )
        joint_pos = inplace_update(
            joint_pos,
            self.arm_actuator_indices,
            arm_joint_pos,
        )
        joint_pos = inplace_update(
            joint_pos, self.leg_pitch_joint_indicies, leg_pitch_joint_pos
        )

        joint_vel = self.default_joint_vel.copy()
        stance_mask = np.ones(2, dtype=np.float32)

        return np.concatenate(
            (
                path_pos,
                path_quat,
                linear_vel,
                angular_vel,
                joint_pos,
                joint_vel,
                stance_mask,
            )
        )

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

        return motor_target

    def neck_ik(self, neck_joint_pos: ArrayType) -> ArrayType:
        neck_motor_pos = neck_joint_pos / self.neck_gear_ratio
        return neck_motor_pos

    def arm_fk(self, arm_motor_pos: ArrayType) -> ArrayType:
        arm_joint_pos = arm_motor_pos * self.arm_gear_ratio
        return arm_joint_pos

    def arm_ik(self, arm_joint_pos: ArrayType) -> ArrayType:
        arm_motor_pos = arm_joint_pos / self.arm_gear_ratio
        return arm_motor_pos

    def leg_ik(self, delta_z: ArrayType):
        knee_angle_cos = (
            self.hip_pitch_to_knee_z**2
            + self.knee_to_ank_pitch_z**2
            - (self.hip_pitch_to_ank_pitch_z + delta_z) ** 2
        ) / (2 * self.hip_pitch_to_knee_z * self.knee_to_ank_pitch_z)
        knee_angle_cos = np.clip(knee_angle_cos, -1.0, 1.0)
        knee_angle = np.abs(np.pi - np.arccos(knee_angle_cos))

        ank_pitch_angle = np.arctan2(
            np.sin(knee_angle),
            np.cos(knee_angle) + self.shin_thigh_ratio,
        )
        hip_pitch_angle = knee_angle - ank_pitch_angle

        return np.concatenate(
            [
                -hip_pitch_angle,
                knee_angle,
                -ank_pitch_angle,
                hip_pitch_angle,
                -knee_angle,
                -ank_pitch_angle,
            ]
        )
