from typing import Optional

from toddlerbot.ref_motion import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np
from toddlerbot.utils.math_utils import gaussian_basis_functions


class SquatReference(MotionReference):
    def __init__(self, robot: Robot, control_dt: float):
        super().__init__("squat", "episodic", robot)

        self.default_joint_pos = np.array(
            list(robot.default_joint_angles.values()), dtype=np.float32
        )
        self.default_joint_vel = np.zeros_like(self.default_joint_pos)

        self.knee_pitch_default = self.default_joint_pos[
            self.robot.joint_ordering.index("left_knee_pitch")
        ]
        self.control_dt = control_dt

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
                - self.hip_pitch_to_ank_pitch_z,
                0.0,
            ],
            dtype=np.float32,
        )

        self.pitch_joint_indicies = [
            robot.joint_ordering.index("left_hip_pitch"),
            robot.joint_ordering.index("left_knee_pitch"),
            robot.joint_ordering.index("left_ank_pitch"),
            robot.joint_ordering.index("right_hip_pitch"),
            robot.joint_ordering.index("right_knee_pitch"),
            robot.joint_ordering.index("right_ank_pitch"),
        ]

        self.com_z_target = np.zeros(1, dtype=np.float32)

    def reset(self):
        self.com_z_target = np.zeros(1, dtype=np.float32)

    def get_phase_signal(
        self, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        phase = np.clip(time_curr, 0.0, 1.0)
        phase_signal = gaussian_basis_functions(phase)
        return phase_signal

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

        linear_vel = np.array([0.0, 0.0, command[0]], dtype=np.float32)
        angular_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.com_z_target = np.clip(
            self.com_z_target + self.control_dt * command[0],
            self.com_z_limits[0],
            self.com_z_limits[1],
        )

        joint_pos = self.default_joint_pos.copy()
        pitch_joint_pos = self.leg_ik(np.array(self.com_z_target, dtype=np.float32))
        joint_pos = inplace_update(
            joint_pos, self.pitch_joint_indicies, pitch_joint_pos
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

        return np.vstack(
            [
                -hip_pitch_angle,
                knee_angle,
                -ank_pitch_angle,
                hip_pitch_angle,
                -knee_angle,
                -ank_pitch_angle,
            ],
            dtype=np.float32,
        ).T

    def override_motor_target(
        self, motor_target: ArrayType, state_ref: ArrayType
    ) -> ArrayType:
        motor_target = inplace_update(
            motor_target,
            self.neck_actuator_indices,
            self.default_motor_pos[self.neck_actuator_indices],
        )
        motor_target = inplace_update(
            motor_target,
            self.arm_actuator_indices,
            self.default_motor_pos[self.arm_actuator_indices],
        )
        motor_target = inplace_update(
            motor_target,
            self.waist_actuator_indices,
            self.default_motor_pos[self.waist_actuator_indices],
        )

        return motor_target
