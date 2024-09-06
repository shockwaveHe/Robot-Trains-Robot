from typing import Optional

from toddlerbot.ref_motion import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np
from toddlerbot.utils.math_utils import gaussian_basis_functions


class SquatReference(MotionReference):
    def __init__(
        self,
        robot: Robot,
        max_knee_pitch: float = np.pi / 2,
        min_knee_pitch: float = 0.0,
    ):
        super().__init__("squat", "episodic", robot)

        self.default_joint_pos = np.array(  # type: ignore
            list(robot.default_joint_angles.values()), dtype=np.float32
        )
        self.default_joint_vel = np.zeros_like(self.default_joint_pos)  # type: ignore

        self.knee_pitch_default = self.default_joint_pos[
            self.robot.joint_ordering.index("left_knee_pitch")
        ]
        self.max_knee_pitch = max_knee_pitch
        self.min_knee_pitch = min_knee_pitch

        self.hip_pitch_to_knee_z = self.robot.data_dict["offsets"][
            "hip_pitch_to_knee_z"
        ]
        self.knee_to_ank_pitch_z = self.robot.data_dict["offsets"][
            "knee_to_ank_pitch_z"
        ]
        self.hip_pitch_to_ank_pitch_z = np.sqrt(  # type: ignore
            self.hip_pitch_to_knee_z**2
            + self.knee_to_ank_pitch_z**2
            - 2
            * self.hip_pitch_to_knee_z
            * self.knee_to_ank_pitch_z
            * np.cos(np.pi - self.knee_pitch_default)  # type: ignore
        )
        self.shin_thigh_ratio = self.knee_to_ank_pitch_z / self.hip_pitch_to_knee_z

        self.knee_limits = np.array(  # type:ignore
            [
                self.knee_pitch_default - min_knee_pitch,
                max_knee_pitch - self.knee_pitch_default,
            ]
        )

        self.left_hip_pitch_idx = self.robot.joint_ordering.index("left_hip_pitch")
        self.left_knee_pitch_idx = self.robot.joint_ordering.index("left_knee_pitch")
        self.left_ank_pitch_idx = self.robot.joint_ordering.index("left_ank_pitch")
        self.right_hip_pitch_idx = self.robot.joint_ordering.index("right_hip_pitch")
        self.right_knee_pitch_idx = self.robot.joint_ordering.index("right_knee_pitch")
        self.right_ank_pitch_idx = self.robot.joint_ordering.index("right_ank_pitch")

        self.num_joints = len(self.robot.joint_ordering)

    def get_phase_signal(
        self, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        time_total = np.max(self.knee_limits / (command[0] + 1e-6))  # type:ignore
        phase = np.clip(time_curr / time_total, 0.0, 1.0)  # type: ignore
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

        linear_vel = np.array([0.0, 0.0, command[0]], dtype=np.float32)  # type: ignore
        angular_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # type: ignore

        joint_pos = self.default_joint_pos.copy()  # type: ignore
        leg_angles = self.calculate_leg_angles(
            np.array(command[0] * time_curr, dtype=np.float32)  # type: ignore
        )
        for idx, angle in leg_angles.items():
            joint_pos = inplace_update(joint_pos, idx, angle)

        joint_vel = self.default_joint_vel.copy()  # type: ignore

        stance_mask = np.ones(2, dtype=np.float32)  # type: ignore

        return np.concatenate(  # type: ignore
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

    def calculate_leg_angles(self, delta_z: ArrayType):
        knee_angle_cos = (
            self.hip_pitch_to_knee_z**2
            + self.knee_to_ank_pitch_z**2
            - (self.hip_pitch_to_ank_pitch_z + delta_z) ** 2
        ) / (2 * self.hip_pitch_to_knee_z * self.knee_to_ank_pitch_z)
        knee_angle_cos = np.clip(knee_angle_cos, -1.0, 1.0)  # type: ignore
        knee_angle = np.abs(np.pi - np.arccos(knee_angle_cos))  # type: ignore
        knee_angle = np.clip(knee_angle, self.min_knee_pitch, self.max_knee_pitch)  # type: ignore

        ank_pitch_angle = np.arctan2(  # type: ignore
            np.sin(knee_angle),  # type: ignore
            np.cos(knee_angle) + self.shin_thigh_ratio,  # type: ignore
        )
        hip_pitch_angle = knee_angle - ank_pitch_angle

        return {
            self.left_hip_pitch_idx: -hip_pitch_angle,
            self.left_knee_pitch_idx: knee_angle,
            self.left_ank_pitch_idx: -ank_pitch_angle,
            self.right_hip_pitch_idx: hip_pitch_angle,
            self.right_knee_pitch_idx: -knee_angle,
            self.right_ank_pitch_idx: -ank_pitch_angle,
        }
