from typing import Optional, Tuple

from toddlerbot.ref_motion import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np
from toddlerbot.utils.math_utils import gaussian_basis_functions


class SquatReference(MotionReference):
    def __init__(
        self,
        robot: Robot,
        episode_time: float,
        default_joint_pos: Optional[ArrayType] = None,
        default_joint_vel: Optional[ArrayType] = None,
        max_knee_pitch: float = np.pi / 2,
        min_knee_pitch: float = 0.0,
    ):
        super().__init__("squat", "episodic", robot)

        self.episode_time = episode_time

        self.default_joint_pos = default_joint_pos
        self.default_joint_vel = default_joint_vel
        if self.default_joint_pos is None:
            self.default_joint_pos = np.zeros(  # type: ignore
                len(self.robot.joint_ordering), dtype=np.float32
            )
            self.knee_pitch_default = 0.0
        else:
            self.knee_pitch_default = self.default_joint_pos[
                self.get_joint_idx("left_knee_pitch")
            ]
        self.max_knee_pitch = max_knee_pitch
        self.min_knee_pitch = min_knee_pitch

        self.num_joints = len(self.robot.joint_ordering)
        self.hip_pitch_to_knee_z = self.robot.data_dict["offsets"][
            "hip_pitch_to_knee_z"
        ]
        self.knee_to_ank_pitch_z = self.robot.data_dict["offsets"][
            "knee_to_ank_pitch_z"
        ]
        self.shin_thigh_ratio = self.knee_to_ank_pitch_z / self.hip_pitch_to_knee_z

    def get_state_ref(
        self,
        path_pos: ArrayType,
        path_quat: ArrayType,
        time_curr: Optional[float | ArrayType] = None,
        command: Optional[ArrayType] = None,
    ) -> Tuple[ArrayType, ArrayType]:
        if time_curr is None:
            raise ValueError(f"time_curr is required for {self.name}")

        if command is None:
            raise ValueError(f"command is required for {self.name}")

        phase_signal = gaussian_basis_functions(np.array(time_curr, dtype=np.float32))  # type: ignore

        linear_vel = np.array([0.0, 0.0, command[0]], dtype=np.float32)  # type: ignore
        angular_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # type: ignore

        assert self.default_joint_pos is not None
        joint_pos = self.default_joint_pos.copy()  # type: ignore

        leg_angles = self.calculate_leg_angles(
            np.array(command[0] * time_curr / self.episode_time, dtype=np.float32)  # type: ignore
        )
        for name, angle in leg_angles.items():
            joint_pos = inplace_update(joint_pos, self.get_joint_idx(name), angle)

        if self.default_joint_vel is None:
            joint_vel = np.zeros(self.num_joints, dtype=np.float32)  # type: ignore
        else:
            joint_vel = self.default_joint_vel.copy()  # type: ignore

        stance_mask = np.ones(2, dtype=np.float32)  # type: ignore

        return phase_signal, np.concatenate(  # type: ignore
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

    def calculate_leg_angles(self, signal: ArrayType):
        signal_sign = (signal < 0).astype(int)  # type: ignore
        delta_knee_pitch = (
            self.knee_pitch_default - self.min_knee_pitch
        ) * signal_sign + (self.max_knee_pitch - self.knee_pitch_default) * (
            1 - signal_sign
        )

        knee_angle = np.abs(signal * delta_knee_pitch + self.knee_pitch_default)  # type: ignore

        ank_pitch_angle = np.arctan2(  # type: ignore
            np.sin(knee_angle),  # type: ignore
            np.cos(knee_angle) + self.shin_thigh_ratio,  # type: ignore
        )
        hip_pitch_angle = knee_angle - ank_pitch_angle

        return {
            "left_hip_pitch": -hip_pitch_angle,
            "left_knee_pitch": knee_angle,
            "left_ank_pitch": -ank_pitch_angle,
            "right_hip_pitch": hip_pitch_angle,
            "right_knee_pitch": -knee_angle,
            "right_ank_pitch": -ank_pitch_angle,
        }
