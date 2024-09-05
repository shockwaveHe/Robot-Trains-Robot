from typing import Optional, Tuple

from toddlerbot.ref_motion import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np
from toddlerbot.utils.math_utils import gaussian_basis_functions


class RotateTorsoReference(MotionReference):
    def __init__(
        self,
        robot: Robot,
        default_joint_pos: Optional[ArrayType] = None,
        default_joint_vel: Optional[ArrayType] = None,
    ):
        super().__init__("rotate_torso", "episodic", robot)

        self.default_joint_pos = default_joint_pos
        self.default_joint_vel = default_joint_vel
        if self.default_joint_pos is None:
            self.default_joint_pos = np.zeros(  # type: ignore
                len(self.robot.joint_ordering), dtype=np.float32
            )

        self.waist_roll_limits = self.robot.joint_limits["waist_roll"]
        self.waist_yaw_limits = self.robot.joint_limits["waist_yaw"]

        self.num_joints = len(self.robot.joint_ordering)

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
        # TODO: Fix this
        phase_signal = np.zeros_like(phase_signal)  # type: ignore

        linear_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # type: ignore
        angular_vel = np.array([command[0], 0.0, command[1]], dtype=np.float32)  # type: ignore

        assert self.default_joint_pos is not None
        joint_pos = self.default_joint_pos.copy()  # type: ignore

        waist_roll_pos = np.clip(  # type: ignore
            command[0] * time_curr,
            self.waist_roll_limits[0],
            self.waist_roll_limits[1],
        )
        waist_yaw_pos = np.clip(  # type: ignore
            command[1] * time_curr,
            self.waist_yaw_limits[0],
            self.waist_yaw_limits[1],
        )
        joint_pos = inplace_update(
            joint_pos, self.get_joint_idx("waist_roll"), waist_roll_pos
        )
        joint_pos = inplace_update(
            joint_pos, self.get_joint_idx("waist_yaw"), waist_yaw_pos
        )

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
