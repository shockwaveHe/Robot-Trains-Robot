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
        default_motor_pos: Optional[ArrayType] = None,
        default_motor_vel: Optional[ArrayType] = None,
    ):
        super().__init__("rotate_torso", "episodic", robot)

        if default_motor_pos is None:
            self.default_motor_pos = np.zeros(  # type: ignore
                len(self.robot.motor_ordering), dtype=np.float32
            )
        else:
            self.default_motor_pos = default_motor_pos

        self.default_motor_vel = default_motor_vel

        self.waist_roll_limits = self.robot.joint_limits["waist_roll"]
        self.waist_yaw_limits = self.robot.joint_limits["waist_yaw"]
        self.waist_roll_coef = self.robot.config["general"]["offsets"][
            "waist_roll_coef"
        ]
        self.waist_yaw_coef = self.robot.config["general"]["offsets"]["waist_yaw_coef"]
        self.waist_act_1_idx = self.robot.motor_ordering.index("waist_act_1")
        self.waist_act_2_idx: int = self.robot.motor_ordering.index("waist_act_2")

        self.nu = len(self.robot.motor_ordering)

    def get_state_ref(
        self,
        path_pos: ArrayType,
        path_quat: ArrayType,
        time_curr: Optional[float | ArrayType] = None,
        time_total: Optional[float | ArrayType] = None,
        command: Optional[ArrayType] = None,
    ) -> Tuple[ArrayType, ArrayType]:
        if time_curr is None:
            raise ValueError(f"time_curr is required for {self.name}")

        if time_total is None:
            raise ValueError(f"time_total is required for {self.name}")

        if command is None:
            raise ValueError(f"command is required for {self.name}")

        phase = np.clip(time_curr / time_total, 0.0, 1.0)  # type: ignore
        phase_signal = gaussian_basis_functions(phase)

        linear_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # type: ignore
        angular_vel = np.array([command[0], 0.0, command[1]], dtype=np.float32)  # type: ignore

        assert self.default_motor_pos is not None
        motor_pos = self.default_motor_pos.copy()  # type: ignore

        waist_roll: ArrayType = np.clip(  # type: ignore
            command[0] * time_curr,
            self.waist_roll_limits[0],
            self.waist_roll_limits[1],
        )
        waist_yaw: ArrayType = np.clip(  # type: ignore
            command[1] * time_curr,
            self.waist_yaw_limits[0],
            self.waist_yaw_limits[1],
        )
        waist_act_1, waist_act_2 = self.waist_ik(waist_roll, waist_yaw)

        motor_pos = inplace_update(motor_pos, self.waist_act_1_idx, waist_act_1)
        motor_pos = inplace_update(motor_pos, self.waist_act_2_idx, waist_act_2)

        if self.default_motor_vel is None:
            motor_vel = np.zeros(self.nu, dtype=np.float32)  # type: ignore
        else:
            motor_vel = self.default_motor_vel.copy()  # type: ignore

        stance_mask = np.ones(2, dtype=np.float32)  # type: ignore

        return phase_signal, np.concatenate(  # type: ignore
            (
                path_pos,
                path_quat,
                linear_vel,
                angular_vel,
                motor_pos,
                motor_vel,
                stance_mask,
            )  # type: ignore
        )

    def waist_ik(
        self, waist_roll: ArrayType, waist_yaw: ArrayType
    ) -> Tuple[ArrayType, ArrayType]:
        roll = waist_roll / self.waist_roll_coef
        yaw = waist_yaw / self.waist_yaw_coef
        waist_act_1 = (-roll + yaw) / 2
        waist_act_2 = (roll + yaw) / 2
        return waist_act_1, waist_act_2
