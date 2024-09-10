import os
from typing import List, Optional

import joblib  # type: ignore

from toddlerbot.ref_motion import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np
from toddlerbot.utils.math_utils import interpolate_action


class BalanceReference(MotionReference):
    def __init__(self, robot: Robot):
        super().__init__("balance", "perceptual", robot)

        arm_motor_names: List[str] = [
            robot.motor_ordering[i] for i in self.arm_motor_indices
        ]
        self.arm_coef = np.ones(len(arm_motor_names), dtype=np.float32)  # type: ignore
        for i, motor_name in enumerate(arm_motor_names):
            motor_config = robot.config["joints"][motor_name]
            if motor_config["transmission"] == "gears":
                self.arm_coef = inplace_update(
                    self.arm_coef, i, -motor_config["gear_ratio"]
                )

        data_path = os.path.join("toddlerbot", "ref_motion", "balance_dataset.lz4")
        data_dict = joblib.load(data_path)  # type: ignore

        # state_array: [time(1), motor_pos(14), fsrL(1), fsrR(1), camera_frame_idx(1)]
        state_arr = data_dict["state_array"]
        self.time_ref = np.array(  # type: ignore
            state_arr[:, 0], dtype=np.float32
        )
        self.time_ref -= self.time_ref[0]
        self.arm_joint_pos_ref = np.array(  # type: ignore
            [
                self.arm_fk(arm_motor_pos)
                for arm_motor_pos in state_arr[:, 1 + self.arm_motor_indices]
            ],
            dtype=np.float32,
        )
        self.ref_size = len(self.time_ref)

    def get_phase_signal(
        self, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        return np.zeros(1, dtype=np.float32)  # type: ignore

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

        linear_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # type: ignore
        angular_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # type: ignore

        ref_idx = (command[0] * (self.ref_size - 1)).astype(int)  # type: ignore
        arm_joint_pos = interpolate_action(  # type: ignore
            self.time_ref[ref_idx] + time_curr, self.time_ref, self.arm_joint_pos_ref
        )

        joint_pos = self.default_joint_pos.copy()  # type: ignore
        joint_pos = inplace_update(
            joint_pos,
            self.arm_motor_indices,  # type: ignore
            arm_joint_pos,
        )

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

    def override_motor_target(
        self, motor_target: ArrayType, state_ref: ArrayType
    ) -> ArrayType:
        motor_target = inplace_update(
            motor_target,
            self.neck_motor_indices,  # type: ignore
            self.default_motor_pos[self.neck_motor_indices],
        )
        arm_joint_pos = state_ref[13 + self.arm_motor_indices]
        arm_motor_pos = self.arm_ik(arm_joint_pos)
        motor_target = inplace_update(
            motor_target,
            self.arm_motor_indices,  # type: ignore
            arm_motor_pos,
        )

        return motor_target

    def arm_fk(self, arm_motor_pos: ArrayType) -> ArrayType:
        arm_joint_pos = arm_motor_pos / self.arm_coef
        return arm_joint_pos

    def arm_ik(self, arm_joint_pos: ArrayType) -> ArrayType:
        arm_motor_pos = arm_joint_pos * self.arm_coef
        return arm_motor_pos
