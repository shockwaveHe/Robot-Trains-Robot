from typing import Optional, Tuple

from toddlerbot.ref_motion import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np
from toddlerbot.utils.math_utils import gaussian_basis_functions


class RotateTorsoReference(MotionReference):
    def __init__(self, robot: Robot):
        super().__init__("rotate_torso", "episodic", robot)

        self.waist_roll_limits = np.array(
            self.robot.joint_limits["waist_roll"], dtype=np.float32
        )
        self.waist_yaw_limits = np.array(
            self.robot.joint_limits["waist_yaw"], dtype=np.float32
        )
        self.waist_roll_idx = self.robot.joint_ordering.index("waist_roll")
        self.waist_yaw_idx = self.robot.joint_ordering.index("waist_yaw")
        self.waist_roll_coef = float(
            self.robot.config["general"]["offsets"]["waist_roll_coef"]
        )
        self.waist_yaw_coef = float(
            self.robot.config["general"]["offsets"]["waist_yaw_coef"]
        )

        self.num_joints = len(self.robot.joint_ordering)

    def get_phase_signal(
        self, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        time_total = np.max(
            np.concatenate(
                [
                    self.waist_roll_limits / (command[0] + 1e-6),
                    self.waist_yaw_limits / (command[1] + 1e-6),
                ]
            )
        )
        phase = np.clip(time_curr / time_total, 0.0, 1.0)
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

        linear_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        angular_vel = np.array([command[0], 0.0, command[1]], dtype=np.float32)

        waist_roll = np.clip(
            command[0] * time_curr,
            self.waist_roll_limits[0],
            self.waist_roll_limits[1],
        )
        waist_yaw = np.clip(
            command[1] * time_curr,
            self.waist_yaw_limits[0],
            self.waist_yaw_limits[1],
        )

        joint_pos = self.default_joint_pos.copy()
        joint_pos = inplace_update(joint_pos, self.waist_roll_idx, waist_roll)
        joint_pos = inplace_update(joint_pos, self.waist_yaw_idx, waist_yaw)

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

        waist_roll_ref = state_ref[13 + self.waist_roll_idx]
        waist_yaw_ref = state_ref[13 + self.waist_yaw_idx]
        waist_act_1_ref, waist_act_2_ref = self.waist_ik(waist_roll_ref, waist_yaw_ref)
        motor_target = inplace_update(
            motor_target, self.waist_actuator_indices[0], waist_act_1_ref
        )
        motor_target = inplace_update(
            motor_target, self.waist_actuator_indices[1], waist_act_2_ref
        )

        return motor_target

    def waist_ik(
        self, waist_roll: ArrayType, waist_yaw: ArrayType
    ) -> Tuple[ArrayType, ArrayType]:
        roll = waist_roll / self.waist_roll_coef
        yaw = waist_yaw / self.waist_yaw_coef
        waist_act_1 = (-roll + yaw) / 2
        waist_act_2 = (roll + yaw) / 2
        return waist_act_1, waist_act_2
