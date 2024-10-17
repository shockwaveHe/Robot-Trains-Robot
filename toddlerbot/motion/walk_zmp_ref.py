import os
import pickle
from typing import Tuple

import jax

from toddlerbot.algorithms.zmp_walk import ZMPWalk
from toddlerbot.motion.motion_ref import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np


class WalkZMPReference(MotionReference):
    def __init__(self, robot: Robot, dt: float, cycle_time: float):
        super().__init__("walk_zmp", "periodic", robot, dt)

        self._setup_zmp(cycle_time)

    def _setup_zmp(self, cycle_time: float):
        self.cycle_time = cycle_time

        self.left_hip_yaw_idx = self.robot.motor_ordering.index("left_hip_yaw_drive")
        self.right_hip_yaw_idx = self.robot.motor_ordering.index("right_hip_yaw_drive")

        self.zmp_walk = ZMPWalk(self.robot, cycle_time)

        lookup_table_path = os.path.join(
            "toddlerbot", "ref_motion", "walk_zmp_lookup_table.pkl"
        )
        if os.path.exists(lookup_table_path):
            with open(lookup_table_path, "rb") as f:
                (
                    lookup_keys,
                    com_ref_list,
                    stance_mask_ref_list,
                    leg_joint_pos_ref_list,
                ) = pickle.load(f)
        else:
            lookup_keys, com_ref_list, leg_joint_pos_ref_list, stance_mask_ref_list = (
                self.zmp_walk.build_lookup_table()
            )
            with open(lookup_table_path, "wb") as f:
                pickle.dump(
                    (
                        lookup_keys,
                        com_ref_list,
                        stance_mask_ref_list,
                        leg_joint_pos_ref_list,
                    ),
                    f,
                )

        self.lookup_keys = np.array(lookup_keys, dtype=np.float32)
        self.lookup_length = np.array(
            [len(stance_mask_ref) for stance_mask_ref in stance_mask_ref_list],
            dtype=np.float32,
        )

        num_total_steps_max = max(
            [len(stance_mask_ref) for stance_mask_ref in stance_mask_ref_list]
        )
        self.stance_mask_lookup = np.zeros(
            (len(stance_mask_ref_list), num_total_steps_max, 2), dtype=np.float32
        )
        self.leg_joint_pos_lookup = np.zeros(
            (len(stance_mask_ref_list), num_total_steps_max, 12), dtype=np.float32
        )
        for i, (stance_mask_ref, leg_joint_pos_ref) in enumerate(
            zip(stance_mask_ref_list, leg_joint_pos_ref_list)
        ):
            self.stance_mask_lookup = inplace_update(
                self.stance_mask_lookup,
                (i, slice(None, len(stance_mask_ref))),
                stance_mask_ref,
            )
            self.leg_joint_pos_lookup = inplace_update(
                self.leg_joint_pos_lookup,
                (i, slice(None, len(leg_joint_pos_ref))),
                leg_joint_pos_ref,
            )

        if self.use_jax:
            self.lookup_keys = jax.device_put(self.lookup_keys)
            self.lookup_length = jax.device_put(self.lookup_length)
            self.stance_mask_lookup = jax.device_put(self.stance_mask_lookup)
            self.leg_joint_pos_lookup = jax.device_put(self.leg_joint_pos_lookup)

    def get_phase_signal(self, time_curr: float | ArrayType) -> ArrayType:
        phase_signal = np.array(
            [
                np.sin(2 * np.pi * time_curr / self.cycle_time),
                np.cos(2 * np.pi * time_curr / self.cycle_time),
            ],
            dtype=np.float32,
        )
        return phase_signal

    def get_vel(self, command: ArrayType) -> Tuple[ArrayType, ArrayType]:
        # The first 5 commands are neck yaw, neck pitch, arm, waist roll, waist yaw
        lin_vel = np.array([command[5], command[6], 0.0], dtype=np.float32)
        ang_vel = np.array([0.0, 0.0, command[7]], dtype=np.float32)
        return lin_vel, ang_vel

    # @profile()
    def get_state_ref(
        self, state_curr: ArrayType, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        torso_state = self.integrate_torso_state(
            state_curr[:3], state_curr[3:7], command
        )

        neck_yaw_pos = np.interp(
            command[0],
            np.array([-1, 0, 1]),
            np.array([self.neck_joint_limits[0, 0], 0.0, self.neck_joint_limits[1, 0]]),
        )
        neck_pitch_pos = np.interp(
            command[1],
            np.array([-1, 0, 1]),
            np.array([self.neck_joint_limits[0, 1], 0.0, self.neck_joint_limits[1, 1]]),
        )
        neck_joint_pos = np.array([neck_yaw_pos, neck_pitch_pos])

        ref_idx = (command[2] * (self.arm_ref_size - 2)).astype(int)
        # Linearly interpolate between p_start and p_end
        arm_joint_pos = self.arm_joint_pos_ref[ref_idx]

        waist_roll_pos = np.interp(
            command[3],
            np.array([-1, 0, 1]),
            np.array(
                [self.waist_joint_limits[0, 0], 0.0, self.waist_joint_limits[1, 0]]
            ),
        )
        waist_yaw_pos = np.interp(
            command[4],
            np.array([-1, 0, 1]),
            np.array(
                [self.waist_joint_limits[0, 1], 0.0, self.waist_joint_limits[1, 1]]
            ),
        )
        waist_joint_pos = np.array([waist_roll_pos, waist_yaw_pos])

        # is_zero_commmand = np.linalg.norm(command) < 1e-6
        nearest_command_idx = np.argmin(
            np.linalg.norm(self.lookup_keys - command[5:], axis=1)
        )
        step_idx = np.round(time_curr / self.dt).astype(int)
        leg_joint_pos = self.leg_joint_pos_lookup[nearest_command_idx][
            (step_idx % self.lookup_length[nearest_command_idx]).astype(int)
        ]
        joint_pos = self.default_joint_pos.copy()
        joint_pos = inplace_update(joint_pos, self.neck_joint_indices, neck_joint_pos)
        joint_pos = inplace_update(joint_pos, self.arm_joint_indices, arm_joint_pos)
        joint_pos = inplace_update(joint_pos, self.waist_joint_indices, waist_joint_pos)
        joint_pos = inplace_update(joint_pos, self.leg_joint_indices, leg_joint_pos)

        joint_vel = self.default_joint_vel.copy()

        stance_mask = self.stance_mask_lookup[nearest_command_idx][
            (step_idx % self.lookup_length[nearest_command_idx]).astype(int)
        ]

        return np.concatenate((torso_state, joint_pos, joint_vel, stance_mask))

    def override_motor_target(
        self, motor_target: ArrayType, state_ref: ArrayType
    ) -> ArrayType:
        neck_joint_pos = state_ref[13 + self.neck_motor_indices]
        neck_motor_pos = self.neck_ik(neck_joint_pos)
        motor_target = inplace_update(
            motor_target,
            self.neck_motor_indices,
            neck_motor_pos,
        )
        arm_joint_pos = state_ref[13 + self.arm_motor_indices]
        arm_motor_pos = self.arm_ik(arm_joint_pos)
        motor_target = inplace_update(
            motor_target,
            self.arm_motor_indices,
            arm_motor_pos,
        )
        waist_joint_pos = state_ref[13 + self.waist_motor_indices]
        waist_motor_pos = self.waist_ik(waist_joint_pos)
        motor_target = inplace_update(
            motor_target,
            self.waist_motor_indices,
            waist_motor_pos,
        )

        return motor_target
