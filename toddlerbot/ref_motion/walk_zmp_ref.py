import os
import pickle
from typing import List, Optional

import jax
import numpy

from toddlerbot.algorithms.zmp.zmp_walk import ZMPWalk
from toddlerbot.ref_motion import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np


class WalkZMPReference(MotionReference):
    def __init__(
        self,
        robot: Robot,
        command_ranges: List[List[float]],
        cycle_time: float,
        control_dt: float,
    ):
        super().__init__("walk_zmp", "periodic", robot)

        self.cycle_time = cycle_time
        self.control_dt = control_dt

        self.default_joint_pos = np.array(
            list(robot.default_joint_angles.values()), dtype=np.float32
        )
        self.default_joint_vel = np.zeros_like(self.default_joint_pos)

        joint_groups = numpy.array(
            [robot.joint_groups[name] for name in robot.joint_ordering]
        )
        self.leg_joint_indices = np.arange(len(robot.joint_ordering))[
            joint_groups == "leg"
        ]

        self.left_hip_yaw_idx = robot.motor_ordering.index("left_hip_yaw_drive")
        self.right_hip_yaw_idx = robot.motor_ordering.index("right_hip_yaw_drive")

        self.zmp_walk = ZMPWalk(robot, cycle_time)

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
            lookup_keys, com_ref_list, stance_mask_ref_list, leg_joint_pos_ref_list = (
                self.zmp_walk.build_lookup_table(command_ranges)
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

        num_commands = len(stance_mask_ref_list)
        num_total_steps_max = max(
            [len(stance_mask_ref) for stance_mask_ref in stance_mask_ref_list]
        )
        self.stance_mask_lookup = np.zeros(
            (num_commands, num_total_steps_max, 2), dtype=np.float32
        )
        self.leg_joint_pos_lookup = np.zeros(
            (num_commands, num_total_steps_max, 12), dtype=np.float32
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

        if os.environ.get("USE_JAX", "false") == "true":
            self.lookup_keys = jax.device_put(self.lookup_keys)
            self.lookup_length = jax.device_put(self.lookup_length)
            self.stance_mask_lookup = jax.device_put(self.stance_mask_lookup)
            self.leg_joint_pos_lookup = jax.device_put(self.leg_joint_pos_lookup)

    def get_phase_signal(
        self, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        phase_signal = np.array(
            [
                np.sin(2 * np.pi * time_curr / self.cycle_time),
                np.cos(2 * np.pi * time_curr / self.cycle_time),
            ],
            dtype=np.float32,
        )
        return phase_signal

    # @profile()
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

        linear_vel = np.array([command[0], command[1], 0.0], dtype=np.float32)
        angular_vel = np.array([0.0, 0.0, command[2]], dtype=np.float32)

        # is_zero_commmand = np.linalg.norm(command) < 1e-6
        nearest_command_idx = np.argmin(
            np.linalg.norm(self.lookup_keys - command, axis=1)
        )
        idx = np.round(time_curr / self.control_dt).astype(int)
        joint_pos = self.default_joint_pos.copy()
        joint_pos = inplace_update(
            joint_pos,
            self.leg_joint_indices,
            self.leg_joint_pos_lookup[nearest_command_idx][
                (idx % self.lookup_length[nearest_command_idx]).astype(int)
            ],
        )

        joint_vel = self.default_joint_vel.copy()

        stance_mask = self.stance_mask_lookup[nearest_command_idx][
            (idx % self.lookup_length[nearest_command_idx]).astype(int)
        ]

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
        # motor_target = inplace_update(
        #     motor_target,
        #     self.waist_actuator_indices,
        #     self.default_motor_pos[self.waist_actuator_indices],
        # )
        motor_target = inplace_update(
            motor_target,
            self.left_hip_yaw_idx,
            self.default_motor_pos[self.left_hip_yaw_idx],
        )
        motor_target = inplace_update(
            motor_target,
            self.right_hip_yaw_idx,
            self.default_motor_pos[self.right_hip_yaw_idx],
        )

        return motor_target
