import os
import pickle
from typing import Tuple

import jax

from toddlerbot.algorithms.zmp_walk import ZMPWalk
from toddlerbot.reference.motion_ref import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import (
    ArrayType,
    conditional_update,
    inplace_add,
    inplace_update,
)
from toddlerbot.utils.array_utils import array_lib as np


class WalkZMPReference(MotionReference):
    """Class for generating a ZMP-based walking reference for the toddlerbot robot."""

    def __init__(
        self, robot: Robot, dt: float, cycle_time: float, waist_roll_max: float
    ):
        """Initializes the walk ZMP (Zero Moment Point) controller.

        Args:
            robot (Robot): The robot instance to be controlled.
            dt (float): The time step for the controller.
            cycle_time (float): The duration of one walking cycle.
            waist_roll_max (float): The maximum allowable roll angle for the robot's waist.
        """
        super().__init__("walk_zmp", "periodic", robot, dt)

        self.cycle_time = cycle_time
        self.waist_roll_max = waist_roll_max

        self._setup_zmp()

    def _setup_zmp(self):
        """Initializes the Zero Moment Point (ZMP) walking parameters and lookup tables for the robot.

        This method sets up the necessary indices for hip joints, calculates the single and double support phase ratios, and initializes the ZMPWalk object. It attempts to load precomputed lookup tables for center of mass (COM) references, stance masks, and leg joint positions from a file. If the file does not exist, it builds the lookup tables and saves them. The lookup tables are then stored as class attributes, and optionally transferred to JAX devices for computation if JAX is enabled.
        """
        left_hip_yaw_idx = self.robot.joint_ordering.index("left_hip_yaw_driven")
        self.left_hip_roll_rel_idx = (
            self.robot.joint_ordering.index("left_hip_roll") - left_hip_yaw_idx
        )
        self.right_hip_roll_rel_idx = (
            self.robot.joint_ordering.index("right_hip_roll") - left_hip_yaw_idx
        )

        single_double_ratio = 2.0
        self.zmp_walk = ZMPWalk(self.robot, self.cycle_time, single_double_ratio)

        # Determine single and double support phase ratios
        self.single_support_ratio = single_double_ratio / (single_double_ratio + 1)
        self.double_support_ratio = 1 - self.single_support_ratio

        lookup_table_path = os.path.join(
            "toddlerbot", "descriptions", self.robot.name, "walk_zmp_lookup_table.pkl"
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
        """Calculate the phase signal as a sine and cosine pair for a given time.

        Args:
            time_curr (float | ArrayType): The current time or an array of time values.

        Returns:
            ArrayType: A numpy array containing the sine and cosine of the phase, with dtype float32.
        """
        phase_signal = np.array(
            [
                np.sin(2 * np.pi * time_curr / self.cycle_time),
                np.cos(2 * np.pi * time_curr / self.cycle_time),
            ],
            dtype=np.float32,
        )
        return phase_signal

    def get_vel(self, command: ArrayType) -> Tuple[ArrayType, ArrayType]:
        """Calculates linear and angular velocities based on the given command array.

        The function interprets the command array to extract linear and angular velocity
        components. The linear velocity is derived from specific indices of the command
        array, while the angular velocity is determined from another index.

        Args:
            command (ArrayType): An array containing control commands, where indices 5 and 6
                are used for linear velocity components and index 7 for angular velocity.

        Returns:
            Tuple[ArrayType, ArrayType]: A tuple containing the linear velocity as the first
            element and the angular velocity as the second element, both as numpy arrays.
        """
        # The first 5 commands are neck yaw, neck pitch, arm, waist roll, waist yaw
        lin_vel = np.array([command[5], command[6], 0.0], dtype=np.float32)
        ang_vel = np.array([0.0, 0.0, command[7]], dtype=np.float32)
        return lin_vel, ang_vel

    # @profile()
    def get_state_ref(
        self, state_curr: ArrayType, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        path_state = self.integrate_path_state(state_curr, command)

        joint_pos = self.default_joint_pos.copy()
        motor_pos = self.default_motor_pos.copy()
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
        joint_pos = inplace_update(joint_pos, self.neck_joint_indices, neck_joint_pos)
        motor_pos = inplace_update(
            motor_pos, self.neck_motor_indices, self.neck_ik(neck_joint_pos)
        )

        ref_idx = (command[2] * (self.arm_ref_size - 2)).astype(int)
        # Linearly interpolate between p_start and p_end
        arm_joint_pos = self.arm_joint_pos_ref[ref_idx]
        joint_pos = inplace_update(joint_pos, self.arm_joint_indices, arm_joint_pos)
        motor_pos = inplace_update(
            motor_pos, self.arm_motor_indices, self.arm_ik(arm_joint_pos)
        )

        is_static_pose = np.logical_or(
            np.linalg.norm(command[5:]) < 1e-6, time_curr < 1e-6
        )
        nearest_command_idx = np.argmin(
            np.linalg.norm(self.lookup_keys - command[5:], axis=1)
        )
        step_idx = np.round(time_curr / self.dt).astype(int)

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
        joint_pos = inplace_update(joint_pos, self.waist_joint_indices, waist_joint_pos)
        motor_pos = inplace_update(
            motor_pos, self.waist_motor_indices, self.waist_ik(waist_joint_pos)
        )

        def get_leg_joint_pos() -> ArrayType:
            leg_joint_pos = self.leg_joint_pos_lookup[nearest_command_idx][
                (step_idx % self.lookup_length[nearest_command_idx]).astype(int)
            ]
            leg_joint_pos = inplace_add(
                leg_joint_pos, self.left_hip_roll_rel_idx, -waist_roll_pos
            )
            leg_joint_pos = inplace_add(
                leg_joint_pos, self.right_hip_roll_rel_idx, waist_roll_pos
            )
            return leg_joint_pos

        leg_joint_pos = get_leg_joint_pos()
        joint_pos = inplace_update(joint_pos, self.leg_joint_indices, leg_joint_pos)
        motor_pos = inplace_update(
            motor_pos, self.leg_motor_indices, self.leg_ik(leg_joint_pos)
        )

        stance_mask = np.where(
            is_static_pose,
            np.ones(2, dtype=np.float32),
            self.stance_mask_lookup[nearest_command_idx][
                (step_idx % self.lookup_length[nearest_command_idx]).astype(int)
            ],
        )
        return np.concatenate((path_state, motor_pos, joint_pos, stance_mask))