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
        """Generate a reference state for a robotic system based on the current state, time, and command inputs.

        This function calculates the desired joint and motor positions for a robot by integrating the current state with the given command inputs. It interpolates neck, waist, and arm positions, and determines leg joint positions based on whether the robot is in a static pose or dynamic motion. The function returns a concatenated array representing the path state, motor positions, joint positions, and stance mask.

        Args:
            state_curr (ArrayType): The current state of the robot.
            time_curr (float | ArrayType): The current time or time array.
            command (ArrayType): Command inputs for the robot's movement.

        Returns:
            ArrayType: A concatenated array of the path state, motor positions, joint positions, and stance mask.
        """
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
        arm_joint_pos = np.where(
            command[2] > 0,
            self.arm_joint_pos_ref[ref_idx],
            self.default_joint_pos[self.arm_joint_indices],
        )
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

        # # Normalize the current time within the cycle
        # cycle_phase = (time_curr / self.cycle_time) % 1.0
        # # Determine if the current phase is in single support or double support
        # in_single_support = np.logical_or(
        #     np.logical_and(
        #         self.double_support_ratio / 2 <= cycle_phase, cycle_phase < 0.5
        #     ),
        #     0.5 + self.double_support_ratio / 2 <= cycle_phase,
        # )
        # in_single_support = np.logical_and(in_single_support, ~is_static_pose)
        # # Set waist_roll_pos based on the current support phase
        # waist_roll_pos = np.where(
        #     ~in_single_support,
        #     0.0,
        #     np.sin(
        #         2
        #         * np.pi
        #         * np.where(
        #             cycle_phase < 0.5,
        #             -np.clip(cycle_phase - self.double_support_ratio / 2, 0, None),
        #             np.clip(cycle_phase - 0.5 - self.double_support_ratio / 2, 0, None),
        #         )
        #         / self.single_support_ratio
        #     )
        #     * self.waist_roll_max,
        # )
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

        def get_leg_joint_pos_init() -> ArrayType:
            """Calculates the initial positions of the leg joints based on the desired center of mass (CoM) position.

            This function computes the initial leg joint positions by first determining the current CoM position and then applying a proportional-derivative (PD) controller to minimize the error between the current and desired CoM positions. The resulting control values are used to compute the leg joint positions through inverse kinematics.

            Returns:
                ArrayType: The initial positions of the leg joints.
            """
            state_ref = np.concatenate((path_state, motor_pos, joint_pos))
            qpos = self.get_qpos_ref(state_ref)
            data = self.forward(qpos)

            com_pos = np.array(data.subtree_com[0], dtype=np.float32)
            # PD controller on CoM position
            com_pos_error = self.desired_com[:2] - com_pos[:2]
            com_ctrl = self.com_kp * com_pos_error
            leg_joint_pos = self.com_ik(0, com_ctrl[0], com_ctrl[1])
            return leg_joint_pos

        def get_leg_joint_pos() -> ArrayType:
            """Retrieve the position of the leg joints based on the nearest command index and step index.

            This function calculates the leg joint positions by looking up precomputed values
            and adjusting them based on the waist roll position. It uses the nearest command
            index to find the appropriate lookup table and applies modulo arithmetic with the
            step index to cycle through the lookup values. Adjustments are made to the hip
            roll positions to account for the waist roll.

            Returns:
                ArrayType: An array representing the adjusted positions of the leg joints.
            """
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

        leg_joint_pos = conditional_update(
            is_static_pose, get_leg_joint_pos_init, get_leg_joint_pos
        )
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
