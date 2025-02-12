from typing import Tuple

from toddlerbot.reference.motion_ref import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np


class WalkSimpleReference(MotionReference):
    def __init__(
        self,
        robot: Robot,
        dt: float,
        cycle_time: float,
        max_knee: float = np.pi / 3,
        double_support_phase: float = 0.1,
    ):
        """Initializes the walking controller for a robot with specified parameters.

        Args:
            robot (Robot): The robot instance to be controlled.
            dt (float): The time step for each control cycle.
            cycle_time (float): The total time for one walking cycle.
            max_knee (float, optional): The maximum knee angle in radians. Defaults to π/3.
            double_support_phase (float, optional): The fraction of the cycle time spent in double support phase. Defaults to 0.1.
        """
        super().__init__("walk_simple", "periodic", robot, dt)

        self.cycle_time = cycle_time

        self.max_knee = max_knee
        self.double_support_phase = double_support_phase

        self.num_joints = len(self.robot.joint_ordering)
        self.shin_thigh_ratio = (
            self.robot.config["general"]["offsets"]["knee_to_ank_pitch_z"]
            / self.robot.config["general"]["offsets"]["hip_pitch_to_knee_z"]
        )

        self.left_pitch_joint_indices = np.array(
            [
                self.robot.joint_ordering.index("left_hip_pitch"),
                self.robot.joint_ordering.index("left_knee"),
                self.robot.joint_ordering.index("left_ank_pitch"),
            ]
        )
        self.right_pitch_joint_indices = np.array(
            [
                self.robot.joint_ordering.index("right_hip_pitch"),
                self.robot.joint_ordering.index("right_knee"),
                self.robot.joint_ordering.index("right_ank_pitch"),
            ]
        )

    def get_phase_signal(self, time_curr: float | ArrayType) -> ArrayType:
        """Calculate the phase signal as a sine and cosine pair for a given time.

        Args:
            time_curr (float | ArrayType): The current time or an array of time values.

        Returns:
            ArrayType: An array containing the sine and cosine of the phase, with dtype float32.
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
        """Calculates linear and angular velocity vectors from a given command.

        Args:
            command (ArrayType): A 3-element array representing the desired velocities,
                where the first two elements are linear velocities in the x and y directions,
                and the third element is the angular velocity around the z-axis.

        Returns:
            Tuple[ArrayType, ArrayType]: A tuple containing two arrays:
                - The first array is the linear velocity vector with the z-component set to zero.
                - The second array is the angular velocity vector with non-zero z-component.
        """
        lin_vel = np.array([command[0], command[1], 0.0], dtype=np.float32)
        ang_vel = np.array([0.0, 0.0, command[2]], dtype=np.float32)
        return lin_vel, ang_vel

    def get_state_ref(
        self, state_curr: ArrayType, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        """Calculates the reference state for a robotic system based on the current state, time, and command inputs.

        This function integrates the current state with the command to determine the path state, computes sinusoidal phase signals to adjust leg positions, and updates motor and joint positions accordingly. It also determines the stance mask based on the phase signal and double support phase.

        Args:
            state_curr (ArrayType): The current state of the system.
            time_curr (float | ArrayType): The current time or an array of time values.
            command (ArrayType): The command input to influence the state transition.

        Returns:
            ArrayType: The concatenated array of path state, motor positions, joint positions, and stance mask.
        """
        path_state = self.integrate_path_state(state_curr, command)

        sin_phase_signal = np.sin(2 * np.pi * time_curr / self.cycle_time)
        signal_left = np.clip(sin_phase_signal, 0, None)
        signal_right = np.clip(sin_phase_signal, None, 0)

        left_leg_pitch_pos = self.get_leg_pitch_pos(signal_left, True)
        right_leg_pitch_pos = self.get_leg_pitch_pos(signal_right, False)

        motor_pos = self.default_motor_pos.copy()
        joint_pos = self.default_joint_pos.copy()
        joint_pos = inplace_update(
            joint_pos, self.left_pitch_joint_indices, left_leg_pitch_pos
        )
        joint_pos = inplace_update(
            joint_pos, self.right_pitch_joint_indices, right_leg_pitch_pos
        )
        double_support_mask = np.abs(sin_phase_signal) < self.double_support_phase
        joint_pos = np.where(double_support_mask, self.default_joint_pos, joint_pos)

        stance_mask = np.zeros(2, dtype=np.float32)
        stance_mask = inplace_update(stance_mask, 0, np.any(sin_phase_signal >= 0))
        stance_mask = inplace_update(stance_mask, 1, np.any(sin_phase_signal < 0))
        stance_mask = np.where(double_support_mask, 1, stance_mask)

        return np.concatenate((path_state, motor_pos, joint_pos, stance_mask))

    def get_leg_pitch_pos(self, signal: ArrayType, is_left: bool):
        """Calculates the pitch positions of the leg joints based on the input signal and leg side.

        Args:
            signal (ArrayType): The input signal representing the desired knee angle.
            is_left (bool): Indicates whether the calculations are for the left leg.

        Returns:
            np.ndarray: An array containing the hip, knee, and ankle pitch angles in radians.
        """
        knee_angle = np.abs(
            signal * (self.max_knee - self.knee_default)
            + (2 * int(is_left) - 1) * self.knee_default
        )
        ank_pitch_angle = np.arctan2(
            np.sin(knee_angle),
            np.cos(knee_angle) + self.shin_thigh_ratio,
        )
        hip_pitch_angle = knee_angle - ank_pitch_angle

        if is_left:
            return np.array(
                [-hip_pitch_angle, knee_angle, -ank_pitch_angle], dtype=np.float32
            )
        else:
            return np.array(
                [hip_pitch_angle, -knee_angle, -ank_pitch_angle], dtype=np.float32
            )
