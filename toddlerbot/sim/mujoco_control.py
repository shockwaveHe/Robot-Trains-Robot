from typing import Optional

from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType
from toddlerbot.utils.array_utils import array_lib as np


class MotorController:
    """A class for controlling the Dynamixel motors of a robot."""

    def __init__(self, robot: Robot):
        """Initializes the control parameters for a robot's joints using attributes specific to "dynamixel" type actuators.

        Args:
            robot (Robot): An instance of the Robot class from which joint attributes are retrieved.
        """
        self.kp = np.array(robot.get_joint_attrs("type", "dynamixel", "kp_sim"))
        self.kd = np.array(robot.get_joint_attrs("type", "dynamixel", "kd_sim"))
        self.tau_max = np.array(robot.get_joint_attrs("type", "dynamixel", "tau_max"))
        self.q_dot_max = np.array(
            robot.get_joint_attrs("type", "dynamixel", "q_dot_max")
        )
        self.tau_q_dot_max = np.array(
            robot.get_joint_attrs("type", "dynamixel", "tau_q_dot_max")
        )
        self.q_dot_tau_max = np.array(
            robot.get_joint_attrs("type", "dynamixel", "q_dot_tau_max")
        )
        self.coeffs = np.array(robot.get_joint_attrs("type", "dynamixel", "coeffs"))

    def step(
        self,
        q: ArrayType,
        q_dot: ArrayType,
        a: ArrayType,
        noise: Optional[ArrayType] = None,
    ):
        """Computes the clamped torque for a control step based on position, velocity, and desired acceleration.

        Args:
            q (ArrayType): Current position array.
            q_dot (ArrayType): Current velocity array.
            a (ArrayType): Desired acceleration array.
            kp (Optional[ArrayType]): Proportional gain array. Defaults to self.kp.
            kd (Optional[ArrayType]): Derivative gain array. Defaults to self.kd.
            tau_max (Optional[ArrayType]): Maximum torque array. Defaults to self.tau_max.
            q_dot_tau_max (Optional[ArrayType]): Velocity threshold for maximum torque. Defaults to self.q_dot_tau_max.
            q_dot_max (Optional[ArrayType]): Maximum velocity array. Defaults to self.q_dot_max.

        Returns:
            ArrayType: Clamped torque array based on the computed control law and constraints.
        """

        error = a - q
        tau_m = self.kp * error - self.kd * q_dot

        abs_q_dot = np.abs(q_dot)

        # Apply vectorized conditions using np.where
        tau_limit = np.where(
            abs_q_dot <= self.q_dot_tau_max,  # Condition 1
            self.tau_max,  # Value when condition 1 is True
            np.where(
                abs_q_dot <= self.q_dot_max,  # Condition 2
                self.coeffs[:, 0] * abs_q_dot**2
                + self.coeffs[:, 1] * abs_q_dot
                + self.coeffs[:, 2],  # Value when condition 2 is True
                self.tau_q_dot_max,  # Value when all conditions are False
            ),
        )

        if noise is not None:
            tau_limit += noise

        tau_m_clamped = np.clip(tau_m, -tau_limit, tau_limit)

        # print(f"tau_m_clamped: {tau_m_clamped}, tau_limit: {tau_limit}, tau_m: {tau_m}")

        return tau_m_clamped


class PositionController:
    """A class for controlling the position of a robot's joints."""

    def step(self, q: ArrayType, q_dot: ArrayType, a: ArrayType):
        """Advances the system state by one time step using the provided acceleration.

        Args:
            q (ArrayType): The current state vector of the system.
            q_dot (ArrayType): The current velocity vector of the system.
            a (ArrayType): The acceleration vector to be applied.

        Returns:
            ArrayType: The acceleration vector `a`.
        """
        return a


class JointController:
    def step(self, q: ArrayType):
        return q
