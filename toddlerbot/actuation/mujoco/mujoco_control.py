from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType
from toddlerbot.utils.array_utils import array_lib as np


class MotorController:
    def __init__(self, robot: Robot):
        self.kp = np.array(robot.get_joint_attrs("type", "dynamixel", "kp_sim"))
        self.kd = np.array(robot.get_joint_attrs("type", "dynamixel", "kd_sim"))
        self.tau_max = np.array(robot.get_joint_attrs("type", "dynamixel", "tau_max"))
        self.q_dot_tau_max = np.array(
            robot.get_joint_attrs("type", "dynamixel", "q_dot_tau_max")
        )
        self.q_dot_max = np.array(
            robot.get_joint_attrs("type", "dynamixel", "q_dot_max")
        )

    def step(self, q: ArrayType, q_dot: ArrayType, a: ArrayType):
        error = a - q
        tau_m = self.kp * error - self.kd * q_dot

        abs_q_dot = np.abs(q_dot)

        # Apply vectorized conditions using np.where
        tau_limit = np.where(
            abs_q_dot <= self.q_dot_tau_max,  # Condition 1
            self.tau_max,  # Value when condition 1 is True
            np.where(
                abs_q_dot <= self.q_dot_max,  # Condition 2
                self.tau_max
                / (self.q_dot_tau_max - self.q_dot_max)
                * (abs_q_dot - self.q_dot_tau_max)
                + self.tau_max,  # Value when condition 2 is True
                np.zeros_like(tau_m),  # Value when all conditions are False
            ),
        )

        tau_m_clamped = np.clip(tau_m, -tau_limit, tau_limit)
        return tau_m_clamped


class PositionController:
    def step(self, q: ArrayType, q_dot: ArrayType, a: ArrayType):
        return a
