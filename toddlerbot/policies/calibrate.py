from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate_action

# This script ensures a more accurate zero-point calibration by running a PID loop with the robot's torso pitch.


class CalibratePolicy(BasePolicy, policy_name="calibrate"):
    """Policy for calibrating zero point with the robot's torso pitch."""

    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        kp: float = 0.1,
        kd: float = 0.01,
        ki: float = 0.2,
    ):
        """Initializes the controller with specified parameters and robot configuration.

        Args:
            name (str): The name of the controller.
            robot (Robot): The robot instance to be controlled.
            init_motor_pos (npt.NDArray[np.float32]): Initial motor positions.
            kp (float, optional): Proportional gain for the controller. Defaults to 0.1.
            kd (float, optional): Derivative gain for the controller. Defaults to 0.01.
            ki (float, optional): Integral gain for the controller. Defaults to 0.2.
        """
        super().__init__(name, robot, init_motor_pos)

        self.default_motor_pos = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )
        self.default_joint_pos = np.array(
            list(robot.default_joint_angles.values()), dtype=np.float32
        )

        leg_pitch_joint_names = [
            "left_hip_pitch",
            "left_knee",
            "left_ank_pitch",
            "right_hip_pitch",
            "right_knee",
            "right_ank_pitch",
        ]
        self.leg_pitch_joint_indicies = np.array(
            [
                self.robot.joint_ordering.index(joint_name)
                for joint_name in leg_pitch_joint_names
            ]
        )
        self.leg_pitch_joint_signs = np.array([-1, -1, 1, 1, 1, -1], dtype=np.float32)

        # PD controller parameters
        self.kp = kp
        self.kd = kd
        self.ki = ki

        # Initialize integral error
        self.integral_error = 0.0

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Executes a control step to maintain the torso pitch at zero using a PD+I controller.

        Args:
            obs (Obs): The current observation containing state information such as time, Euler angles, and angular velocities.
            is_real (bool, optional): Flag indicating whether the step is being executed in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing an empty dictionary and an array of motor target angles.
        """
        # Preparation phase
        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return {}, action

        # PD+I controller to maintain torso pitch at 0
        error = obs.euler[1] + 0.05  # 0.05 cancels some backlash
        error_derivative = obs.ang_vel[1]

        # Update integral error (with a basic anti-windup mechanism)
        self.integral_error += error * self.control_dt
        self.integral_error = np.clip(self.integral_error, -10.0, 10.0)  # Anti-windup

        # PID controller output
        ctrl = (
            self.kp * error + self.ki * self.integral_error - self.kd * error_derivative
        )

        # Update joint positions based on the PID controller command
        joint_pos = self.default_joint_pos.copy()
        joint_pos[self.leg_pitch_joint_indicies] += self.leg_pitch_joint_signs * ctrl

        # Convert joint positions to motor angles
        motor_angles = self.robot.joint_to_motor_angles(
            dict(zip(self.robot.joint_ordering, joint_pos))
        )
        motor_target = np.array(list(motor_angles.values()), dtype=np.float32)

        return {}, motor_target
