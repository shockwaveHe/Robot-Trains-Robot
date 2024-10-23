import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate_action


class CalibratePolicy(BasePolicy, policy_name="calibrate"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        kp: float = 0.1,
        kd: float = 0.01,
        ki: float = 0.2,
    ):
        super().__init__(name, robot, init_motor_pos)

        self.default_motor_pos = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )
        self.default_joint_pos = np.array(
            list(robot.default_joint_angles.values()), dtype=np.float32
        )

        self.leg_pitch_joint_indicies = np.array(
            [
                self.robot.joint_ordering.index(joint_name)
                for joint_name in [
                    "left_hip_pitch",
                    "left_knee",
                    "left_ank_pitch",
                    "right_hip_pitch",
                    "right_knee",
                    "right_ank_pitch",
                ]
            ]
        )
        self.leg_pitch_joint_signs = np.array([1, 1, 1, -1, -1, 1], dtype=np.float32)

        # PD controller parameters
        self.kp = kp
        self.kd = kd
        self.ki = ki

        # Initialize integral error
        self.integral_error = 0.0

        self.prep_duration = 2.0
        self.prep_time, self.prep_action = self.move(
            -self.control_dt, init_motor_pos, self.default_motor_pos, self.prep_duration
        )

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        # Preparation phase
        if obs.time < self.prep_time[-1]:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return action

        # PD+I controller to maintain torso pitch at 0
        error = obs.torso_euler[1]
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

        return motor_target
