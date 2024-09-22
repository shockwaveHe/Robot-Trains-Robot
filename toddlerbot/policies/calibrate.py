import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate_action


class CalibratePolicy(BasePolicy, policy_name="calibrate"):
    def __init__(
        self, name: str, robot: Robot, init_motor_pos: npt.NDArray[np.float32]
    ):
        super().__init__(name, robot, init_motor_pos)

        self.default_motor_pos = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )
        self.default_joint_pos = np.array(
            list(robot.default_joint_angles.values()), dtype=np.float32
        )

        self.left_hip_pitch_idx = robot.joint_ordering.index("left_hip_pitch")
        self.right_hip_pitch_idx = robot.joint_ordering.index("right_hip_pitch")
        self.left_knee_pitch_idx = robot.joint_ordering.index("left_knee_pitch")
        self.right_knee_pitch_idx = robot.joint_ordering.index("right_knee_pitch")
        self.left_ank_pitch_idx = robot.joint_ordering.index("left_ank_pitch")
        self.right_ank_pitch_idx = robot.joint_ordering.index("right_ank_pitch")

        self.prep_duration = 2.0
        self.prep_time, self.prep_action = self.move(
            -self.control_dt, init_motor_pos, self.default_motor_pos, self.prep_duration
        )

        # PD controller parameters
        self.kp = 1.0  # Proportional gain
        self.kd = 0.1  # Derivative gain
        self.previous_error = 0.0

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        # Preparation phase
        if obs.time < self.prep_time[-1]:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return action

        # PD controller to maintain torso pitch at 0
        pitch_curr = obs.euler[1]  # Torso pitch angle (obs.euler[1])
        # Calculate the error (difference between desired and current pitch)
        error = pitch_curr

        # Derivative of the error (rate of change)
        error_derivative = (error - self.previous_error) / self.control_dt
        self.previous_error = error

        # PD controller output
        ctrl = self.kp * error + self.kd * error_derivative

        # Update joint positions based on the PD controller command
        joint_pos = self.default_joint_pos.copy()

        joint_pos[self.left_hip_pitch_idx] += ctrl
        joint_pos[self.right_hip_pitch_idx] -= ctrl
        joint_pos[self.left_knee_pitch_idx] += ctrl
        joint_pos[self.right_knee_pitch_idx] -= ctrl
        joint_pos[self.left_ank_pitch_idx] += ctrl
        joint_pos[self.right_ank_pitch_idx] += ctrl

        # Convert joint positions to motor angles
        motor_angles = self.robot.joint_to_motor_angles(
            dict(zip(self.robot.joint_ordering, joint_pos))
        )
        motor_target = np.array(list(motor_angles.values()), dtype=np.float32)

        return motor_target
