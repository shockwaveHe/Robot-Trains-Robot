from typing import Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.ref_motion.balance_ref import BalanceReference
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate_action

default_pose = np.array(
    [
        -0.6028545,
        -0.90198064,
        0.01840782,
        1.2379225,
        0.52615595,
        0.4985056,
        -1.1320779,
        0.5031457,
        -0.9372623,
        -0.248505,
        1.2179809,
        -0.35434943,
        -0.6473398,
        -1.1581556,
    ]
)


class BalancePDPolicy(BasePolicy, policy_name="balance_pd"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        super().__init__(name, robot, init_motor_pos)

        if fixed_command is None:
            self.fixed_command = np.array([0.0], dtype=np.float32)
        else:
            self.fixed_command = fixed_command

        self.default_motor_pos = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )
        self.default_joint_pos = np.array(
            list(robot.default_joint_angles.values()), dtype=np.float32
        )
        self.motor_limits = np.array(
            [robot.joint_limits[name] for name in robot.motor_ordering]
        )

        # Indices for the pitch joints
        self.left_hip_pitch_idx = robot.joint_ordering.index("left_hip_pitch")
        self.right_hip_pitch_idx = robot.joint_ordering.index("right_hip_pitch")
        self.left_knee_pitch_idx = robot.joint_ordering.index("left_knee_pitch")
        self.right_knee_pitch_idx = robot.joint_ordering.index("right_knee_pitch")
        self.left_ank_pitch_idx = robot.joint_ordering.index("left_ank_pitch")
        self.right_ank_pitch_idx = robot.joint_ordering.index("right_ank_pitch")

        self.left_hip_roll_idx = robot.joint_ordering.index("left_hip_roll")
        self.right_hip_roll_idx = robot.joint_ordering.index("right_hip_roll")
        self.left_ank_roll_idx = robot.joint_ordering.index("left_ank_roll")
        self.right_ank_roll_idx = robot.joint_ordering.index("right_ank_roll")

        teleop_default_motor_pos = self.default_motor_pos.copy()
        arm_motor_slice = slice(
            robot.motor_ordering.index("left_sho_pitch"),
            robot.motor_ordering.index("right_wrist_roll") + 1,
        )
        teleop_default_motor_pos[arm_motor_slice] = default_pose

        self.motion_ref = BalanceReference(robot)

        self.prep_duration = 7.0
        self.prep_time, self.prep_action = self.move(
            -self.control_dt,
            init_motor_pos,
            teleop_default_motor_pos,
            self.prep_duration,
            end_time=5.0,
        )

        # PD controller parameters
        self.hip_pitch_kp = 5.0
        self.hip_pitch_kd = 0.01
        self.knee_pitch_kp = 5.0
        self.knee_pitch_kd = 0.01
        self.ankle_pitch_kp = 5.0  
        self.ankle_pitch_kd = 0.1
        self.hip_roll_kp = 0.0
        self.hip_roll_kd = 0.0
        self.ankle_roll_kp = 0.0
        self.ankle_roll_kd = 0.0
        
        self.step_curr = 0
        self.previous_error = np.zeros(2, dtype=np.float32)

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        # Preparation phase
        if obs.time < self.prep_time[-1]:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return action

        command = self.fixed_command

        time_curr = self.step_curr * self.control_dt
        state_ref = self.motion_ref.get_state_ref(
            np.zeros(3, dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            time_curr,
            command,
        )

        error = obs.com_pos[:2]
        error_derivative = (error - self.previous_error) / self.control_dt
        self.previous_error = error

        # Update joint positions based on the PD controller command
        joint_pos = self.default_joint_pos.copy()

        # Distribute the command across hip, knee, and ankle joints
        hip_pitch_ctrl = self.hip_pitch_kp * error[0] - self.hip_pitch_kd * error_derivative[0]
        knee_pitch_ctrl = self.knee_pitch_kp * error[0] - self.knee_pitch_kd * error_derivative[0]
        ankle_pitch_ctrl = self.ankle_pitch_kp * error[0] - self.ankle_pitch_kd * error_derivative[0]
        hip_roll_ctrl = self.hip_roll_kp * error[1] - self.hip_roll_kd * error_derivative[1]
        ankle_roll_ctrl = self.ankle_roll_kp * error[1] - self.ankle_roll_kd * error_derivative[1]

        joint_pos[self.left_hip_pitch_idx] += hip_pitch_ctrl
        joint_pos[self.left_knee_pitch_idx] += knee_pitch_ctrl
        joint_pos[self.left_ank_pitch_idx] += ankle_pitch_ctrl
        joint_pos[self.right_hip_pitch_idx] += -hip_pitch_ctrl
        joint_pos[self.right_knee_pitch_idx] += -knee_pitch_ctrl
        joint_pos[self.right_ank_pitch_idx] += ankle_pitch_ctrl

        joint_pos[self.left_hip_roll_idx] += -hip_roll_ctrl
        joint_pos[self.left_ank_roll_idx] += -ankle_roll_ctrl
        joint_pos[self.right_hip_roll_idx] += hip_roll_ctrl
        joint_pos[self.right_ank_roll_idx] += -ankle_roll_ctrl

        # Convert joint positions to motor angles
        motor_angles = self.robot.joint_to_motor_angles(
            dict(zip(self.robot.joint_ordering, joint_pos))
        )
        motor_target = np.array(list(motor_angles.values()), dtype=np.float32)

        motor_target = np.asarray(
            self.motion_ref.override_motor_target(motor_target, state_ref)
        )
        motor_target = np.clip(
            motor_target, self.motor_limits[:, 0], self.motor_limits[:, 1]
        )

        self.step_curr += 1

        return motor_target
