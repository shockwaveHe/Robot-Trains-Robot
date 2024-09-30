from typing import Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.policies.balance_pd import BalancePDPolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick


class LookPDPolicy(BalancePDPolicy, policy_name="look_pd"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        joystick: Optional[Joystick] = None,
    ):
        super().__init__(name, robot, init_motor_pos)

        self.neck_yaw_idx = robot.joint_ordering.index("neck_yaw_driven")
        self.neck_pitch_idx = robot.joint_ordering.index("neck_pitch_driven")
        self.neck_yaw_limits = robot.joint_limits["neck_yaw_driven"]
        self.neck_pitch_limits = robot.joint_limits["neck_pitch_driven"]

        self.neck_yaw_target = 0.0
        self.neck_pitch_target = 0.0
        self.joint_target_curr = None

        if joystick is None:
            self.joystick = Joystick()
        else:
            self.joystick = joystick

    def reset(self):
        super().reset()
        self.joint_target_curr = None

    def get_joint_target(self, obs: Obs, time_curr: float) -> npt.NDArray[np.float32]:
        control_inputs = self.joystick.get_controller_input()
        command = np.zeros(2, dtype=np.float32)
        for task, input in control_inputs.items():
            if task == "look_up" and input > 0:
                command[1] = input
            elif task == "look_down" and input > 0:
                command[1] = -input
            elif task == "look_left" and input > 0:
                command[0] = input
            elif task == "look_right" and input > 0:
                command[0] = -input

        self.neck_yaw_target = np.clip(
            self.neck_yaw_target + command[0] * self.control_dt,
            *self.neck_yaw_limits,
        )
        self.neck_pitch_target = np.clip(
            self.neck_pitch_target + command[1] * self.control_dt,
            *self.neck_pitch_limits,
        )

        if self.joint_target_curr is None:
            joint_angles = self.robot.motor_to_joint_angles(
                dict(zip(self.robot.motor_ordering, obs.motor_pos))
            )
            self.joint_target_curr = np.array(
                list(joint_angles.values()), dtype=np.float32
            )

        joint_target = self.joint_target_curr.copy()
        joint_target[self.neck_yaw_idx] = self.neck_yaw_target
        joint_target[self.neck_pitch_idx] = self.neck_pitch_target

        return joint_target
