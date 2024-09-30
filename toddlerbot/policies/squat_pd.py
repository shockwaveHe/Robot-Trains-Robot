from typing import Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.policies.balance_pd import BalancePDPolicy
from toddlerbot.ref_motion.squat_ref import SquatReference
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick


class SquatPDPolicy(BalancePDPolicy, policy_name="squat_pd"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        joystick: Optional[Joystick] = None,
        squat_speed=0.03,
    ):
        super().__init__(name, robot, init_motor_pos)

        self.squat_speed = squat_speed
        self.motion_ref = SquatReference(robot, self.control_dt)

        if joystick is None:
            self.joystick = Joystick()
        else:
            self.joystick = joystick

    def reset(self):
        super().reset()
        self.motion_ref.reset()

    def get_joint_target(self, obs: Obs, time_curr: float) -> npt.NDArray[np.float32]:
        control_inputs = self.joystick.get_controller_input()
        command = np.zeros(2, dtype=np.float32)
        for task, input in control_inputs.items():
            if task == "squat":
                command[0] = -input * self.squat_speed

        joint_target = self.motion_ref.get_state_ref(
            np.zeros(3, dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            time_curr,
            command,
        )[13 : 13 + self.robot.nu]

        joint_angles = self.robot.motor_to_joint_angles(
            dict(zip(self.robot.motor_ordering, obs.motor_pos))
        )
        joint_target[self.neck_yaw_idx] = joint_angles["neck_yaw_driven"]
        joint_target[self.neck_pitch_idx] = joint_angles["neck_pitch_driven"]

        return joint_target
