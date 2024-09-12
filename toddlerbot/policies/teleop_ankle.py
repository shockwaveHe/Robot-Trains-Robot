import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.teleop.joystick import get_controller_input, initialize_joystick
from toddlerbot.utils.math_utils import interpolate_action


class TeleopAnklePolicy(BasePolicy, policy_name="teleop_ankle"):
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

        self.ank_pitch_command_range = [-0.1, 0.1]
        self.left_ank_pitch_idx = robot.joint_ordering.index("left_ank_pitch")
        self.right_ank_pitch_idx = robot.joint_ordering.index("right_ank_pitch")

        self.joystick = None
        try:
            self.joystick = initialize_joystick()
        except Exception:
            pass

        self.prep_duration = 2.0
        self.prep_time, self.prep_action = self.move(
            -self.control_dt, init_motor_pos, self.default_motor_pos, self.prep_duration
        )

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        if obs.time < self.prep_time[-1]:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return action

        if self.joystick is None:
            raise ValueError("Joystick is required for this policy.")
        else:
            command = np.array(
                get_controller_input(self.joystick, [self.ank_pitch_command_range]),
                dtype=np.float32,
            )

        joint_pos = self.default_joint_pos.copy()
        joint_pos[self.left_ank_pitch_idx] += command[0]
        joint_pos[self.right_ank_pitch_idx] += command[0]

        motor_angles = self.robot.joint_to_motor_angles(
            dict(zip(self.robot.joint_ordering, joint_pos))
        )
        motor_pos = np.array(list(motor_angles.values()), dtype=np.float32)

        return motor_pos
