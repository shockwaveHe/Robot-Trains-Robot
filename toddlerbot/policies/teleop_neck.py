import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.math_utils import interpolate_action


class TeleopNeckPolicy(BasePolicy, policy_name="teleop_neck"):
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

        self.neck_yaw_idx = robot.joint_ordering.index("neck_yaw_driven")
        self.neck_pitch_idx = robot.joint_ordering.index("neck_pitch_driven")
        self.neck_yaw_target = 0.0
        self.neck_pitch_target = 0.0
        self.neck_yaw_limits = robot.joint_limits["neck_yaw_driven"]
        self.neck_pitch_limits = robot.joint_limits["neck_pitch_driven"]

        self.joystick = None
        try:
            self.joystick = Joystick()
        except Exception:
            pass

        self.prep_duration = 2.0
        self.prep_time, self.prep_action = self.move(
            -self.control_dt, init_motor_pos, self.default_motor_pos, self.prep_duration
        )

    def get_command(self) -> npt.NDArray[np.float32]:
        if self.joystick is None:
            raise ValueError("Joystick is required for this policy.")
        else:
            control_inputs = self.joystick.get_controller_input()
            command = np.zeros(2, dtype=np.float32)
            for task, input in control_inputs.items():
                if task == "neck_up" and input > 0:
                    command[1] = input
                elif task == "neck_down" and input > 0:
                    command[1] = -input
                elif task == "neck_left" and input > 0:
                    command[0] = input
                elif task == "neck_right" and input > 0:
                    command[0] = -input

        return command

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        if obs.time < self.prep_time[-1]:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return action

        command = self.get_command()

        self.neck_yaw_target = np.clip(
            self.neck_yaw_target + command[0] * self.control_dt,
            *self.neck_yaw_limits,
        )
        self.neck_pitch_target = np.clip(
            self.neck_pitch_target + command[1] * self.control_dt,
            *self.neck_pitch_limits,
        )

        joint_pos = self.default_joint_pos.copy()
        joint_pos[self.neck_yaw_idx] = self.neck_yaw_target
        joint_pos[self.neck_pitch_idx] = self.neck_pitch_target

        motor_angles = self.robot.joint_to_motor_angles(
            dict(zip(self.robot.joint_ordering, joint_pos))
        )
        motor_pos = np.array(list(motor_angles.values()), dtype=np.float32)

        return motor_pos
