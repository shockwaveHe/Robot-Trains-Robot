from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.policies.balance_pd import BalancePDPolicy
from toddlerbot.sensing.camera import Camera
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.comm_utils import ZMQNode
from toddlerbot.utils.math_utils import interpolate_action


class ResetPDPolicy(BalancePDPolicy, policy_name="reset_pd"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        joystick: Optional[Joystick] = None,
        camera: Optional[Camera] = None,
        zmq_receiver: Optional[ZMQNode] = None,
        zmq_sender: Optional[ZMQNode] = None,
        ip: str = "127.0.0.1",
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        super().__init__(
            name,
            robot,
            init_motor_pos,
            joystick,
            camera,
            zmq_receiver,
            zmq_sender,
            ip,
            fixed_command,
        )

        self.reset_duration = 5.0
        self.reset_end_time = 1.0
        self.reset_time = None

    def get_arm_motor_pos(self, obs: Obs) -> npt.NDArray[np.float32]:
        return obs.motor_pos[self.arm_motor_indices]

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        command, motor_target = super().step(obs, is_real)

        # Log the data
        if self.is_button_pressed and self.reset_time is None:
            self.reset_time, self.reset_action = self.move(
                obs.time - self.control_dt,
                obs.motor_pos[self.reset_motor_indices],
                self.default_motor_pos[self.reset_motor_indices],
                self.reset_duration,
                end_time=self.reset_end_time,
            )

        if self.reset_time is not None:
            if obs.time < self.reset_time[-1]:
                reset_motor_pos = np.asarray(
                    interpolate_action(obs.time, self.reset_time, self.reset_action)
                )
                motor_target[self.reset_motor_indices] = reset_motor_pos
            else:
                motor_target[self.reset_motor_indices] = self.default_motor_pos[
                    self.reset_motor_indices
                ]
                self.reset_time = None

        return command, motor_target
