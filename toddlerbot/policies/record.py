from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.policies.balance_pd import BalancePDPolicy
from toddlerbot.sensing.camera import Camera
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.comm_utils import ZMQNode


class RecordPolicy(BalancePDPolicy, policy_name="record"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        joystick: Optional[Joystick] = None,
        camera: Optional[Camera] = None,
        zmq_receiver: Optional[ZMQNode] = None,
        zmq_sender: Optional[ZMQNode] = None,
        ip: str = "",
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

        self.disable_motor_indices = np.concatenate([self.leg_motor_indices])

        self.is_prepared = False
        self.is_running = False
        self.toggle_motor = False

    def get_arm_motor_pos(self, obs: Obs) -> npt.NDArray[np.float32]:
        if self.arm_motor_pos is None:
            return self.default_motor_pos[self.arm_motor_indices]
        else:
            return self.arm_motor_pos

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        control_inputs, motor_target = super().step(obs, is_real)

        if obs.time >= self.prep_duration:
            if not self.is_running:
                self.is_running = True
                self.toggle_motor = True

            motor_target[self.disable_motor_indices] = obs.motor_pos[
                self.disable_motor_indices
            ]

        return control_inputs, motor_target
