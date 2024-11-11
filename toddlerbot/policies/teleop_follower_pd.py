from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.policies.balance_pd import BalancePDPolicy
from toddlerbot.sensing.camera import Camera
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.comm_utils import ZMQNode
from toddlerbot.utils.dataset_utils import Data, DatasetLogger


class TeleopFollowerPDPolicy(BalancePDPolicy, policy_name="teleop_follower_pd"):
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

        self.dataset_logger = DatasetLogger()

    def get_arm_motor_pos(self, obs: Obs) -> npt.NDArray[np.float32]:
        if self.arm_motor_pos is None:
            return self.default_motor_pos[self.arm_motor_indices]
        else:
            return self.arm_motor_pos

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        command, motor_target = super().step(obs, is_real)

        # Log the data
        if self.is_ended:
            self.is_ended = False
            self.dataset_logger.save()
        elif self.is_running:
            self.dataset_logger.log_entry(
                Data(obs.time, obs.motor_pos, self.fsr, self.camera_frame)
            )

        return command, motor_target
