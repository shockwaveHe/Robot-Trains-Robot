import platform
from typing import Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.policies.balance_pd import BalancePDPolicy
from toddlerbot.sensing.camera import Camera
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.comm_utils import ZMQNode
from toddlerbot.utils.dataset_utils import Data, DatasetLogger

SYS_NAME = platform.system()


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

    def get_arm_motor_pos(self) -> npt.NDArray[np.float32]:
        return self.arm_motor_pos

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        motor_target = super().step(obs, is_real)
        # Log the data
        if self.is_logging:
            self.dataset_logger.log_entry(
                Data(obs.time, obs.motor_pos, self.fsr, self.camera_frame)
            )
        elif self.is_button_pressed:
            self.dataset_logger.log_episode_end()
            print(f"\nLogged {self.n_logs} entries.")
            self.n_logs += 1
        else:
            # clean up the log when not logging.
            self.dataset_logger.maintain_log()

        return motor_target
