import platform
from typing import Dict, Optional

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

    def get_command(self, control_inputs: Dict[str, float]) -> npt.NDArray[np.float32]:
        command = np.zeros(len(self.command_range), dtype=np.float32)
        for task, input in control_inputs.items():
            if task == "manipulate":
                if abs(input) > 0.5:
                    # Button is pressed
                    if not self.is_button_pressed:
                        self.is_button_pressed = True  # Mark the button as pressed
                        self.is_logging = not self.is_logging  # Toggle logging

                        # Log the episode end if logging is toggled to off
                        if not self.is_logging:
                            self.dataset_logger.log_episode_end()
                            print(f"\nLogged {self.n_logs} entries.")
                            self.n_logs += 1

                        print(
                            f"\nLogging is now {'enabled' if self.is_logging else 'disabled'}."
                        )
                else:
                    # Button is released
                    self.is_button_pressed = False  # Reset button pressed state

            elif task == "look_left" and input > 0:
                command[0] = input * self.command_range[0][1]
            elif task == "look_right" and input > 0:
                command[0] = input * self.command_range[0][0]
            elif task == "look_up" and input > 0:
                command[1] = input * self.command_range[1][1]
            elif task == "look_down" and input > 0:
                command[1] = input * self.command_range[1][0]
            elif task == "lean_left" and input > 0:
                command[3] = input * self.command_range[3][0]
            elif task == "lean_right" and input > 0:
                command[3] = input * self.command_range[3][1]
            elif task == "twist_left" and input > 0:
                command[4] = input * self.command_range[4][0]
            elif task == "twist_right" and input > 0:
                command[4] = input * self.command_range[4][1]

        return command

    def get_arm_motor_pos(self) -> npt.NDArray[np.float32]:
        return self.arm_motor_pos

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        motor_target = super().step(obs, is_real)
        # Log the data
        if self.is_logging:
            self.dataset_logger.log_entry(
                Data(obs.time, obs.motor_pos, self.fsr, self.camera_frame)
            )
        else:
            # clean up the log when not logging.
            self.dataset_logger.maintain_log()

        return motor_target
