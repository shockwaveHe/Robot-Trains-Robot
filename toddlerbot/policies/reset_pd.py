from typing import Dict, List, Optional, Tuple

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
        cameras: Optional[List[Camera]] = None,
        zmq_receiver: Optional[ZMQNode] = None,
        zmq_sender: Optional[ZMQNode] = None,
        ip: str = "",
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
        use_torso_pd: bool = True,
    ):
        super().__init__(
            name,
            robot,
            init_motor_pos,
            joystick,
            cameras,
            zmq_receiver,
            zmq_sender,
            ip,
            fixed_command,
            use_torso_pd,
        )
        self.reset_motor_indices = np.arange(robot.nu)
        self.reset_vel = 0.3
        self.reset_time = None

    def get_arm_motor_pos(self, obs: Obs) -> npt.NDArray[np.float32]:
        return obs.motor_pos[self.arm_motor_indices]

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        # Log the data
        if self.is_button_pressed and self.reset_time is None:
            pos_curr = obs.motor_pos[self.reset_motor_indices]
            pos_target = self.default_motor_pos[self.reset_motor_indices]
            self.reset_time, self.reset_action = self.move(
                obs.time - self.control_dt,
                pos_curr,
                pos_target,
                np.max(np.abs((pos_target - pos_curr) / self.reset_vel)),
                end_time=0.5,
            )

        control_inputs, motor_target = super().step(obs, is_real)

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

        return control_inputs, motor_target
