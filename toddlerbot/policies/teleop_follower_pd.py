import os
from typing import Dict, List, Optional, Tuple

import joblib
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
        cameras: Optional[List[Camera]] = None,
        zmq_receiver: Optional[ZMQNode] = None,
        zmq_sender: Optional[ZMQNode] = None,
        ip: str = "",
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
        prep: str = "",
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
        )

        self.neck_pitch_idx = robot.motor_ordering.index("neck_pitch_act")
        self.neck_pitch_ratio = 0.25 # Stitch

        if len(prep) > 0:
            motion_file_path = os.path.join("toddlerbot", "motion", f"{prep}.pkl")
            if os.path.exists(motion_file_path):
                data_dict = joblib.load(motion_file_path)
            else:
                raise ValueError(f"No data files found in {motion_file_path}")

            self.prep_motor_pos = np.array(data_dict["action_traj"], dtype=np.float32)[
                -1
            ]
            self.prep_motor_pos[self.neck_pitch_idx] *= self.neck_pitch_ratio

        self.capture_frame = True

        self.dataset_logger = DatasetLogger()

    def get_arm_motor_pos(self, obs: Obs) -> npt.NDArray[np.float32]:
        if self.arm_motor_pos is None:
            return self.prep_motor_pos[self.arm_motor_indices]
        else:
            return self.arm_motor_pos

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        control_inputs, motor_target = super().step(obs, is_real)

        # Log the data
        if self.is_ended:
            self.is_ended = False
            self.dataset_logger.save()
        elif self.is_running:
            self.dataset_logger.log_entry(
                Data(obs.time, obs.motor_pos, self.fsr, self.camera_frame)
            )

        return control_inputs, motor_target
