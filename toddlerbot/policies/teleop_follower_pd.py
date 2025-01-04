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
        use_torso_pd: bool = True,
        task: str = "",
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

        self.task = task
        prep = "hold" if task == "hug" else "kneel"

        self.neck_pitch_idx = robot.motor_ordering.index("neck_pitch_act")
        self.neck_pitch_ratio = 1.0  # 0.25  # Stitch
        self.num_arm_motors = 7

        if len(task) > 0:
            self.manip_duration = 2.0

            motion_file_path = os.path.join("toddlerbot", "motion", f"{prep}.pkl")
            if os.path.exists(motion_file_path):
                data_dict = joblib.load(motion_file_path)
            else:
                raise ValueError(f"No data files found in {motion_file_path}")

            self.manip_motor_pos = np.array(data_dict["action_traj"], dtype=np.float32)[
                -1
            ]
            if prep != "kneel":
                self.manip_motor_pos[self.neck_pitch_idx] *= self.neck_pitch_ratio

            if robot.has_gripper:
                self.manip_motor_pos = np.concatenate(
                    [self.manip_motor_pos, np.zeros(2, dtype=np.float32)]
                )

        self.capture_frame = True

        self.dataset_logger = DatasetLogger()

    def get_arm_motor_pos(self, obs: Obs) -> npt.NDArray[np.float32]:
        if self.arm_motor_pos is None:
            return self.manip_motor_pos[self.arm_motor_indices]
        else:
            return self.arm_motor_pos

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        control_inputs, motor_target = super().step(obs, is_real)

        if self.task == "pick" and obs.time - self.prep_duration >= self.manip_duration:
            motor_target[self.neck_motor_indices] = self.manip_motor_pos[self.waist_motor_indices]
            motor_target[self.arm_motor_indices[: self.num_arm_motors]] = (
                self.manip_motor_pos[self.arm_motor_indices[: self.num_arm_motors]]
            )
            motor_target[self.waist_motor_indices] = self.manip_motor_pos[self.waist_motor_indices]
            motor_target[self.leg_motor_indices] = self.manip_motor_pos[
                self.leg_motor_indices
            ]

        # Log the data
        if self.is_ended:
            self.is_ended = False
            self.dataset_logger.save()
            self.reset()
        elif self.is_running:
            self.dataset_logger.log_entry(
                Data(obs.time, obs.motor_pos, self.fsr, self.camera_frame)
            )

        return control_inputs, motor_target
