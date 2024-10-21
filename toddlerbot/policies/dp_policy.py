import os
import time
from collections import deque
from typing import List, Optional

import cv2
import numpy as np
import numpy.typing as npt

from toddlerbot.manipulation.inference_class import DPModel
from toddlerbot.policies.balance_pd import BalancePDPolicy
from toddlerbot.sensing.camera import Camera
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.comm_utils import ZMQNode


class DPPolicy(BalancePDPolicy, policy_name="dp"):
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
        self.control_dt = 0.1

        policy_path = os.path.join(
            "toddlerbot", "policies", "checkpoints", "teleop_model.pth"
        )

        pred_horizon, obs_horizon, action_horizon = 16, 5, 8
        action_dim = len(self.arm_motor_indices) + 2
        lowdim_obs_dim = action_dim

        self.model = DPModel(
            policy_path,
            pred_horizon,
            obs_horizon,
            action_horizon,
            lowdim_obs_dim,
            action_dim,
        )
        self.model.num_diffusion_iters = 10

        # deque for observation
        self.obs_deque: deque = deque([], maxlen=self.model.obs_horizon)
        self.model_action_seq: List[npt.NDArray[np.float32]] = []

    def get_arm_motor_pos(self) -> npt.NDArray[np.float32]:
        if len(self.obs_deque) == self.model.obs_horizon:
            if len(self.model_action_seq) == 0:
                t1 = time.time()
                self.model_action_seq = list(
                    self.model.get_action_from_obs(self.obs_deque)
                )
                t2 = time.time()
                print(f"Model inference time: {t2-t1:.3f}s")

            arm_motor_pos = self.model_action_seq.pop(0)[:14]
        else:
            arm_motor_pos = self.default_motor_pos[self.arm_motor_indices]

        print(f"Arm motor pos: {arm_motor_pos}")
        return arm_motor_pos

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        motor_target = super().step(obs, is_real)

        if self.is_logging and self.camera_frame is not None:
            self.camera_frame = (
                cv2.resize(self.camera_frame, (171, 96))[:96, 38:134] / 255.0
            ).transpose(2, 0, 1)

            obs_entry = {
                "image": self.camera_frame,
                "agent_pos": np.concatenate(
                    [
                        obs.motor_pos[self.arm_motor_indices],
                        np.zeros(2, dtype=np.float32),
                    ]
                ),
            }
            self.obs_deque.append(obs_entry)

        return motor_target
