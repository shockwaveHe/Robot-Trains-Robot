import os
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import numpy.typing as npt

from toddlerbot.manipulation.inference_class import DPModel
from toddlerbot.policies.balance_pd import BalancePDPolicy
from toddlerbot.sensing.camera import Camera
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.comm_utils import ZMQNode
from toddlerbot.utils.math_utils import interpolate_action


class DPPolicy(BalancePDPolicy, policy_name="dp"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ckpt: str,
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
        self.control_dt = 0.1

        policy_path = os.path.join(
            "toddlerbot", "policies", "checkpoints", f"{robot.name}_{ckpt}_dp.pth"
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
        # self.model.num_diffusion_iters = 10

        # deque for observation
        self.obs_deque: deque = deque([], maxlen=self.model.obs_horizon)
        self.model_action_seq: List[npt.NDArray[np.float32]] = []

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

        self.reset_duration = 5.0
        self.reset_end_time = 1.0
        self.reset_time = None

    def get_arm_motor_pos(self, obs: Obs) -> npt.NDArray[np.float32]:
        arm_motor_pos = self.prep_motor_pos[self.arm_motor_indices]
        if len(self.obs_deque) == self.model.obs_horizon:
            if len(self.model_action_seq) == 0:
                t1 = time.time()
                self.model_action_seq = list(
                    self.model.get_action_from_obs(self.obs_deque)
                )
                t2 = time.time()
                print(f"Model inference time: {t2-t1:.3f}s")

            arm_motor_pos = self.model_action_seq.pop(0)[:14]

        elif not self.is_running:
            if self.is_button_pressed and self.reset_time is None:
                self.reset_time, self.reset_action = self.move(
                    obs.time - self.control_dt,
                    obs.motor_pos[self.arm_motor_indices],
                    self.prep_motor_pos[self.arm_motor_indices],
                    self.reset_duration,
                    end_time=self.reset_end_time,
                )

            if self.reset_time is not None and obs.time < self.reset_time[-1]:
                arm_motor_pos = np.asarray(
                    interpolate_action(obs.time, self.reset_time, self.reset_action)
                )
        else:
            # If the user presses the button while resetting,
            # the reset action with be overridden by the default action.
            self.reset_time = None

        return arm_motor_pos

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        control_inputs, motor_target = super().step(obs, is_real)

        if obs.time >= self.prep_duration:
            self.is_running = True

        if self.is_running:
            if self.camera_frame is not None:
                self.camera_frame = (
                    cv2.resize(self.camera_frame, (128, 96))[:96, 16:112] / 255.0
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
            else:
                raise ValueError("Camera frame is needed for DP policy.")
        else:
            self.obs_deque.clear()
            self.model_action_seq = []

        return control_inputs, motor_target
