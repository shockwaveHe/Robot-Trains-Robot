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
        )

        self.zmq_sender = None

        self.task = task
        prep = "hold" if task == "hug" else "kneel"

        if len(ckpt) > 0:
            run_name = f"{self.robot.name}_{task}_dp_{ckpt}"
            policy_path = os.path.join("results", run_name, "best_ckpt.pth")
            if not os.path.exists(policy_path):
                policy_path = os.path.join("results", run_name, "last_ckpt.pth")
        else:
            policy_path = os.path.join(
                "toddlerbot",
                "policies",
                "checkpoints",
                f"{self.robot.name}_{task}_dp.pth",
            )

        self.model = DPModel(policy_path)
        print(f"Loading policy from {policy_path}")

        # deque for observation
        self.obs_deque: deque = deque([], maxlen=self.model.obs_horizon)
        self.model_action_seq: List[npt.NDArray[np.float32]] = []

        self.neck_pitch_idx = robot.motor_ordering.index("neck_pitch_act")
        self.neck_pitch_ratio = 1.0  # 0.25  # Stitch
        self.num_arm_motors = 7

        if len(task) > 0:
            self.manip_duration = 2.0
            self.idle_duration = 6.0

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

            if self.robot.has_gripper:
                self.manip_motor_pos = np.concatenate(
                    [self.manip_motor_pos, np.zeros(2, dtype=np.float32)]
                )

        self.capture_frame = True
        self.last_arm_motor_pos = None

        self.manip_count = 0
        self.wrap_up_time = None

        self.reset_duration = 5.0
        self.reset_end_time = 1.0
        self.reset_time = None

    def get_arm_motor_pos(self, obs: Obs) -> npt.NDArray[np.float32]:
        arm_motor_pos = self.manip_motor_pos[self.arm_motor_indices]
        if len(self.obs_deque) == self.model.obs_horizon:
            if len(self.model_action_seq) == 0:
                t1 = time.time()
                self.model_action_seq = list(
                    self.model.get_action_from_obs(self.obs_deque)
                )
                t2 = time.time()
                print(f"Model inference time: {t2-t1:.3f}s")

            if self.robot.has_gripper:
                arm_motor_pos = self.model_action_seq.pop(0)
            else:
                arm_motor_pos = self.model_action_seq.pop(0)[:-2]

        return arm_motor_pos

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        control_inputs, motor_target = super().step(obs, is_real)

        if (
            obs.time - self.prep_duration
            >= self.manip_duration
            + (self.manip_count + 1) * self.idle_duration
            + self.manip_count * 6.0
        ):
            self.is_running = False
            if self.wrap_up_time is None:
                if self.task == "hug":
                    twist_motor_pos = obs.motor_pos.copy()
                    waist_motor_pos = self.robot.waist_ik([0.0, -np.pi / 2])
                    twist_motor_pos[self.waist_motor_indices] = waist_motor_pos
                    release_motor_pos = self.manip_motor_pos.copy()
                    release_motor_pos[self.waist_motor_indices] = waist_motor_pos

                    twist_time, twist_action = self.move(
                        obs.time - self.control_dt, obs.motor_pos, twist_motor_pos, 2.0
                    )
                    release_time, release_action = self.move(
                        twist_time[-1], twist_motor_pos, release_motor_pos, 2.0
                    )
                    back_time, back_action = self.move(
                        release_time[-1], release_motor_pos, self.manip_motor_pos, 2.0
                    )

                    self.wrap_up_time = np.concatenate(
                        [twist_time, release_time, back_time]
                    )
                    self.wrap_up_action = np.concatenate(
                        [twist_action, release_action, back_action]
                    )
                else:
                    self.wrap_up_time, self.wrap_up_action = self.move(
                        obs.time - self.control_dt,
                        obs.motor_pos[self.arm_motor_indices],
                        self.manip_motor_pos[self.arm_motor_indices],
                        self.reset_duration,
                        end_time=self.reset_end_time,
                    )

            if self.wrap_up_time is not None and obs.time < self.wrap_up_time[-1]:
                motor_target = np.asarray(
                    interpolate_action(obs.time, self.wrap_up_time, self.wrap_up_action)
                )
            else:
                self.manip_count += 1
                self.wrap_up_time = None

        elif obs.time - self.prep_duration >= self.manip_duration:
            self.is_running = True
            if self.task == "pick":
                motor_target[self.neck_motor_indices] = self.manip_motor_pos[self.waist_motor_indices]
                motor_target[self.arm_motor_indices[: self.num_arm_motors]] = (
                    self.manip_motor_pos[self.arm_motor_indices[: self.num_arm_motors]]
                )
                motor_target[self.waist_motor_indices] = self.manip_motor_pos[self.waist_motor_indices]
                motor_target[self.leg_motor_indices] = self.manip_motor_pos[
                    self.leg_motor_indices
                ]

        if self.is_running:
            if self.camera_frame is not None:
                image = cv2.resize(self.camera_frame, (128, 96))[:96, 16:112] / 255.0

                # Visualize the cropped frame
                # cv2.imshow("Camera Frame", image)
                # cv2.waitKey(1)  # Needed to update the display window

                image = image.transpose(2, 0, 1)
                agent_pos = obs.motor_pos[self.arm_motor_indices]
                if not self.robot.has_gripper:
                    agent_pos = np.concatenate(
                        [agent_pos, np.zeros(2, dtype=np.float32)]
                    )

                obs_entry = {
                    "image": image,
                    "agent_pos": agent_pos,
                }
                self.obs_deque.append(obs_entry)
            else:
                raise ValueError("Camera frame is needed for DP policy.")
        else:
            self.obs_deque.clear()
            self.model_action_seq = []

        return control_inputs, motor_target
