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
    """Policy for executing manipulation tasks using a deep policy model."""

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
        """Initializes the robot control system with specified parameters and loads the appropriate policy model.

        Args:
            name (str): The name of the robot control instance.
            robot (Robot): The robot object to be controlled.
            init_motor_pos (npt.NDArray[np.float32]): Initial positions of the robot's motors.
            ckpt (str): Checkpoint identifier for loading the policy model.
            joystick (Optional[Joystick]): Joystick object for manual control, if available.
            cameras (Optional[List[Camera]]): List of camera objects for visual input, if available.
            zmq_receiver (Optional[ZMQNode]): ZMQ node for receiving data, if applicable.
            zmq_sender (Optional[ZMQNode]): ZMQ node for sending data, if applicable.
            ip (str): IP address for network communication.
            fixed_command (Optional[npt.NDArray[np.float32]]): Predefined command sequence, if any.
            task (str): Task identifier, such as "pick" or "hug", to configure task-specific parameters.

        Raises:
            ValueError: If the motion data file for the specified task is not found.
        """
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
        prep = "kneel" if task == "pick" else "hold"

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
        self.action_dropout = 3

        self.neck_pitch_idx = robot.motor_ordering.index("neck_pitch_act")
        if task == "hug":
            self.neck_pitch_ratio = 1.0
        elif task == "pick":
            self.neck_pitch_ratio = 0.75
        else:
            self.neck_pitch_ratio = 1.0

        self.left_arm_indices = self.arm_motor_indices[:7]
        self.right_arm_indices = self.arm_motor_indices[7:14]
        if self.robot.has_gripper:
            self.left_arm_indices = np.concatenate(
                [self.left_arm_indices, self.arm_motor_indices[-2:-1]]
            )
            self.right_arm_indices = np.concatenate(
                [self.right_arm_indices, self.arm_motor_indices[-1:]]
            )

        if len(task) > 0:
            self.manip_duration = 2.0
            self.idle_duration = 5.0 if task == "hug" else 6.0
            self.reset_duration = 7.0 if task == "hug" else 2.0

            motion_file_path = os.path.join("motion", f"{prep}.pkl")
            if os.path.exists(motion_file_path):
                data_dict = joblib.load(motion_file_path)
            else:
                raise ValueError(f"No data files found in {motion_file_path}")

            self.manip_motor_pos = np.array(data_dict["action_traj"], dtype=np.float32)[
                -1
            ]
            self.manip_motor_pos[self.neck_pitch_idx] *= self.neck_pitch_ratio
            if task == "hug":
                self.manip_motor_pos[self.left_sho_pitch_idx] -= 0.2
                self.manip_motor_pos[self.right_sho_pitch_idx] += 0.2

            if self.robot.has_gripper:
                self.manip_motor_pos = np.concatenate(
                    [self.manip_motor_pos, np.zeros(2, dtype=np.float32)]
                )

        self.capture_frame = True
        self.manip_count = 0
        self.wrap_up_time = None

    def reset(self) -> None:
        """Resets the state of the object to its initial configuration.

        This method clears the observation deque, resets the model action sequence,
        sets the manipulation count to zero, and clears the wrap-up time. It also
        invokes the reset method of the superclass to ensure any inherited state
        is also reset.
        """
        super().reset()

        self.obs_deque = deque([], maxlen=self.model.obs_horizon)
        self.model_action_seq = []
        self.manip_count = 0
        self.wrap_up_time = None

    def get_arm_motor_pos(self, obs: Obs) -> npt.NDArray[np.float32]:
        """Calculates the arm motor positions based on the current observation and task.

        This function determines the appropriate motor positions for the robot's arm by utilizing a sequence of model-generated actions. If the observation deque is full, it retrieves actions from the model and adjusts them based on the task type. For a "pick" task, it combines the model actions with the current manipulator positions. If the robot lacks a gripper, it adjusts the action sequence accordingly. If the observation deque is not full, it defaults to using the current manipulator positions.

        Args:
            obs (Obs): The current observation of the environment.

        Returns:
            npt.NDArray[np.float32]: The calculated arm motor positions.
        """
        if len(self.obs_deque) == self.model.obs_horizon:
            if len(self.model_action_seq) == 0:
                t1 = time.time()
                self.model_action_seq = list(
                    self.model.get_action_from_obs(self.obs_deque)[
                        self.action_dropout :
                    ]
                )
                t2 = time.time()
                print(f"Model inference time: {t2 - t1:.3f}s")

            arm_motor_pos = self.model_action_seq.pop(0)
            if self.task == "pick":
                arm_motor_pos = np.concatenate(
                    [
                        self.manip_motor_pos[self.left_arm_indices[:-1]],
                        arm_motor_pos[:-1],
                        self.manip_motor_pos[self.left_arm_indices[-1:]],
                        arm_motor_pos[-1:],
                    ]
                )
            else:
                if not self.robot.has_gripper:
                    arm_motor_pos = arm_motor_pos[:-2]
        else:
            arm_motor_pos = self.manip_motor_pos[self.arm_motor_indices]

        return arm_motor_pos

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Executes a step in the robot's control loop, updating motor targets and handling task-specific logic.

        Args:
            obs (Obs): The current observation containing sensor data and motor positions.
            is_real (bool, optional): Flag indicating whether the step is being executed in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing control inputs and the target motor positions.
        """
        control_inputs, motor_target = super().step(obs, is_real)

        if (
            obs.time - self.time_start
            >= self.manip_duration
            + (self.manip_count + 1) * self.idle_duration
            + self.manip_count * self.reset_duration
        ):
            self.is_running = False
            if self.wrap_up_time is None:
                if self.task == "hug":
                    twist_motor_pos = obs.motor_pos.copy()
                    waist_motor_pos = self.robot.waist_ik([-0.1, -np.pi / 2])
                    twist_motor_pos[self.waist_motor_indices] = waist_motor_pos
                    twist_motor_pos[self.left_sho_pitch_idx] -= 0.2
                    twist_motor_pos[self.right_sho_pitch_idx] += 0.2
                    release_motor_pos = self.manip_motor_pos.copy()
                    release_motor_pos[self.waist_motor_indices] = waist_motor_pos

                    twist_time, twist_action = self.move(
                        obs.time - self.control_dt,
                        obs.motor_pos,
                        twist_motor_pos,
                        (self.reset_duration - 1) / 3,  # 2
                    )
                    release_time, release_action = self.move(
                        twist_time[-1], twist_motor_pos, release_motor_pos, 1.0
                    )
                    back_motor_pos = self.manip_motor_pos.copy()
                    back_motor_pos[self.left_sho_pitch_idx] = self.default_motor_pos[
                        self.left_sho_pitch_idx
                    ]
                    back_motor_pos[self.right_sho_pitch_idx] = self.default_motor_pos[
                        self.right_sho_pitch_idx
                    ]
                    back_motor_pos[self.left_sho_roll_idx] = -1.4
                    back_motor_pos[self.right_sho_roll_idx] = -1.4
                    back_time, back_action = self.move(
                        release_time[-1],
                        release_motor_pos,
                        back_motor_pos,  # self.manip_motor_pos,
                        (self.reset_duration - 1) / 3,  # 2
                    )
                    default_time, default_action = self.move(
                        back_time[-1],
                        back_motor_pos,
                        self.default_motor_pos,
                        (self.reset_duration - 1) / 3,
                    )

                    self.wrap_up_time = np.concatenate(
                        [twist_time, release_time, back_time, default_time]
                    )
                    self.wrap_up_action = np.concatenate(
                        [twist_action, release_action, back_action, default_action]
                    )
                else:
                    self.wrap_up_time, self.wrap_up_action = self.move(
                        obs.time - self.control_dt,
                        obs.motor_pos,
                        self.manip_motor_pos,
                        self.reset_duration,
                    )

            if self.wrap_up_time is not None and obs.time < self.wrap_up_time[-1]:
                motor_target = np.asarray(
                    interpolate_action(obs.time, self.wrap_up_time, self.wrap_up_action)
                )
            else:
                motor_target = self.wrap_up_action[-1]
                self.manip_count += 1
                self.wrap_up_time = None

        elif obs.time - self.time_start >= self.manip_duration:
            self.is_running = True
            if self.task == "pick":
                motor_target[self.neck_motor_indices] = self.manip_motor_pos[
                    self.neck_motor_indices
                ]
                motor_target[self.left_arm_indices] = self.manip_motor_pos[
                    self.left_arm_indices
                ]
                motor_target[self.waist_motor_indices] = self.manip_motor_pos[
                    self.waist_motor_indices
                ]
                motor_target[self.leg_motor_indices] = self.manip_motor_pos[
                    self.leg_motor_indices
                ]

        if self.is_running:
            if self.camera_frame is not None:
                image = cv2.resize(self.camera_frame, (128, 96))[:96, 16:112] / 255.0  # type: ignore

                # Visualize the cropped frame
                # cv2.imshow("Camera Frame", image)
                # cv2.waitKey(1)  # Needed to update the display window

                image = image.transpose(2, 0, 1)
                if self.task == "pick":
                    agent_pos = obs.motor_pos[self.right_arm_indices]
                else:
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
            self.obs_deque.clear()
            self.model_action_seq = []

        return control_inputs, motor_target
