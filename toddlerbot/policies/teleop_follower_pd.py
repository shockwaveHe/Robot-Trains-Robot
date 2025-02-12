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
    """Teleoperation follower policy for the follower robot of ToddlerBot."""

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
        """Initializes the object with specified parameters and sets up task-specific configurations.

        Args:
            name (str): The name of the object.
            robot (Robot): The robot instance associated with the object.
            init_motor_pos (npt.NDArray[np.float32]): Initial motor positions for the robot.
            joystick (Optional[Joystick]): Joystick instance for controlling the robot, if any.
            cameras (Optional[List[Camera]]): List of camera instances for capturing images, if any.
            zmq_receiver (Optional[ZMQNode]): ZMQ node for receiving data, if any.
            zmq_sender (Optional[ZMQNode]): ZMQ node for sending data, if any.
            ip (str): IP address for network communication.
            fixed_command (Optional[npt.NDArray[np.float32]]): Fixed command array for the robot, if any.
            use_torso_pd (bool): Flag to use proportional-derivative control for the torso.
            task (str): The task to be performed, affecting motor position configurations.

        Raises:
            ValueError: If the motion file for the specified task does not exist.
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
            use_torso_pd,
        )

        self.task = task
        prep = "kneel" if task == "pick" else "hold"

        self.neck_pitch_idx = robot.motor_ordering.index("neck_pitch_act")
        if task == "hug":
            self.neck_pitch_ratio = 1.0
        elif task == "pick":
            self.neck_pitch_ratio = 0.75
        else:
            self.neck_pitch_ratio = 1.0

        # self.neck_pitch_ratio = 0.0
        self.num_arm_motors = 7

        if len(task) > 0:
            self.manip_duration = 2.0

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

            if robot.has_gripper:
                self.manip_motor_pos = np.concatenate(
                    [self.manip_motor_pos, np.zeros(2, dtype=np.float32)]
                )

        self.capture_frame = True

        self.dataset_logger = DatasetLogger()

    def get_arm_motor_pos(self, obs: Obs) -> npt.NDArray[np.float32]:
        """Retrieves the current positions of the arm motors.

        If the arm motor positions have been explicitly set, returns those values.
        Otherwise, extracts and returns the positions from the manipulation motor
        positions using predefined arm motor indices.

        Args:
            obs (Obs): The observation object containing relevant data.

        Returns:
            npt.NDArray[np.float32]: An array of arm motor positions.
        """
        if self.arm_motor_pos is None:
            return self.manip_motor_pos[self.arm_motor_indices]
        else:
            return self.arm_motor_pos

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Executes a step in the control loop, adjusting motor targets based on the task and logging data.

        Args:
            obs (Obs): The current observation containing time and motor positions.
            is_real (bool, optional): Flag indicating if the step is in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing control inputs and the updated motor target positions.
        """
        control_inputs, motor_target = super().step(obs, is_real)

        if self.task == "pick" and obs.time - self.prep_duration >= self.manip_duration:
            motor_target[self.neck_motor_indices] = self.manip_motor_pos[
                self.neck_motor_indices
            ]
            motor_target[self.arm_motor_indices[: self.num_arm_motors]] = (
                self.manip_motor_pos[self.arm_motor_indices[: self.num_arm_motors]]
            )
            motor_target[self.waist_motor_indices] = self.manip_motor_pos[
                self.waist_motor_indices
            ]
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
                Data(obs.time, motor_target, obs.motor_pos, self.camera_frame)
            )

        return control_inputs, motor_target
