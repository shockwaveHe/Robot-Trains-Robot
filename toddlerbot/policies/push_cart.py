from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.policies.dp_policy import DPPolicy
from toddlerbot.policies.walk import WalkPolicy
from toddlerbot.sensing.camera import Camera
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.comm_utils import ZMQNode
from toddlerbot.utils.math_utils import interpolate_action


class PushCartPolicy(BasePolicy, policy_name="push_cart"):
    """Policy for pushing a cart."""

    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ckpt: str = "",
        joystick: Optional[Joystick] = None,
        cameras: Optional[List[Camera]] = None,
        zmq_receiver: Optional[ZMQNode] = None,
        zmq_sender: Optional[ZMQNode] = None,
        ip: str = "",
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        """Initializes the class with specified parameters and sets up grasp and walk policies.

        Args:
            name (str): The name of the instance.
            robot (Robot): The robot object to be controlled.
            init_motor_pos (npt.NDArray[np.float32]): Initial motor positions for the robot.
            ckpt (str, optional): Checkpoint for loading policy configurations. Defaults to an empty string.
            joystick (Optional[Joystick], optional): Joystick for manual control. Defaults to None.
            cameras (Optional[List[Camera]], optional): List of camera objects for visual input. Defaults to None.
            zmq_receiver (Optional[ZMQNode], optional): ZMQ node for receiving data. Defaults to None.
            zmq_sender (Optional[ZMQNode], optional): ZMQ node for sending data. Defaults to None.
            ip (str, optional): IP address for network communication. Defaults to an empty string.
            fixed_command (Optional[npt.NDArray[np.float32]], optional): Fixed command array for the robot. Defaults to None.
        """
        super().__init__(name, robot, init_motor_pos)

        self.forward_command = np.array([0, 0, 0, 0, 0, 0.1, 0, 0], dtype=np.float32)

        self.grasp_policy = DPPolicy(
            "grasp",
            robot,
            init_motor_pos,
            ckpt,
            task="grasp",
            joystick=joystick,
            cameras=cameras,
            zmq_receiver=zmq_receiver,
            zmq_sender=zmq_sender,
            ip=ip,
            fixed_command=fixed_command,
        )
        self.walk_policy = WalkPolicy(
            "walk",
            robot,
            init_motor_pos,
            fixed_command=self.forward_command,
            ckpt="PPOConfig.num_timesteps=300000000,PPOConfig.num_evals=3000,PPOConfig.seed=1_20250108_234358",
        )

        self.time_start = self.prep_duration
        self.grasp_duration = 6.0
        self.grasp_pose: npt.NDArray[np.float32] | None

        self.is_prepared = False
        self.is_grasped = False

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Executes a step in the control loop, determining the appropriate motor actions based on the current observation and state.

        Args:
            obs (Obs): The current observation containing sensor data and time information.
            is_real (bool, optional): Flag indicating whether the operation is in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing control inputs as a dictionary and the target motor positions as a NumPy array.
        """
        if not self.is_prepared:
            self.is_prepared = True
            self.prep_duration = 7.0 if is_real else 2.0
            self.prep_time, self.prep_action = self.move(
                -self.control_dt,
                self.init_motor_pos,
                self.default_motor_pos,
                self.prep_duration,
                end_time=5.0 if is_real else 0.0,
            )

        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return {}, action

        if obs.time < self.time_start + self.grasp_duration:
            control_inputs, motor_target = self.grasp_policy.step(obs, is_real)

        else:
            walk_command = self.walk_policy.get_command(self.walk_policy.control_inputs)
            if (
                np.any(walk_command)
                or not self.walk_policy.is_standing
                or not self.is_grasped
            ):
                if not self.is_grasped:
                    self.is_grasped = True
                    self.grasp_pose = obs.motor_pos

                control_inputs, motor_target = self.walk_policy.step(obs, is_real)
            else:
                control_inputs, motor_target = {}, self.default_motor_pos.copy()

            assert self.grasp_pose is not None
            motor_target[self.neck_motor_indices] = self.grasp_pose[
                self.neck_motor_indices
            ]
            motor_target[self.arm_motor_indices] = self.grasp_pose[
                self.arm_motor_indices
            ]

        return control_inputs, motor_target
