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
    """Policy for resetting the robot to the default position."""

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
        """Initializes an instance of the class with specified parameters.

        Args:
            name (str): The name of the instance.
            robot (Robot): The robot associated with this instance.
            init_motor_pos (npt.NDArray[np.float32]): Initial motor positions.
            joystick (Optional[Joystick]): Joystick for controlling the robot, if any.
            cameras (Optional[List[Camera]]): List of cameras attached to the robot, if any.
            zmq_receiver (Optional[ZMQNode]): ZMQ node for receiving data, if any.
            zmq_sender (Optional[ZMQNode]): ZMQ node for sending data, if any.
            ip (str): IP address for network communication.
            fixed_command (Optional[npt.NDArray[np.float32]]): Fixed command to be used, if any.
            use_torso_pd (bool): Flag to use torso proportional-derivative control. Defaults to True.
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
        self.reset_vel = 0.3
        self.reset_time = None

    def get_arm_motor_pos(self, obs: Obs) -> npt.NDArray[np.float32]:
        """Retrieves the positions of the arm motors from the observation data.

        Args:
            obs (Obs): An observation object containing motor position data.

        Returns:
            npt.NDArray[np.float32]: An array of float32 values representing the positions of the arm motors.
        """
        return obs.motor_pos[self.arm_motor_indices]

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Executes a control step for a robotic system, managing motor positions and reset sequences.

        This function processes the current observation to determine the appropriate motor targets. If a reset is required due to button press and waist motor positions exceeding a threshold, it calculates the necessary reset trajectory to bring the motors to their default positions. Otherwise, it performs a standard control step.

        Args:
            obs (Obs): The current observation containing motor positions and time.
            is_real (bool, optional): Flag indicating if the operation is on a real system. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing control inputs and the target motor positions.
        """
        if self.is_button_pressed and self.reset_time is None:
            if np.any(np.abs(obs.motor_pos[self.waist_motor_indices]) > 0.5):
                # reset waist first
                upright_motor_pos = obs.motor_pos.copy()
                upright_motor_pos[self.waist_motor_indices] = 0.0
                upright_duration = np.max(
                    np.abs((upright_motor_pos - obs.motor_pos) / self.reset_vel)
                )
                reset_time_upright, reset_action_upright = self.move(
                    obs.time - self.control_dt,
                    obs.motor_pos,
                    upright_motor_pos,
                    upright_duration,
                )
                reset_duration = np.max(
                    np.abs(
                        (self.default_motor_pos - upright_motor_pos) / self.reset_vel
                    )
                )
                reset_time_default, reset_action_default = self.move(
                    reset_time_upright[-1],
                    upright_motor_pos,
                    self.default_motor_pos,
                    reset_duration,
                    end_time=0.5,
                )
                self.reset_time = np.concatenate(
                    [reset_time_upright, reset_time_default]
                )
                self.reset_action = np.concatenate(
                    [reset_action_upright, reset_action_default]
                )
            else:
                reset_duration = np.max(
                    np.abs((self.default_motor_pos - obs.motor_pos) / self.reset_vel)
                )
                self.reset_time, self.reset_action = self.move(
                    obs.time - self.control_dt,
                    obs.motor_pos,
                    self.default_motor_pos,
                    reset_duration,
                    end_time=0.5,
                )

        control_inputs, motor_target = super().step(obs, is_real)

        if self.reset_time is not None:
            if obs.time < self.reset_time[-1]:
                motor_target = np.asarray(
                    interpolate_action(obs.time, self.reset_time, self.reset_action)
                )
            else:
                motor_target = self.default_motor_pos
                self.reset_time = None

        return control_inputs, motor_target
