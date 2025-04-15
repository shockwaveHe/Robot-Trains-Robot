from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.locomotion.mjx_config import get_env_config
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick


class WalkPolicy(MJXPolicy, policy_name="walk"):
    """Walking policy for the toddlerbot robot."""

    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ckpt: str = "",
        joystick: Optional[Joystick] = None,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
        exp_folder: Optional[str] = "",
        use_torch: bool = False,
    ):
        """Initializes the class with configuration for a walking environment.

        Args:
            name (str): The name of the environment or instance.
            robot (Robot): The robot instance to be controlled.
            init_motor_pos (npt.NDArray[np.float32]): Initial positions of the robot's motors.
            ckpt (str, optional): Path to a checkpoint file for loading pre-trained models. Defaults to an empty string.
            joystick (Optional[Joystick], optional): Joystick instance for manual control. Defaults to None.
            fixed_command (Optional[npt.NDArray[np.float32]], optional): Predefined command array for the robot. Defaults to None.
            use_torch (bool, optional): Flag indicating whether to use PyTorch for neural network models. Defaults to False.
        """
        env_cfg = get_env_config("walk")
        self.cycle_time = env_cfg.action.cycle_time
        self.command_discount_factor = np.array([0.5, 1.0, 0.75], dtype=np.float32)

        super().__init__(
            name,
            robot,
            init_motor_pos,
            ckpt,
            joystick,
            fixed_command,
            env_cfg,
            use_torch,
            exp_folder,
        )

    def get_phase_signal(self, time_curr: float):
        """Calculate the phase signal as a 2D vector for a given time.

        Args:
            time_curr (float): The current time for which to calculate the phase signal.

        Returns:
            np.ndarray: A 2D vector containing the sine and cosine components of the phase signal, with dtype np.float32.
        """
        phase_signal = np.array(
            [
                np.sin(2 * np.pi * time_curr / self.cycle_time),
                np.cos(2 * np.pi * time_curr / self.cycle_time),
            ],
            dtype=np.float32,
        )
        return phase_signal

    def get_command(self, control_inputs: Dict[str, float]) -> npt.NDArray[np.float32]:
        """Generates a command array based on control inputs for walking.

        Args:
            control_inputs (Dict[str, float]): A dictionary containing control inputs with keys 'walk_x', 'walk_y', and 'walk_turn'.

        Returns:
            npt.NDArray[np.float32]: A numpy array representing the command, with the first five elements as zeros and the remaining elements scaled by the command discount factor.
        """
        command = np.zeros(self.num_commands, dtype=np.float32)
        command[5:] = self.command_discount_factor * np.array(
            [
                control_inputs["walk_x"],
                control_inputs["walk_y"],
                control_inputs["walk_turn"],
            ]
        )

        # print(f"walk_command: {command}")
        return command

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Executes a control step based on the observed state and updates the standing status.

        Args:
            obs (Obs): The current observation of the system state.
            is_real (bool, optional): Flag indicating whether the step is being executed in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing the control inputs as a dictionary and the motor target as a NumPy array.
        """
        control_inputs, motor_target, _ = super().step(obs, is_real)

        if len(self.command_list) >= int(1 / self.control_dt):
            last_commands = self.command_list[-int(1 / self.control_dt) :]
            all_zeros = all(np.all(command == 0) for command in last_commands)
            self.is_standing = all_zeros and abs(self.phase_signal[1]) > 1 - 1e-6
        else:
            self.is_standing = False

        return control_inputs, motor_target, obs
