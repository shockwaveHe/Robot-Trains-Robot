import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict, List

import numpy as np
import numpy.typing as npt

from toddlerbot.actuation import BaseController, JointState
from toddlerbot.actuation.dynamixel_client import DynamixelClient

# from toddlerbot.utils.math_utils import interpolate_pos
from toddlerbot.utils.misc_utils import log  # profile

CONTROL_MODE_DICT: Dict[str, int] = {
    "current": 0,
    "velocity": 1,
    "position": 3,
    "extended_position": 4,
    "current_based_position": 5,
    "pwm": 16,
}


def get_env_path():
    """Determines the path of the current Python environment.

    Returns:
        str: The path to the current Python environment. If a virtual environment is active, returns the virtual environment's path. If a conda environment is active, returns the conda environment's path. Otherwise, returns the system environment's path.
    """
    # Check if using a virtual environment
    if hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix:
        return sys.prefix
    # If using conda, the CONDA_PREFIX environment variable is set
    elif "CONDA_PREFIX" in os.environ:
        return os.environ["CONDA_PREFIX"]
    else:
        # If not using virtualenv or conda, assume system environment
        return sys.prefix


def set_latency_timer(latency_value: int = 1):
    """Sets the LATENCY_TIMER variable in the port_handler.py file to the specified value.

    This function locates the port_handler.py file within the dynamixel_sdk package and updates the LATENCY_TIMER variable to the given `latency_value`. If the file or the variable is not found, an error message is printed.

    Args:
        latency_value (int): The value to set for LATENCY_TIMER. Defaults to 1.
    """
    env_path = get_env_path()

    # Construct the path to port_handler.py
    port_handler_path = os.path.join(
        env_path,
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
        "dynamixel_sdk",
        "port_handler.py",
    )

    if not os.path.exists(port_handler_path):
        print(f"Error: port_handler.py not found at {port_handler_path}")
        return

    try:
        # Read the content of port_handler.py
        with open(port_handler_path, "r") as file:
            lines = file.readlines()

        # Search for the LATENCY_TIMER line and modify it
        modified = False
        for i, line in enumerate(lines):
            if "LATENCY_TIMER" in line:
                lines[i] = f"LATENCY_TIMER = {latency_value}\n"
                modified = True
                break

        if modified:
            # Write the modified content back to port_handler.py
            with open(port_handler_path, "w") as file:
                file.writelines(lines)

            print(f"LATENCY_TIMER set to 1 in {port_handler_path}")
        else:
            print("LATENCY_TIMER variable not found in port_handler.py")

    except Exception as e:
        print(f"Error while modifying the file: {e}")


@dataclass
class DynamixelConfig:
    """Data class for storing Dynamixel configuration parameters."""

    port: str
    baudrate: int
    control_mode: List[str]
    kP: List[float]
    kI: List[float]
    kD: List[float]
    kFF2: List[float]
    kFF1: List[float]
    init_pos: List[float]
    default_vel: float = np.pi
    interp_method: str = "cubic"
    return_delay_time: int = 1


class DynamixelController(BaseController):
    """Class for controlling Dynamixel motors."""

    def __init__(self, config: DynamixelConfig, motor_ids: List[int]):
        """Initializes the motor controller with the given configuration and motor IDs.

        Args:
            config (DynamixelConfig): The configuration settings for the Dynamixel motors.
            motor_ids (List[int]): A list of motor IDs to be controlled.

        Attributes:
            config (DynamixelConfig): Stores the configuration settings.
            motor_ids (List[int]): Stores the list of motor IDs.
            lock (Lock): A threading lock to ensure thread-safe operations.
            init_pos (np.ndarray): An array of initial positions for the motors, initialized to zeros if not provided in the config.
        """
        self.config = config
        self.motor_ids: List[int] = motor_ids
        self.lock = Lock()

        self.connect_to_client()
        self.initialize_motors()

        if len(self.config.init_pos) == 0:
            self.init_pos = np.zeros(len(motor_ids), dtype=np.float32)
        else:
            self.init_pos = np.array(config.init_pos, dtype=np.float32)
            self.update_init_pos()

    def connect_to_client(self, latency_value: int = 1):
        """Connects to a Dynamixel client and sets the USB latency timer.

        This method sets the USB latency timer for the specified port and attempts to connect to a Dynamixel client. The latency timer is set differently based on the operating system. If the connection fails, an error is logged or raised.

        Args:
            latency_value (int): The desired latency timer value. Defaults to 1.

        Raises:
            ConnectionError: If the connection to the Dynamixel port fails.
        """
        os_type = platform.system()
        try:
            set_latency_timer(latency_value)

            if os_type == "Linux":
                # Construct the command to set the latency timer on Linux
                command = f"echo {latency_value} | sudo tee /sys/bus/usb-serial/devices/{self.config.port.split('/')[-1]}/latency_timer"
            elif os_type == "Darwin":
                command = f"./toddlerbot/actuation/latency_timer_setter_macOS/set_latency_timer -l {latency_value}"
            else:
                raise Exception()

            # Run the command
            result = subprocess.run(
                command, shell=True, text=True, check=True, stdout=subprocess.PIPE
            )
            log(f"Latency Timer set: {result.stdout.strip()}", header="Dynamixel")

        except Exception as e:
            if os_type == "Windows":
                log(
                    "Make sure you're set the latency in the device manager!",
                    header="Dynamixel",
                    level="warning",
                )
            else:
                log(
                    f"Failed to set latency timer: {e}",
                    header="Dynamixel",
                    level="error",
                )

        time.sleep(0.1)

        try:
            self.client = DynamixelClient(
                self.motor_ids, self.config.port, self.config.baudrate
            )
            self.client.connect()
            log(f"Connected to the port: {self.config.port}", header="Dynamixel")

        except Exception:
            raise ConnectionError("Could not connect to the Dynamixel port.")

    def initialize_motors(self):
        """Initialize the motors by rebooting, checking voltage, and configuring settings.

        This method performs the following steps:
        1. Reboots the motors.
        2. Checks the input voltage to ensure it is above a safe threshold.
        3. Configures various motor settings such as return delay time, control mode, and PID gains.
        4. Enables torque on the motors.

        Raises:
            ValueError: If the input voltage is below 10V, indicating a potential power supply issue.
        """
        log("Initializing motors...", header="Dynamixel")
        self.client.reboot(self.motor_ids)
        time.sleep(0.2)

        _, v_in = self.client.read_vin()
        log(f"Voltage (V): {v_in}", header="Dynamixel")
        if np.any(v_in < 10):
            raise ValueError(
                "Voltage too low. Please check the power supply or charge the batteries."
            )

        time.sleep(0.2)

        # This sync writing section has to go after the voltage reading to make sure the motors are powered up
        # Set the return delay time to 1*2=2us
        self.client.sync_write(
            self.motor_ids, [self.config.return_delay_time] * len(self.motor_ids), 9, 1
        )
        self.client.sync_write(
            self.motor_ids,
            [CONTROL_MODE_DICT[m] for m in self.config.control_mode],
            11,
            1,
        )
        self.client.sync_write(self.motor_ids, self.config.kD, 80, 2)
        self.client.sync_write(self.motor_ids, self.config.kI, 82, 2)
        self.client.sync_write(self.motor_ids, self.config.kP, 84, 2)
        self.client.sync_write(self.motor_ids, self.config.kFF2, 88, 2)
        self.client.sync_write(self.motor_ids, self.config.kFF1, 90, 2)
        # self.client.sync_write(self.motor_ids, self.config.current_limit, 102, 2)

        self.client.set_torque_enabled(self.motor_ids, True)

        time.sleep(0.2)

    def update_init_pos(self):
        """Update the initial position to account for any changes in position.

        This method reads the current position from the client and calculates the
        difference from the stored initial position. It then adjusts the initial
        position to reflect any changes, ensuring that the position remains within
        the range of [-π, π].
        """
        _, pos_arr = self.client.read_pos(retries=-1)
        delta_pos = pos_arr - self.init_pos
        delta_pos = (delta_pos + np.pi) % (2 * np.pi) - np.pi
        self.init_pos = pos_arr - delta_pos

    def close_motors(self):
        """Closes all active Dynamixel motor clients.

        This method iterates over all currently open Dynamixel clients and forces them to close if they are in use. It logs a message for each client that is being forcibly closed and then sets the client's port handler to not in use before disconnecting the client.
        """
        open_clients: List[DynamixelClient] = list(DynamixelClient.OPEN_CLIENTS)  # type: ignore
        for open_client in open_clients:
            if open_client.port_handler.is_using:
                log("Forcing client to close.", header="Dynamixel")
            open_client.port_handler.is_using = False
            open_client.disconnect()

    # Only disable the torque, but stay connected through comm. If no id is provided, disable all motors
    def disable_motors(self, ids=None):
        """Disables the torque for specified motors or all motors if no IDs are provided.

        Args:
            ids (list, optional): A list of motor IDs to disable. If None, all motors will be disabled.
        """
        open_clients: List[DynamixelClient] = list(DynamixelClient.OPEN_CLIENTS)  # type: ignore
        for open_client in open_clients:
            if ids is not None:
                # get the intersecting list between ids and motor_ids
                ids_to_disable = list(set(open_client.motor_ids) & set(ids))
                print(f"\nDisabling motor id {ids_to_disable}\n")
                open_client.set_torque_enabled(ids_to_disable, False, retries=0)
            else:
                print("\nDisabling all the motors\n")
                open_client.set_torque_enabled(open_client.motor_ids, False, retries=0)

    def enable_motors(self, ids=None):
        """Enables torque for specified motors or all motors if no IDs are provided.

        Args:
            ids (list, optional): A list of motor IDs to enable. If None, all motors will be enabled.
        """
        open_clients: List[DynamixelClient] = list(DynamixelClient.OPEN_CLIENTS)  # type: ignore
        for open_client in open_clients:
            if ids is not None:
                # get the intersecting list between ids and motor_ids
                ids_to_enable = list(set(open_client.motor_ids) & set(ids))
                print(f"\nEnabling motor id {ids_to_enable}\n")
                open_client.set_torque_enabled(ids_to_enable, True)
            else:
                print("\nEnabling all the motors\n")
                open_client.set_torque_enabled(open_client.motor_ids, True)

    def set_kp(self, kp: List[float]):
        """Set the proportional gain (Kp) for the motors.

        This method updates the proportional gain values for the specified motors by writing to their control table.

        Args:
            kp (List[float]): A list of proportional gain values to be set for the motors.
        """
        with self.lock:
            self.client.sync_write(self.motor_ids, kp, 84, 2)

    def set_kp_kd(self, kp: float, kd: float):
        """Set the proportional (kp) and derivative (kd) gains for the motor.

        This function updates the motor's control parameters by writing the specified
        proportional and derivative gains to the motor's registers. The operation is
        thread-safe.

        Args:
            kp (float): The proportional gain to be set for the motor.
            kd (float): The derivative gain to be set for the motor.
        """
        log(f"Setting motor kp={kp} kd={kd}", header="Dynamixel")
        with self.lock:
            self.client.sync_write(self.motor_ids, [kd] * len(self.motor_ids), 80, 2)
            self.client.sync_write(self.motor_ids, [kp] * len(self.motor_ids), 84, 2)

    def set_parameters(self, kp=None, kd=None, ki=None, kff1=None, kff2=None, ids=None):
        """Sets the motor control parameters for specified Dynamixel motors.

        This function updates the proportional (kp), derivative (kd), integral (ki),
        and feedforward (kff1, kff2) gains for the motors identified by the given IDs.

        Args:
            kp (int, optional): Proportional gain. If None, the parameter is not updated.
            kd (int, optional): Derivative gain. If None, the parameter is not updated.
            ki (int, optional): Integral gain. If None, the parameter is not updated.
            kff1 (int, optional): First feedforward gain. If None, the parameter is not updated.
            kff2 (int, optional): Second feedforward gain. If None, the parameter is not updated.
            ids (list of int, optional): List of motor IDs to update. If None, no motors are updated.
        """
        log("Setting motor parameters", header="Dynamixel")
        with self.lock:
            if kp is not None:
                self.client.sync_write(ids, [kp], 84, 2)
            if kd is not None:
                self.client.sync_write(ids, [kd], 80, 2)
            if ki is not None:
                self.client.sync_write(ids, [ki], 82, 2)
            if kff1 is not None:
                self.client.sync_write(ids, [kff1], 90, 2)
            if kff2 is not None:
                self.client.sync_write(ids, [kff2], 88, 2)

    # @profile()
    def set_pos(self, pos: List[float]):
        """Sets the position of the motors by updating the desired position.

        Args:
            pos (List[float]): A list of position values to set for the motors.
        """
        pos_arr: npt.NDArray[np.float32] = np.array(pos)
        pos_arr_drive = self.init_pos + pos_arr
        with self.lock:
            self.client.write_desired_pos(self.motor_ids, pos_arr_drive)

    def set_cur(self, cur: List[float]):
        """Sets the desired current for the motors.

        This method writes the specified current values to the motors identified by `motor_ids`. The operation is thread-safe, ensuring that the current values are set without interference from other threads.

        Args:
            cur (List[float]): A list of current values to be set for the motors.
        """
        with self.lock:
            self.client.write_desired_cur(self.motor_ids, np.array(cur))

    # @profile()
    def get_motor_state(self, retries: int = 0) -> Dict[int, JointState]:
        """Retrieves the current state of the motors, including position, velocity, and current.

        Args:
            retries (int): The number of retry attempts for reading motor data in case of failure. Defaults to 0.

        Returns:
            Dict[int, JointState]: A dictionary mapping motor IDs to their respective `JointState`, which includes time, position, velocity, and torque.
        """

        # log(f"Start... {time.time()}", header="Dynamixel", level="warning")
        state_dict: Dict[int, JointState] = {}
        with self.lock:
            # time, pos_arr = self.client.read_pos(retries=retries)
            # time, pos_arr, vel_arr = self.client.read_pos_vel(retries=retries)
            time, pos_arr, vel_arr, cur_arr = self.client.read_pos_vel_cur(
                retries=retries
            )

        # log(f"Pos: {np.round(pos_arr, 4)}", header="Dynamixel", level="debug")
        # log(f"Vel: {np.round(vel_arr, 4)}", header="Dynamixel", level="debug")
        # log(f"Cur: {np.round(cur_arr, 4)}", header="Dynamixel", level="debug")

        pos_arr -= self.init_pos

        for i, motor_id in enumerate(self.motor_ids):
            state_dict[motor_id] = JointState(
                time=time, pos=pos_arr[i], vel=vel_arr[i], tor=cur_arr[i]
            )

        # log(f"End... {time.time()}", header="Dynamixel", level="warning")

        return state_dict
