"""
Ben Katz
Motor Module Python API
Assumes the serial device is a nucleo running the firmware at:
Corresponding STM32F446 Firmware here:
https://os.mbed.com/users/benkatz/code/CanMaster/
"""

import struct
import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import serial

from toddlerbot.actuation import BaseController, JointState
from toddlerbot.utils.math_utils import interpolate_pos
from toddlerbot.utils.misc_utils import log, profile


@dataclass
class SunnySkyConfig:
    port: str
    kP: List[int]
    kD: List[int]
    i_ff: List[float]
    gear_ratio: List[float]
    joint_limit: List[float]
    init_pos: List[float]
    baudrate: int = 115200
    default_vel: float = np.pi
    interp_method: str = "cubic"
    timeout: float = 0.1
    tx_data_prefix: str = ">tx_data:"
    tx_timeout: float = 1.0
    soft_limit: float = 0.02
    soft_factor: float = 0.1


# https://github.com/pyserial/pyserial/issues/216#issuecomment-369414522
class ReadLine:
    def __init__(self, s: serial.Serial):
        self.buf = bytearray()
        self.s = s

    def readline(self) -> bytearray:
        i = self.buf.find(b"\n")
        if i >= 0:
            r = self.buf[: i + 1]
            self.buf = self.buf[i + 1 :]
            return r
        while True:
            i = max(1, min(2048, self.s.in_waiting))
            data = self.s.read(i)
            i = data.find(b"\n")
            if i >= 0:
                r: bytearray = self.buf + data[: i + 1]
                self.buf[0:] = data[i + 1 :]
                return r
            else:
                self.buf.extend(data)


class SunnySkyController(BaseController):
    def __init__(self, config: SunnySkyConfig, motor_ids: List[int]):
        super().__init__()

        self.config = config
        self.motor_ids: List[int] = motor_ids
        if len(config.init_pos) == 0:
            self.init_pos = {id: 0.0 for id in motor_ids}
        else:
            self.init_pos = {id: pos for id, pos in zip(motor_ids, config.init_pos)}

        if len(config.gear_ratio) == 0:
            self.gear_ratio = {id: 1.0 for id in motor_ids}
        else:
            self.gear_ratio = {id: gr for id, gr in zip(motor_ids, config.gear_ratio)}

        self.lock = Lock()

        self.connect_to_client()
        self.initialize_motors()

    def connect_to_client(self):
        try:
            self.client = serial.Serial(
                self.config.port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout,
            )
            self.reader = ReadLine(self.client)
            log(f"Connected to the port: {self.config.port}", header="SunnySky")

        except Exception:
            raise ConnectionError("Could not connect to any SunnySky port.")

    def initialize_motors(self):
        log("Initializing motors...", header="SunnySky")
        self.enable_motors(self.motor_ids)

    def close_motors(self):
        self.disable_motors(self.motor_ids)

    # @profile()
    def send_commands(self, byte_commands: List[bytes]):
        """
        Sends a single command to a single motor.
        """
        num_motors = len(byte_commands)
        # Prepare the message payload without start/end markers
        message = bytes([num_motors]) + b"".join(byte_commands)
        # Calculate the payload length
        payload_length = len(message)
        # Construct the final message with start marker, payload length, actual payload, and end marker
        final_message = (
            b"<" + payload_length.to_bytes(2, byteorder="little") + message + b">"
        )

        # with self.lock:
        self.client.write(final_message)

    def enable_motors(self, motor_ids: List[int]):
        """
        Puts motor with CAN ID "id" into torque-control mode. 2nd red LED will turn on
        """
        byte_commands: List[bytes] = []
        for id in motor_ids:
            b = bytes([id]) + b"\xff\xff\xff\xff\xff\xff\xff\xfc"
            byte_commands.append(b)

        self.send_commands(byte_commands)

    def disable_motors(self, motor_ids: List[int]):
        """
        Removes motor with CAN ID "id" from torque-control mode. 2nd red LED will turn off
        """
        byte_commands: List[bytes] = []
        for id in motor_ids:
            b = bytes([id]) + b"\xff\xff\xff\xff\xff\xff\xff\xfd"
            byte_commands.append(b)

        self.send_commands(byte_commands)

    def calibrate_motors(self):
        log("Calibrating motors...", header="SunnySky")
        state_dict = self.get_motor_state()
        pos_arr_curr = np.array([state.pos for state in state_dict.values()])

        log("Testing lower limit for motors...", header="SunnySky")
        self.set_pos(
            list(pos_arr_curr - np.array(self.config.joint_limit)), use_limit=False
        )

        time.sleep(0.2)

        state_dict: Dict[int, JointState] = self.get_motor_state()
        init_pos: Dict[int, float] = {}
        for id, state in state_dict.items():
            init_pos[id] = state.pos

        zero_pos = list(init_pos.values())
        log(f"Setting zero position {zero_pos} for motors...", header="SunnySky")
        self.set_pos(zero_pos, use_limit=False)

        time.sleep(0.2)

        self.init_pos = init_pos

        return init_pos

    # @profile()
    def set_pos(
        self,
        pos: List[float],
        interp: bool = True,
        use_limit: bool = True,
        delta_t: float = -1,
        vel: List[float] = [],
        kP: List[int] = [],
        kD: List[int] = [],
        i_ff: List[float] = [],
    ):
        def set_pos_helper(pos_arr: npt.NDArray[np.float32]):
            byte_commands: List[bytes] = []
            for i, id in enumerate(self.motor_ids):
                if len(kP) == 0:
                    kP_local = self.config.kP[i]
                else:
                    kP_local = kP[i]

                if len(kD) == 0:
                    kD_local = self.config.kD[i]
                else:
                    kD_local = kD[i]

                if len(i_ff) == 0:
                    i_ff_local = self.config.i_ff[i]
                else:
                    i_ff_local = i_ff[i]

                if use_limit:
                    # TODO: Remove the hard coded values
                    joint_limits = [0.0, self.config.joint_limit[i]]
                    lower_limit = min(joint_limits)
                    upper_limit = max(joint_limits)
                    p = np.clip(pos_arr[i], lower_limit, upper_limit)

                    if (
                        p - lower_limit < self.config.soft_limit
                        or upper_limit - p < self.config.soft_limit
                    ):
                        kP_local = int(self.config.soft_factor * kP_local)
                        kD_local = int(self.config.soft_factor * kP_local)

                p_drive: float = (self.init_pos[id] + pos_arr[i]) * self.gear_ratio[id]

                b = struct.pack(
                    "<B2f2Hf",
                    id,
                    p_drive,
                    0.0,
                    kP_local,
                    kD_local,
                    i_ff_local,
                )
                byte_commands.append(b)

            self.send_commands(byte_commands)

        pos_arr: npt.NDArray[np.float32] = np.array(pos)

        if interp:
            pos_arr_start: npt.NDArray[np.float32] = np.array(
                [state.pos for state in self.get_motor_state().values()]
            )

            if len(vel) == 0 and delta_t < 0:
                delta_t = max(np.abs(pos_arr - pos_arr_start) / self.config.default_vel)
            elif delta_t < 0:
                delta_t = max(np.abs(pos_arr - pos_arr_start) / np.array(vel))

            interpolate_pos(
                set_pos_helper,
                pos_arr_start,
                pos_arr,
                delta_t,
                self.config.interp_method,
            )
        else:
            set_pos_helper(pos_arr)

    # @profile()
    def get_motor_state(self) -> Dict[int, JointState]:
        # log(f"Start... {time.time()}", header="SunnySky", level="warning")

        self.client.reset_input_buffer()

        byte_commands: List[bytes] = []
        for id in self.motor_ids:
            b = bytes([id]) + b"get_state"
            byte_commands.append(b)

        self.send_commands(byte_commands)

        state_dict: Dict[int, JointState] = {}
        time_start = time.time()
        while (time.time() - time_start) < self.config.tx_timeout:
            # with self.lock:
            line = self.reader.readline()

            if not line:
                continue  # Skip empty lines

            decoded_line = line.decode().strip()
            if "error" in decoded_line.lower():
                log(decoded_line, header="SunnySky", level="warning")
            # else:
            #     log(decoded_line, header="SunnySky", level="debug")

            if decoded_line.startswith(self.config.tx_data_prefix):
                _, data_str = decoded_line.split(self.config.tx_data_prefix, 1)
                for single_data_str in data_str.split(";"):
                    if len(single_data_str) == 0:
                        continue

                    # ID, position, velocity, current, voltage
                    id, p, v, _, _ = map(float, single_data_str.split(","))
                    id = int(id)
                    state_dict[id] = JointState(
                        time=time.time(),
                        pos=p / self.gear_ratio[id] - self.init_pos[id],
                        vel=v / self.gear_ratio[id],
                    )

            if len(state_dict) == len(self.motor_ids):
                # log(f"End... {time.time()}", header="SunnySky", level="warning")
                return state_dict

        for id in self.motor_ids:
            if id not in state_dict:
                log(
                    f"Failed to retrieve Motor {id}'s state after {self.config.tx_timeout}s.",
                    header="SunnySky",
                    level="warning",
                )

        return state_dict
