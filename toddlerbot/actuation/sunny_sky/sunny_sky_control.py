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

import numpy as np
import serial

from toddlerbot.actuation import BaseController, JointState
from toddlerbot.utils.file_utils import ReadLine, find_ports
from toddlerbot.utils.math_utils import interpolate_pos
from toddlerbot.utils.misc_utils import log, precise_sleep, profile


@dataclass
class SunnySkyConfig:
    port: str
    kP: int = 40
    kD: int = 50
    baudrate: int = 115200
    timeout: float = 0.1
    gear_ratio: float = 2.0
    tx_data_prefix: str = ">tx_data:"
    tx_timeout: float = 1.0
    interp_method: str = "cubic"
    default_vel: float = np.pi / 2
    soft_limit: float = 0.02
    soft_kP: int = 4
    soft_kD: int = 5


class SunnySkyController(BaseController):
    def __init__(self, config, joint_range_dict):
        super().__init__(config)

        self.config = config
        self.motor_ids = list(joint_range_dict.keys())
        self.init_pos = {id: 0.0 for id in self.motor_ids}
        self.joint_range_dict = joint_range_dict
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
        self.enable_motor(self.motor_ids)
        self.calibrate_motors()

    def close_motors(self):
        self.disable_motor(self.motor_ids)

    # @profile()
    def send_commands(self, byte_commands):
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

    def enable_motor(self, id):
        """
        Puts motor with CAN ID "id" into torque-control mode. 2nd red LED will turn on
        """
        if not isinstance(id, list):
            id = [id]

        byte_commands = []
        for single_id in id:
            b = bytes([single_id]) + b"\xff\xff\xff\xff\xff\xff\xff\xfc"
            byte_commands.append(b)

        self.send_commands(byte_commands)

    def disable_motor(self, id):
        """
        Removes motor with CAN ID "id" from torque-control mode. 2nd red LED will turn off
        """

        if not isinstance(id, list):
            id = [id]

        byte_commands = []
        for single_id in id:
            b = bytes([single_id]) + b"\xff\xff\xff\xff\xff\xff\xff\xfd"
            byte_commands.append(b)

        self.send_commands(byte_commands)

    def calibrate_motors(self):
        log("Calibrating motors...", header="SunnySky")
        state_dict = self.get_motor_state()
        pos_curr = np.array([state.pos for state in state_dict.values()])
        joint_range = np.array(
            [
                self.joint_range_dict[id][1] - self.joint_range_dict[id][0]
                for id in self.motor_ids
            ]
        )

        log("Testing lower limit for motors...", header="SunnySky")
        self.set_pos(pos_curr - joint_range, limit=False)
        precise_sleep(0.1)

        state_dict = self.get_motor_state()
        zero_pos = np.array([state.pos for state in state_dict.values()])
        log(f"Setting zero position {list(zero_pos)} for motors...", header="SunnySky")
        self.set_pos(zero_pos, limit=False)
        precise_sleep(0.1)

        self.init_pos = {id: pos for id, pos in zip(self.motor_ids, zero_pos)}

    def set_pos(
        self,
        pos,
        interp=True,
        limit=True,
        delta_t=None,
        vel=None,
        kP=None,
        kD=None,
        i_ff=None,
    ):
        def set_pos_helper(pos):
            byte_commands = []
            for id, p in zip(self.motor_ids, pos):
                kP_local = kP
                kD_local = kD
                if limit:
                    lower_limit, upper_limit = sorted(self.joint_range_dict[id])
                    p = np.clip(p, lower_limit, upper_limit)

                    if (
                        p - lower_limit < self.config.soft_limit
                        or upper_limit - p < self.config.soft_limit
                    ):
                        if kP_local is None:
                            kP_local = self.config.soft_kP

                        if kD_local is None:
                            kD_local = self.config.soft_kD

                p_drive = (p + self.init_pos[id]) * self.config.gear_ratio

                b = struct.pack(
                    "<B2f2Hf",
                    id,
                    p_drive,
                    0.0,
                    self.config.kP if kP_local is None else kP_local,
                    self.config.kD if kD_local is None else kD_local,
                    0.0 if i_ff is None else i_ff,
                )
                byte_commands.append(b)

            self.send_commands(byte_commands)

        if interp:
            pos = np.array(pos)
            state_dict = self.get_motor_state()
            pos_start = np.array([state.pos for state in state_dict.values()])

            if vel is None and delta_t is None:
                delta_t = max(np.abs(pos - pos_start) / self.config.default_vel)
            elif delta_t is None:
                delta_t = max(np.abs(pos - pos_start) / vel)

            interpolate_pos(
                set_pos_helper,
                pos_start,
                pos,
                delta_t,
                self.config.interp_method,
                "sunny_sky",
            )
        else:
            set_pos_helper(pos)

    # @profile()
    def get_motor_state(self):
        # log(f"Start... {time.time()}", header="SunnySky", level="warning")

        self.client.reset_input_buffer()

        byte_commands = []
        for id in self.motor_ids:
            b = bytes([id]) + b"get_state"
            byte_commands.append(b)

        self.send_commands(byte_commands)

        state_dict = {}
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

                    id, p, v, t, vb = map(float, single_data_str.split(","))
                    id = int(id)
                    state_dict[id] = JointState(
                        time=time.time(),
                        pos=p / self.config.gear_ratio - self.init_pos[id],
                        vel=v,
                    )

            if len(state_dict) == len(self.motor_ids):
                # log(f"End... {time.time()}", header="SunnySky", level="warning")
                return state_dict

        log(
            f"Failed to retrieve Motor {id}'s state after {self.config.tx_timeout}s.",
            header="SunnySky",
            level="warning",
        )


if __name__ == "__main__":
    joint_range_dict = {1: (0, np.pi / 2), 2: (0, -np.pi / 2)}
    pos_ref_seq = [
        [0.0, 0.0],
        [np.pi / 2, -np.pi / 4],
        [0.0, 0.0],
        [np.pi / 4, -np.pi / 2],
        [0.64, -0.64],
    ]

    # joint_range_dict = {1: (0, np.pi / 2)}
    # pos_ref_seq = [[0.0], [np.pi / 2], [0.0], [np.pi / 4], [0.64]]

    # joint_range_dict = {2: (0, -np.pi / 2)}
    # pos_ref_seq = [[0.0], [-np.pi / 2], [0.0], [-np.pi / 4], [-0.64]]

    config = SunnySkyConfig(port=find_ports("Feather"))
    controller = SunnySkyController(config, joint_range_dict=joint_range_dict)

    time_start = time.time()
    for pos_ref in pos_ref_seq:
        controller.set_pos(pos_ref)
        state_dict = controller.get_motor_state()

        message = "Motor states:"
        for id, state in state_dict.items():
            message += f" {id}: {state.pos:.4f} at {state.time - time_start:.4f}s"

        log(message, header="SunnySky", level="debug")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    controller.close_motors()
