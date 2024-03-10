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
from typing import Dict, Tuple

import numpy as np
import serial

from toddlerbot.actuation import BaseController


@dataclass
class SunnySkyConfig:
    port: str
    kP: int = 80
    kD: int = 50
    baudrate: int = 115200
    tx_data_prefix: str = ">tx_data:"
    tx_timeout: float = 1.0
    joint_limits: Tuple = (-1.57, 1.57)
    gear_ratio: float = 2.0
    p_tol: float = 0.01


@dataclass
class SunnySkyCommand:
    id: int = 1
    p_des: float = 0
    v_des: float = 0
    kp: int = 40
    kd: int = 8
    i_ff: float = 0


class SunnySkyController(BaseController):
    def __init__(self, config, motor_ids):
        super().__init__(config)

        self.config = config
        self.motor_ids = motor_ids

        self.client = self.connect_to_client()
        self.initialize_motors()

    def connect_to_client(self):
        try:
            client = serial.Serial(
                self.config.port, baudrate=self.config.baudrate, timeout=0.05
            )
            print(f"SunnySky: Connected to the port: {self.config.port}")
            return client
        except Exception as e:
            raise ConnectionError("Could not connect to any SunnySky port.")

    def initialize_motors(self):
        print("SunnySky: Initializing motors...")
        for id in self.motor_ids:
            self.enable_motor(id)

    def close_motors(self):
        for id in self.motor_ids:
            self.disable_motor(id)

    def write_to_serial_with_markers(self, data):
        self.client.write(b"<" + data + b">")

    def send_command(self, command: SunnySkyCommand):
        """
        send_command(desired position, desired velocity, position gain, velocity gain, feed-forward current)

        Sends data over CAN, reads response, and populates rx_data with the response.
        """
        b = struct.pack(
            "<B2f2Hf",
            command.id,
            command.p_des,
            command.v_des,
            command.kp,
            command.kd,
            command.i_ff,
        )
        self.write_to_serial_with_markers(b)

    def enable_motor(self, id):
        """
        Puts motor with CAN ID "id" into torque-control mode. 2nd red LED will turn on
        """
        # print(f"Enabling motor with ID {id}...")
        b = b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFC"
        b = b + bytes(bytearray([id]))
        self.write_to_serial_with_markers(b)
        time.sleep(0.1)
        # self.ser.flushInput()

    def disable_motor(self, id):
        """
        Removes motor with CAN ID "id" from torque-control mode. 2nd red LED will turn off
        """
        # print(f"Disabling motor with ID {id}...")
        b = b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFD"
        b = b + bytes(bytearray([id]))
        self.write_to_serial_with_markers(b)
        time.sleep(0.1)

    def zero_motor(self, id):
        """
        Zero Position Sensor. Sets the mechanical position to zero.
        """
        # print(f"Zeroing motor with ID {id}...")
        b = b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFE"
        b = b + bytes(bytearray([id]))
        self.write_to_serial_with_markers(b)
        time.sleep(0.1)

    def set_pos(self, pos):
        """
        Set the position of the motor
        """
        pos = self.config.gear_ratio * np.clip(
            pos, self.config.joint_limits[0], self.config.joint_limits[1]
        )

        for id, p_des in zip(self.motor_ids, pos):
            p_error = float("inf")
            while p_error > self.config.p_tol:
                # start_time = time.time()

                # Create command with dynamic kd and fixed kp, i_ff
                cmd = SunnySkyCommand(
                    id=id,
                    p_des=p_des,
                    v_des=0.0,
                    kp=self.config.kP,
                    kd=self.config.kD,
                    i_ff=0.0,
                )
                self.send_command(cmd)
                state = self.read_state()
                p_error = abs(pos - state[id][0])

                # print(
                #     f"Command Output: ID={id}, Position={state[id][0]}, Velocity={state[id][1]}, Current={state[id][2]}, Voltage={state[id][3]}"
                # )
                # print(f"Position Error: {p_error}")
                # print(f"Control Frequency: {1 / (time.time() - start_time):.2f} Hz")

    def read_state(self):
        self.client.reset_input_buffer()

        state = {}
        start_time = time.time()
        while time.time() - start_time < self.config.tx_timeout:
            line = self.client.readline()
            decoded_line = line.decode().strip()
            if decoded_line.startswith(self.config.tx_data_prefix):
                _, data_str = decoded_line.split(self.config.tx_data_prefix, 1)
                id, p, v, t, vb = map(float, data_str.split(","))
                id = int(id)
                if not id in state:
                    state[id] = p, v, t, vb

                if sorted(list(state.keys())) == sorted(self.motor_ids):
                    return state

        return state


if __name__ == "__main__":
    controller = SunnySkyController(
        SunnySkyConfig(port="/dev/tty.usbmodem11101"), motor_ids=[1]
    )

    try:
        while True:
            controller.set_pos([0.785])
            time.sleep(0.02)
    finally:
        controller.close_motors()
        print("Process completed successfully.")
