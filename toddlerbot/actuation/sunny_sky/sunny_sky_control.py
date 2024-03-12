"""
Ben Katz
Motor Module Python API
Assumes the serial device is a nucleo running the firmware at:
Corresponding STM32F446 Firmware here:
https://os.mbed.com/users/benkatz/code/CanMaster/
"""

import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import serial

from toddlerbot.actuation import BaseController


@dataclass
class SunnySkyConfig:
    port: str
    init_pos: float = np.pi
    kP: int = 40
    kD: int = 8
    # overshoot: float = 0.1
    vel: float = np.pi
    baudrate: int = 115200
    tx_data_prefix: str = ">tx_data:"
    tx_timeout: float = 1.0
    joint_limits: Tuple = (-np.pi / 2, np.pi / 2)
    gear_ratio: float = 2.0


@dataclass
class SunnySkyCommand:
    id: int = 1
    p_des: float = 0.0
    v_des: float = 0.0
    kP: int = 40
    kD: int = 8
    i_ff: float = 0.0


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

        self.set_pos([0.0] * len(self.motor_ids))
        time.sleep(1)

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
            command.kP,
            command.kD,
            command.i_ff,
        )
        self.client.reset_output_buffer()
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

    def set_pos(self, pos, vel=None, kP=None, kD=None, i_ff=None):
        """
        Set the position of the motor
        """
        pos_raw = np.clip(pos, self.config.joint_limits[0], self.config.joint_limits[1])
        if vel is None:
            vel = self.config.vel

        for id, p_des_raw in zip(self.motor_ids, pos_raw):
            # Create command with dynamic kd and fixed kp, i_ff
            time_start = time.time()
            time_now = 0
            p_start_raw = self.read_state()[id][0]
            delta_t = np.abs(p_des_raw - p_start_raw) / vel
            # print(
            #     f"Moving motor {id} from {p_start_raw} to {p_des_raw} in {delta_t} seconds..."
            # )
            while time_now <= delta_t:
                time_now = time.time() - time_start
                pos_interp_raw = np.interp(
                    time_now, [0, delta_t], [p_start_raw, p_des_raw]
                )
                pos_interp = (
                    pos_interp_raw * self.config.gear_ratio + self.config.init_pos
                )
                # print(time_now, pos_interp_raw, pos_interp)

                cmd = SunnySkyCommand(
                    id=id,
                    p_des=pos_interp,
                    v_des=0.0,
                    kP=self.config.kP if kP is None else kP,
                    kD=self.config.kD if kD is None else kD,
                    i_ff=0.0 if i_ff is None else i_ff,
                )
                self.send_command(cmd)

                self.client.reset_input_buffer()

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
                p = (p - self.config.init_pos) / self.config.gear_ratio
                if not id in state:
                    state[id] = p, v, t, vb

                if sorted(list(state.keys())) == sorted(self.motor_ids):
                    return state

        return state


if __name__ == "__main__":
    from toddlerbot.utils.vis_plot import plot_line_graph

    config = SunnySkyConfig(port="/dev/tty.usbmodem1301")
    controller = SunnySkyController(config, motor_ids=[1])

    pos_seq_ref = [
        0.0,
        np.pi / 6,
        0.0,
        -np.pi / 6,
        0.0,
        np.pi / 6,
        0.0,
        -np.pi / 6,
        0.0,
        np.pi / 6,
        0.0,
    ]
    time_seq_ref = [0.0] + list(np.cumsum(np.abs(np.diff(pos_seq_ref))) / config.vel)

    time_seq = []
    pos_seq = []
    time_start = time.time()
    try:
        for pos_ref in pos_seq_ref:
            controller.set_pos([pos_ref])
            pos_now = controller.read_state()[1][0]
            time_now = time.time() - time_start
            pos_seq.append(pos_now)
            time_seq.append(time_now)
    finally:
        controller.close_motors()

        plot_line_graph(
            [pos_seq_ref, pos_seq],
            [time_seq_ref, time_seq],
            title="PD Tuning",
            x_label="time (s)",
            y_label="position (rad)",
            save_config=True,
            save_path="results/plots",
            time_suffix=f"",
            legend_labels=["ref", "real"],
        )()
        print("Process completed successfully.")
