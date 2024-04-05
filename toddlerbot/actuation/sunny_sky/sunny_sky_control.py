"""
Ben Katz
Motor Module Python API
Assumes the serial device is a nucleo running the firmware at:
Corresponding STM32F446 Firmware here:
https://os.mbed.com/users/benkatz/code/CanMaster/
"""

import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Tuple

import numpy as np
import serial

from toddlerbot.actuation import *


@dataclass
class SunnySkyConfig:
    port: str
    kP: int = 40
    kD: int = 50
    kD_schedule: Tuple = (0.1, 20)
    # overshoot: float = 0.1
    vel: float = np.pi / 2
    interp_method: str = "cubic"
    current_limit: float = 4.0
    gear_ratio: float = 2.0
    baudrate: int = 115200
    tx_data_prefix: str = ">tx_data:"
    tx_timeout: float = 0.1
    control_freq: int = 5000


@dataclass
class SunnySkyCommand:
    id: int
    p_des: float
    v_des: float
    kP: int
    kD: int
    i_ff: float


@dataclass
class SunnySkyState:
    time: float
    pos: float
    vel: float
    current: float
    voltage: float


class SunnySkyController(BaseController):
    def __init__(self, config, joint_range_dict):
        super().__init__(config)

        self.config = config
        self.motor_ids = list(joint_range_dict.keys())
        self.init_pos = {id: 0.0 for id in self.motor_ids}
        self.joint_range_dict = joint_range_dict
        self.serial_lock = Lock()
        self.client = self.connect_to_client()

        self.initialize_motors()

    def connect_to_client(self):
        try:
            client = serial.Serial(
                self.config.port, baudrate=self.config.baudrate, timeout=0.05
            )
            log(f"Connected to the port: {self.config.port}", header="SunnySky")
            return client
        except Exception as e:
            raise ConnectionError("Could not connect to any SunnySky port.")

    def initialize_motors(self):
        log("Initializing motors...", header="SunnySky")
        for id in self.motor_ids:
            self.enable_motor(id)

        self.calibrate_motors()

        time.sleep(0.1)

    def close_motors(self):
        for id in self.motor_ids:
            self.disable_motor(id)

    def write_to_serial_with_markers(self, data):
        with self.serial_lock:
            self.client.write(b"<" + data + b">")

    def send_command(self, command: SunnySkyCommand):
        """
        send_command(desired position, desired velocity, position gain, velocity gain, feed-forward current)

        Sends data over CAN, reads response, and populates rx_data with the response.
        """
        with self.serial_lock:
            self.client.reset_output_buffer()

        b = struct.pack(
            "<B2f2Hf",
            command.id,
            command.p_des,
            command.v_des,
            command.kP,
            command.kD,
            command.i_ff,
        )
        self.write_to_serial_with_markers(b)

    def enable_motor(self, id):
        """
        Puts motor with CAN ID "id" into torque-control mode. 2nd red LED will turn on
        """
        b = bytes([id]) + b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFC"
        self.write_to_serial_with_markers(b)
        time.sleep(0.1)
        # self.ser.flushInput()

    def disable_motor(self, id):
        """
        Removes motor with CAN ID "id" from torque-control mode. 2nd red LED will turn off
        """
        b = bytes([id]) + b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFD"
        self.write_to_serial_with_markers(b)
        time.sleep(0.1)

    def zero_motor(self, id):
        """
        Zero Position Sensor. Sets the mechanical position to zero.
        """
        b = bytes([id]) + b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFE"
        self.write_to_serial_with_markers(b)
        time.sleep(0.1)

    def calibrate_motors(self):
        log(f"Calibrating motors...", header="SunnySky")
        state_dict = self.get_motor_state()
        pos_curr = np.array([state.pos for state in state_dict.values()])
        joint_range = np.array(
            [
                self.joint_range_dict[id][1] - self.joint_range_dict[id][0]
                for id in self.motor_ids
            ]
        )

        log(f"Testing lower limit for motors...", header="SunnySky")
        self.set_pos(pos_curr - joint_range, limit=False)
        time.sleep(0.1)

        state_dict = self.get_motor_state()
        zero_pos = np.array([state.pos for state in state_dict.values()])
        log(f"Setting zero position {list(zero_pos)} for motors...", header="SunnySky")
        self.set_pos(zero_pos, limit=False)
        time.sleep(0.1)

        self.init_pos = {id: pos for id, pos in zip(self.motor_ids, zero_pos)}

    def _set_pos_single(
        self,
        id,
        pos,
        limit=True,
        schedule=True,
        delta_t=None,
        vel=None,
        kP=None,
        kD=None,
        i_ff=None,
    ):
        # Create command with dynamic kd and fixed kp, i_ff
        if limit:
            pos_driven = np.clip(pos, *sorted(self.joint_range_dict[id]))
        else:
            pos_driven = pos

        state = self._get_motor_state_single(id)
        pos_start_driven = state.pos

        if vel is None and delta_t is None:
            delta_t = np.abs(pos_driven - pos_start_driven) / self.config.vel
        elif delta_t is None:
            delta_t = np.abs(pos_driven - pos_start_driven) / vel

        time_start = time.time()
        time_curr = 0
        counter = 0
        while time_curr <= delta_t:
            time_curr = time.time() - time_start
            pos_interp_driven = interpolate(
                pos_start_driven,
                pos_driven,
                delta_t,
                time_curr,
                interp_type=self.config.interp_method,
            )
            pos_interp = (
                pos_interp_driven + self.init_pos[id]
            ) * self.config.gear_ratio

            if schedule and abs(state.pos - pos_driven) < self.config.kD_schedule[0]:
                kD_curr = self.config.kD_schedule[1]
            else:
                kD_curr = self.config.kD if kD is None else kD

            cmd = SunnySkyCommand(
                id=id,
                p_des=pos_interp,
                v_des=0.0,
                kP=self.config.kP if kP is None else kP,
                kD=kD_curr,
                i_ff=0.0 if i_ff is None else i_ff,
            )
            self.send_command(cmd)

            if not limit:
                state = self._get_motor_state_single(id)
                if abs(state.current) > self.config.current_limit:
                    log(
                        f"Motor {id} current limit reached: {state.current} A",
                        header="SunnySky",
                        level="warning",
                    )

            elapsed_time = time.time() - time_start - time_curr
            sleep_time = 1.0 / self.config.control_freq - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            counter += 1

        if limit:
            time_end = time.time()
            control_freq = counter / (time_end - time_start)
            log(f"Control frequency: {control_freq}", header="SunnySky", level="debug")

    def set_pos(
        self,
        pos,
        limit=True,
        schedule=True,
        delta_t=None,
        vel=None,
        kP=None,
        kD=None,
        i_ff=None,
    ):
        """
        Set the position of the motor
        """
        with ThreadPoolExecutor(max_workers=len(self.motor_ids)) as executor:
            for id, p in zip(self.motor_ids, pos):
                executor.submit(
                    self._set_pos_single,
                    id,
                    p,
                    limit,
                    schedule,
                    delta_t,
                    vel[id] if vel is not None and id in vel else None,
                    kP[id] if kP is not None and id in kP else None,
                    kD[id] if kD is not None and id in kD else None,
                    i_ff[id] if i_ff is not None and id in i_ff else None,
                )

    def _get_motor_state_single(self, id, max_retries=10):
        retries = 0
        while retries < max_retries:
            try:
                # Ensure the input buffer is clean
                with self.serial_lock:
                    self.client.reset_input_buffer()

                # Send the command
                command = bytes([id]) + b"get_state"
                self.write_to_serial_with_markers(command)

                # Wait for response
                valid_data = self._wait_for_response(id)
                if valid_data:
                    return valid_data  # Return on first successful data receipt

            except Exception as e:
                log(
                    f"Error during motor state retrieval: {e}",
                    header="SunnySky",
                    level="error",
                )

            retries += 1

        log(
            f"Failed to retrieve motor state after maximum retries.",
            header="SunnySky",
            level="error",
        )
        return None

    def _wait_for_response(self, id):
        time_start = time.time()
        while (time.time() - time_start) < self.config.tx_timeout:
            with self.serial_lock:
                line = self.client.readline()
                if not line:
                    continue  # Skip empty lines

            decoded_line = line.decode().strip()
            if decoded_line.startswith(self.config.tx_data_prefix):
                try:
                    _, data_str = decoded_line.split(self.config.tx_data_prefix, 1)
                    id_recv, p, v, t, vb = map(float, data_str.split(","))
                    if id_recv == id:  # ID matches requested
                        return SunnySkyState(
                            time=time.time(),
                            pos=p / self.config.gear_ratio - self.init_pos[id],
                            vel=v,
                            current=t,
                            voltage=vb,
                        )
                except ValueError:
                    log(
                        "Parsing error for received data.",
                        header="SunnySky",
                        level="warning",
                    )

        return None

    def get_motor_state(self):
        with ThreadPoolExecutor(max_workers=len(self.motor_ids)) as executor:
            future_dict = {}
            for id in self.motor_ids:
                future_dict[id] = executor.submit(self._get_motor_state_single, id)

            state_dict = {}
            for id in self.motor_ids:
                state_dict[id] = future_dict[id].result()

            return state_dict


if __name__ == "__main__":
    from toddlerbot.utils.vis_plot import plot_line_graph

    joint_range_dict = {1: (0, np.pi / 2), 2: (0, -np.pi / 2)}
    pos_seq_ref = [
        [0.0, 0.0],
        [np.pi / 2, -np.pi / 4],
        [0.0, 0.0],
        [np.pi / 4, -np.pi / 2],
        [0.0, 0.0],
    ]

    # joint_range_dict = {2: (0, -np.pi / 2)}
    # pos_seq_ref = [[0.0], [-np.pi / 2], [0.0], [-np.pi / 4], [0.0]]

    # joint_range_dict = {1: (0, np.pi / 2)}
    # pos_seq_ref = [[0.0], [np.pi / 2], [0.0], [np.pi / 4], [0.0]]

    config = SunnySkyConfig(port="/dev/tty.usbmodem101", vel=np.pi / 4)
    controller = SunnySkyController(config, joint_range_dict=joint_range_dict)

    time_seq = []
    pos_seq = []
    time_start = time.time()
    for pos_ref in pos_seq_ref:
        controller.set_pos(pos_ref)
        state_dict = controller.get_motor_state()

        message = "Motor states:"
        for id, state in state_dict.items():
            message += f" {id}: {state.pos:.4f} at {state.time - time_start:.4f}s"

        log(message, header="SunnySky", level="debug")

    controller.close_motors()

    log("Process completed successfully.", header="SunnySky")
