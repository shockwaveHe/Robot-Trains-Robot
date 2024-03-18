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
    joint_limits: Tuple = (-np.pi / 2, np.pi / 2)
    gear_ratio: float = 2.0
    baudrate: int = 115200
    tx_data_prefix: str = ">tx_data:"
    tx_timeout: float = 1.0
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
    def __init__(self, config, motor_ids):
        super().__init__(config)

        self.config = config
        self.motor_ids = motor_ids
        self.init_pos = {id: 0.0 for id in motor_ids}
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
            self.calibrate_motor(id)

        self.set_pos([0.0] * len(self.motor_ids))

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
        b = b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFC"
        b = b + bytes(bytearray([id]))
        self.write_to_serial_with_markers(b)
        time.sleep(0.1)
        # self.ser.flushInput()

    def disable_motor(self, id):
        """
        Removes motor with CAN ID "id" from torque-control mode. 2nd red LED will turn off
        """
        b = b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFD"
        b = b + bytes(bytearray([id]))
        self.write_to_serial_with_markers(b)
        time.sleep(0.1)

    def zero_motor(self, id):
        """
        Zero Position Sensor. Sets the mechanical position to zero.
        """
        b = b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFE"
        b = b + bytes(bytearray([id]))
        self.write_to_serial_with_markers(b)
        time.sleep(0.1)

    def calibrate_motor(self, id):
        log(f"Calibrating motor with ID {id}...", header="SunnySky")
        pos_curr = self._read_state_single(id).pos
        joint_range = self.config.joint_limits[1] - self.config.joint_limits[0]

        log(f"Setting lower limit for motor with ID {id}...", header="SunnySky")
        self._set_pos_single(id, pos_curr - joint_range, limit=False)
        time.sleep(0.1)
        self.lower_limit = self._read_state_single(id).pos

        # log(f"Setting upper limit for motor with ID {id}...", header="SunnySky")
        # self._set_pos_single(id, pos_curr + joint_range, limit=False)
        # time.sleep(0.1)
        # self.upper_limit = self._read_state_single(id).pos

        log(f"Setting zero position for motor with ID {id}...", header="SunnySky")
        zero_pos = self.lower_limit + np.pi / 2
        self._set_pos_single(id, zero_pos, limit=False)
        time.sleep(0.1)
        self.init_pos[id] = zero_pos

    def _set_pos_single(
        self, id, pos, limit=True, schedule=True, vel=None, kP=None, kD=None, i_ff=None
    ):
        # Create command with dynamic kd and fixed kp, i_ff
        if limit:
            pos_driven = np.clip(pos, *self.config.joint_limits)
        else:
            pos_driven = pos

        if vel is None:
            vel = self.config.vel

        state = self._read_state_single(id)
        pos_start_driven = state.pos
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
                state = self._read_state_single(id)
                if abs(state.current) > self.config.current_limit:
                    log(
                        f"Current limit reached: {state.current} A",
                        header="SunnySky",
                        level="warning",
                    )

            elapsed_time = time.time() - time_start - time_curr
            sleep_time = 1.0 / self.config.control_freq - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            counter += 1

        time_end = time.time()
        control_freq = counter / (time_end - time_start)
        log(f"Control frequency: {control_freq}", header="SunnySky", level="debug")

    def set_pos(
        self, pos, limit=True, schedule=True, vel=None, kP=None, kD=None, i_ff=None
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
                    vel[id] if vel is not None and id in vel else None,
                    kP[id] if kP is not None and id in kP else None,
                    kD[id] if kD is not None and id in kD else None,
                    i_ff[id] if i_ff is not None and id in i_ff else None,
                )

    def _read_state_single(self, id):
        self.client.reset_input_buffer()

        time_start = time.time()
        time_curr = 0
        while time_curr < self.config.tx_timeout:
            with self.serial_lock:
                line = self.client.readline()

            decoded_line = line.decode().strip()
            time_curr = time.time() - time_start

            if decoded_line.startswith(self.config.tx_data_prefix):
                _, data_str = decoded_line.split(self.config.tx_data_prefix, 1)
                id_recv, p, v, t, vb = map(float, data_str.split(","))
                id_recv = int(id_recv)
                if id_recv == id:
                    p = p / self.config.gear_ratio - self.init_pos[id]
                    return SunnySkyState(
                        time=time.time(),
                        pos=p,
                        vel=v,
                        current=t,
                        voltage=vb,
                    )

        return None

    def read_state(self):
        with ThreadPoolExecutor(max_workers=len(self.motor_ids)) as executor:
            future_dict = {}
            for id in self.motor_ids:
                future_dict[id] = executor.submit(self._read_state_single, id)

            state_dict = {}
            for id in self.motor_ids:
                state_dict[id] = future_dict[id].result()

            return state_dict


if __name__ == "__main__":
    from toddlerbot.utils.vis_plot import plot_line_graph

    id = 1
    config = SunnySkyConfig(port="/dev/tty.usbmodem1201")
    controller = SunnySkyController(config, motor_ids=[id])

    pos_seq_ref = [0.0, np.pi / 4, -np.pi / 4, np.pi / 4, -np.pi / 4, np.pi / 4, 0.0]
    time_seq_ref = [0.0] + list(np.cumsum(np.abs(np.diff(pos_seq_ref))) / config.vel)

    time_seq = []
    pos_seq = []
    time_start = time.time()
    try:
        for pos_ref in pos_seq_ref:
            state_dict = controller.set_pos([pos_ref])
            time_seq += [state.time - time_start for state in state_dict[id]]
            pos_seq += [state.pos for state in state_dict[id]]
    finally:
        controller.close_motors()
        log("Process completed successfully.", header="SunnySky")

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
