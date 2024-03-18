from dataclasses import dataclass
from typing import List

import numpy as np

from toddlerbot.actuation import *
from toddlerbot.actuation.dynamixel.dynamixel_client import *


@dataclass
class DynamixelConfig:
    port: str
    kP: List[float]
    kI: List[float]
    kD: List[float]
    current_limit: List[float]
    init_pos: List[float]
    vel: float = np.pi / 2
    interp_method: str = "cubic"
    baudrate: int = 3000000
    control_mode: int = 5
    control_freq: int = 5000


@dataclass
class DynamixelState:
    time: float
    pos: float
    vel: float
    current: float


class DynamixelController(BaseController):
    def __init__(self, config, motor_ids):
        super().__init__(config)

        self.config = config
        self.motor_ids = motor_ids

        self.client = self.connect_to_client()
        self.initialize_motors()

    def connect_to_client(self):
        try:
            client = DynamixelClient(
                self.motor_ids, self.config.port, self.config.baudrate
            )
            client.connect()

            log(f"Connected to the port: {self.config.port}", header="Dynamixel")
            return client
        except Exception as e:
            raise ConnectionError("Could not connect to the Dynamixel port.")

    def initialize_motors(self):
        log("Initializing motors...", header="Dynamixel")
        self.client.sync_write(
            self.motor_ids,
            np.ones(len(self.motor_ids)) * self.config.control_mode,
            11,
            1,
        )
        self.client.set_torque_enabled(self.motor_ids, True)
        self.client.sync_write(self.motor_ids, self.config.kP, 84, 2)
        self.client.sync_write(self.motor_ids, self.config.kI, 82, 2)
        self.client.sync_write(self.motor_ids, self.config.kD, 80, 2)
        self.client.sync_write(
            self.motor_ids,
            self.config.current_limit,
            102,
            2,
        )
        self.set_pos(np.zeros(len(self.motor_ids)))
        time.sleep(0.1)

    def close_motors(self):
        open_clients = list(DynamixelClient.OPEN_CLIENTS)
        for open_client in open_clients:
            if open_client.port_handler.is_using:
                logging.warning("Forcing client to close.")
            open_client.port_handler.is_using = False
            open_client.disconnect()

    # @profile
    # Receive pos and directly control the robot
    def set_pos(self, pos, vel=None):
        # pos_raw = self.config.init_pos - np.array(pos)
        # self.client.write_desired_pos(self.motor_ids, pos_raw)

        pos = np.array(pos)
        if vel is None:
            vel = self.config.vel

        pos_start = self.config.init_pos - self.client.read_pos()
        delta_t = max(np.abs(pos - pos_start) / vel)
        time_start = time.time()
        time_curr = 0
        counter = 0
        while time_curr <= delta_t:
            time_curr = time.time() - time_start
            pos_interp = interpolate(
                pos_start,
                pos,
                delta_t,
                time_curr,
                interp_type=self.config.interp_method,
            )
            pos_interp_raw = self.config.init_pos - pos_interp

            self.client.write_desired_pos(self.motor_ids, pos_interp_raw)

            elapsed_time = time.time() - time_start - time_curr
            sleep_time = 1.0 / self.config.control_freq - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            counter += 1

        time_end = time.time()
        control_freq = counter / (time_end - time_start)
        log(f"Control frequency: {control_freq}", header="Dynamixel", level="debug")

    # @profile
    def get_motor_state(self):
        state_dict = {}
        pos_arr_raw, vel_arr, current_arr = self.client.read_pos_vel_cur()
        pos_arr = self.config.init_pos - pos_arr_raw
        for i, id in enumerate(self.motor_ids):
            state_dict[id] = DynamixelState(
                time=time.time(),
                pos=pos_arr[i],
                vel=vel_arr[i],
                current=current_arr[i],
            )

        return state_dict


if __name__ == "__main__":
    controller = DynamixelController(
        DynamixelConfig(
            port="/dev/tty.usbserial-FT8ISUJY",
            kP=[100, 200, 200, 100, 200, 200],
            kI=[0, 0, 0, 0, 0, 0],
            kD=[100, 100, 100, 100, 100, 100],
            current_limit=[350, 350, 350, 350, 350, 350],
            init_pos=np.radians([135, 180, 180, 225, 180, 180]),
        ),
        motor_ids=[7, 8, 9, 10, 11, 12],
    )

    i = 0
    while i < 30:
        controller.set_pos([np.pi / 12] * 6)
        i += 1

    i = 0
    while i < 30:
        controller.set_pos([0.0] * 6)
        i += 1

    controller.close_motors()

    log("Process completed successfully.", header="Dynamixel")
