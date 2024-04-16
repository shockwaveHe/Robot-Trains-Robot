import time
from dataclasses import dataclass
from threading import Lock
from typing import List

import numpy as np

from toddlerbot.actuation import BaseController
from toddlerbot.actuation.dynamixel.dynamixel_client import DynamixelClient
from toddlerbot.utils.math_utils import interpolate_pos
from toddlerbot.utils.misc_utils import log, precise_sleep


@dataclass
class DynamixelConfig:
    port: str
    kFF2: List[float]
    kFF1: List[float]
    kP: List[float]
    kI: List[float]
    kD: List[float]
    current_limit: List[float]
    init_pos: np.ndarray
    gear_ratio: np.ndarray
    baudrate: int = 3000000
    control_mode: int = 5
    default_vel: float = np.pi / 2
    interp_method: str = "cubic"


@dataclass
class DynamixelState:
    time: float
    pos: float
    # vel: float
    # current: float


class DynamixelController(BaseController):
    def __init__(self, config, motor_ids):
        super().__init__(config)

        self.config = config
        self.motor_ids = motor_ids
        self.lock = Lock()

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
        except Exception:
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
        self.client.sync_write(self.motor_ids, self.config.kD, 80, 2)
        self.client.sync_write(self.motor_ids, self.config.kI, 82, 2)
        self.client.sync_write(self.motor_ids, self.config.kP, 84, 2)
        self.client.sync_write(self.motor_ids, self.config.kFF2, 88, 2)
        self.client.sync_write(self.motor_ids, self.config.kFF1, 90, 2)
        self.client.sync_write(self.motor_ids, self.config.current_limit, 102, 2)
        self.set_pos(np.zeros(len(self.motor_ids)))
        precise_sleep(0.1)

    def close_motors(self):
        open_clients = list(DynamixelClient.OPEN_CLIENTS)
        for open_client in open_clients:
            if open_client.port_handler.is_using:
                log("Forcing client to close.", header="Dynamixel")
            open_client.port_handler.is_using = False
            open_client.disconnect()

    # @profile
    # Receive pos and directly control the robot
    def set_pos(self, pos, interp=True, vel=None, delta_t=None):
        def set_pos_helper(pos):
            pos = np.array(pos)
            pos_drive = self.config.init_pos - pos / self.config.gear_ratio
            # with self.lock:
            self.client.write_desired_pos(self.motor_ids, pos_drive)

        if interp:
            pos = np.array(pos)
            pos_start = np.array(
                [state.pos for state in self.get_motor_state().values()]
            )
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
                "dynamixel",
            )
        else:
            set_pos_helper(pos)

    # @profile
    def get_motor_state(self):
        state_dict = {}
        # with self.lock:
        pos_arr = self.client.read_pos()
        # pos_arr, vel_arr, current_arr = self.client.read_pos_vel_cur()
        # if current_arr.max() > 700:
        #     log(f"Current: {current_arr.max()}", header="Dynamixel", level="debug")
        pos_arr_driven = (self.config.init_pos - pos_arr) * self.config.gear_ratio
        for i, id in enumerate(self.motor_ids):
            state_dict[id] = DynamixelState(
                time=time.time(),
                pos=pos_arr_driven[i],
                # vel=vel_arr[i],
                # current=current_arr[i],
            )

        return state_dict


if __name__ == "__main__":
    controller = DynamixelController(
        DynamixelConfig(
            port="/dev/tty.usbserial-FT8ISUJY",
            kFF2=[0, 0, 0, 0, 0, 0],
            kFF1=[0, 0, 0, 0, 0, 0],
            kP=[400, 1200, 1200, 400, 1200, 1200],
            kI=[0, 0, 0, 0, 0, 0],
            kD=[200, 400, 400, 200, 400, 400],
            current_limit=[350, 350, 350, 350, 350, 350],
            init_pos=np.radians([245, 180, 180, 322, 180, 180]),
            gear_ratio=np.array([19 / 21, 1, 1, 19 / 21, 1, 1]),
        ),
        motor_ids=[7, 8, 9, 10, 11, 12],
    )

    i = 0
    while i < 30:
        controller.set_pos(
            # [np.pi / 12] * 6
            [0.0, np.pi / 12, np.pi / 12, np.pi / 2, np.pi / 12, np.pi / 12]
        )
        i += 1

    i = 0
    while i < 30:
        controller.set_pos([0.0] * 6)
        i += 1

    precise_sleep(0.1)
    controller.close_motors()

    log("Process completed successfully.", header="Dynamixel")
