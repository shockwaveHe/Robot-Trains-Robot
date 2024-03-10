from dataclasses import dataclass
from typing import List

import numpy as np

from toddlerbot.actuation import BaseController
from toddlerbot.actuation.dynamixel.dynamixel_client import *


@dataclass
class DynamixelConfig:
    port: str
    kP: List[float]
    kI: List[float]
    kD: List[float]
    current_limit: List[float]
    init_pos: List[float]
    baudrate: int = 57600
    control_mode: int = 5


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

            print(f"Dynamixel: Connected to the port: {self.config.port}")
            return client
        except Exception as e:
            raise ConnectionError("Could not connect to the Dynamixel port.")

    def initialize_motors(self):
        print("Dynamixel: Initializing motors...")
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
        self.client.write_desired_pos(self.motor_ids, np.array(self.config.init_pos))

    # Receive pos and directly control the robot
    def set_pos(self, pos):
        self.client.write_desired_pos(self.motor_ids, np.array(pos))

    # read state
    def read_state(self):
        return self.client.read_pos_vel_cur()


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

    pos = np.radians([150, 165, 165, 210, 165, 195])
    i = 0
    while i < 30:
        controller.set_pos(pos)
        state = controller.read_state()
        # print(state)
        p_error = np.abs(state[0] - pos)
        print(p_error)

        if p_error.mean() < 0.01:
            break

        i += 1

    i = 0
    while i < 30:
        controller.set_pos(controller.config.init_pos)
        state = controller.read_state()
        # print(state)
        p_error = np.abs(state[0] - controller.config.init_pos)
        print(p_error)

        if p_error.mean() < 0.01:
            break

        i += 1

    print("Process completed successfully.")
