from dataclasses import dataclass, field
from typing import List

import numpy as np

from toddlerbot.actuation.dynamixel.dynamixel_client import *


@dataclass
class DynamixelConfig:
    kP: float = 600
    kI: float = 0
    kD: float = 200
    current_limit: float = 350
    baud_rate: int = 57600
    control_mode: int = 5
    ports: List[str] = field(
        default_factory=lambda: ["/dev/ttyUSB0", "/dev/ttyUSB1", "COM13"]
    )


class DynamixelNode:
    def __init__(self, config, n_motors, init_pos):
        self.config = config
        self.n_motors = n_motors
        self.curr_pos = init_pos

        self.dxl_client = self.connect_to_client(self.config.ports)
        self.initialize_motors()

    def connect_to_client(self, ports):
        for port in ports:
            try:
                client = DynamixelClient(
                    list(range(self.n_motors)), port, self.config.baud_rate
                )
                client.connect()
                return client
            except Exception as e:
                print(f"Failed to connect on {port}: {e}")
        raise ConnectionError("Could not connect to any Dynamixel port.")

    def initialize_motors(self):
        motors = list(range(self.n_motors))
        self.dxl_client.sync_write(
            motors, np.ones(len(motors)) * self.config.control_mode, 11, 1
        )
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.config.kP, 84, 2)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.config.kI, 82, 2)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.config.kD, 80, 2)
        self.dxl_client.sync_write(
            motors, np.ones(len(motors)) * self.config.current_limit, 102, 2
        )
        self.dxl_client.write_desired_pos(motors, self.curr_pos)

        self.motors = motors

    # Receive LEAP pos and directly control the robot
    def set_pos(self, pos):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pos)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # read position
    def read_pos(self):
        return self.dxl_client.read_pos()

    # read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()

    # read current
    def read_cur(self):
        return self.dxl_client.read_cur()
