from dataclasses import dataclass
from typing import List

import numpy as np

from toddlerbot.actuation import BaseController
from toddlerbot.actuation.mighty_zap import mighty_zap


@dataclass
class MightyZapConfig:
    port: str
    init_pos: List[float]
    baudrate: int = 57600


class MightyZapController(BaseController):
    def __init__(self, config, motor_ids):
        super().__init__(config)

        self.config = config
        self.motor_ids = motor_ids

        self.client = self.connect_to_client()
        self.initialize_motors()

    def connect_to_client(self):
        try:
            mighty_zap.OpenMightyZap(self.config.port, self.config.baudrate)
            print(f"MightyZap: Connected to the port: {self.config.port}")
        except Exception as e:
            raise ConnectionError("Could not connect to any MightyZap port.")

    def initialize_motors(self):
        print("MightyZap: Initializing motors...")
        self.set_pos(self.config.init_pos)

    # Receive LEAP pos and directly control the robot
    def set_pos(self, pos):
        for i in self.motor_ids:
            mighty_zap.GoalPosition(i, pos[i])

    # read position
    def read_state(self):
        pos = []
        for i in self.motor_ids:
            pos.append(mighty_zap.PresentPosition(i))

        return pos


if __name__ == "__main__":
    init_pos = [1800, 1800]
    controller = MightyZapController(
        MightyZapConfig(port="/dev/tty.usbserial-0001", init_pos=init_pos),
        motor_ids=[0, 1],
    )

    controller.set_pos([3000, 3000])
    while True:
        state = controller.read_state()
        print(f"Actuator 1 Position: {state[0]}, Actuator 2 Position: {state[1]}")

        if state[0] >= 2990 and state[1] >= 2990:
            break

    controller.set_pos(init_pos)

    print("Process completed successfully.")
