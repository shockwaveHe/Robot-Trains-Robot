import time
from dataclasses import dataclass
from typing import List

import numpy as np

from toddlerbot.actuation import *
from toddlerbot.actuation.mighty_zap import mighty_zap


@dataclass
class MightyZapConfig:
    port: str
    init_pos: List[float]
    baudrate: int = 57600


@dataclass
class MightyZapState:
    time: float
    pos: float


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
            log(f"Connected to the port: {self.config.port}", header="MightyZap")
        except Exception as e:
            raise ConnectionError("Could not connect to any MightyZap port.")

    def initialize_motors(self):
        log("Initializing motors...", header="MightyZap")
        self.set_pos(self.config.init_pos)
        time.sleep(0.1)

    def close_motors(self):
        mighty_zap.CloseMightyZap()

    # Receive LEAP pos and directly control the robot
    def set_pos(self, pos):
        for i in self.motor_ids:
            mighty_zap.GoalPosition(i, pos[i])

    # read position
    def read_state(self):
        state_dict = {}
        for i, id in enumerate(self.motor_ids):
            state_dict[id] = MightyZapState(
                time=time.time(), pos=mighty_zap.PresentPosition(id)
            )

        return state_dict


if __name__ == "__main__":
    init_pos = [1808, 1808]
    controller = MightyZapController(
        MightyZapConfig(port="/dev/tty.usbserial-0001", init_pos=init_pos),
        motor_ids=[0, 1],
    )

    controller.set_pos([3000, 3000])
    while True:
        state = controller.read_state()
        log(
            f"Actuator 1 Position: {state[0]}, Actuator 2 Position: {state[1]}",
            header="MightyZap",
        )

        if state[0] >= 2990 and state[1] >= 2990:
            break

    controller.set_pos(init_pos)
    time.sleep(1)

    controller.close_motors()

    log("Process completed successfully.", header="MightyZap")
