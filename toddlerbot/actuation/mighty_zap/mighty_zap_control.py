import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import List

import numpy as np

from toddlerbot.actuation import *
from toddlerbot.actuation.mighty_zap import mighty_zap


@dataclass
class MightyZapConfig:
    port: str
    init_pos: List[float]
    vel: float = 4000
    interp_method: str = "cubic"
    baudrate: int = 115200
    control_freq: int = 2000


@dataclass
class MightyZapState:
    time: float
    pos: float


class MightyZapController(BaseController):
    def __init__(self, config, motor_ids):
        super().__init__(config)

        self.config = config
        self.motor_ids = motor_ids
        self.serial_lock = Lock()

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
        for id in self.motor_ids:
            mighty_zap.ForceEnable(id, 0)

        mighty_zap.CloseMightyZap()

    # @profile
    def set_pos(self, pos, delta_t=None, vel=None):
        pos = np.array(pos)
        state_dict = self.get_motor_state()
        pos_start = np.array([state.pos for state in state_dict.values()])

        if vel is None and delta_t is None:
            delta_t = max(np.abs(pos - pos_start) / self.config.vel)
        elif delta_t is None:
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

            for id, p in zip(self.motor_ids, pos_interp):
                mighty_zap.GoalPosition(id, round(p))

            elapsed_time = time.time() - time_start - time_curr
            sleep_time = 1.0 / self.config.control_freq - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            counter += 1

        time_end = time.time()
        control_freq = counter / (time_end - time_start)
        log(f"Control frequency: {control_freq}", header="MightyZap", level="debug")

    # @profile
    def get_motor_state(self):
        state_dict = {}
        for id in self.motor_ids:
            pos = -1
            while pos < 0:
                pos = mighty_zap.PresentPosition(id)

            state_dict[id] = MightyZapState(time=time.time(), pos=pos)

        return state_dict


if __name__ == "__main__":
    motor_ids = [0, 1, 2, 3]
    init_pos = [1808] * len(motor_ids)
    controller = MightyZapController(
        MightyZapConfig(port="/dev/tty.usbserial-0001", init_pos=init_pos),
        motor_ids=motor_ids,
    )

    controller.set_pos([3000] * len(motor_ids))
    while True:
        state_dict = controller.get_motor_state()
        message = "Motor states:"
        for id, state in state_dict.items():
            message += f" {id}: {state.pos}"

        log(message, header="MightyZap", level="debug")
        if state_dict[0].pos >= 2990:
            break

    controller.set_pos(init_pos)
    time.sleep(1)

    controller.close_motors()

    log("Process completed successfully.", header="MightyZap")
