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
    vel: float = 1000
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

    # Receive LEAP pos and directly control the robot
    def _set_pos_single(self, id, pos, vel=None):
        if vel is None:
            vel = self.config.vel

        state = self._read_state_single(id)
        pos_start = state.pos
        delta_t = np.abs(pos - pos_start) / vel

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
            with self.serial_lock:
                mighty_zap.GoalPosition(id, round(pos_interp))

            elapsed_time = time.time() - time_start - time_curr
            sleep_time = 1.0 / self.config.control_freq - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            counter += 1

            # print(
            #     f"ID: {id}, Start: {pos_start}, Pos: {state.pos}, Ref: {pos_interp}, Time: {time_curr}"
            # )

        time_end = time.time()
        control_freq = counter / (time_end - time_start)
        log(f"Control frequency: {control_freq}", header="MightyZap", level="debug")

    def set_pos(self, pos, vel=None):
        with ThreadPoolExecutor(max_workers=len(self.motor_ids)) as executor:
            for id, p in zip(self.motor_ids, pos):
                executor.submit(
                    self._set_pos_single,
                    id,
                    p,
                    vel is None and self.config.vel or vel,
                )

    def _read_state_single(self, id):
        with self.serial_lock:
            pos = -1
            while pos < 0:
                pos = mighty_zap.PresentPosition(id)

        return MightyZapState(time=time.time(), pos=pos)

    # read position
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

        if state[0].pos >= 2990 and state[1].pos >= 2990:
            break

    controller.set_pos(init_pos)
    time.sleep(1)

    controller.close_motors()

    log("Process completed successfully.", header="MightyZap")
