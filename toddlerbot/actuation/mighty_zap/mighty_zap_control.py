import time
from dataclasses import dataclass
from threading import Lock
from typing import List

import numpy as np

from toddlerbot.actuation import BaseController
from toddlerbot.actuation.mighty_zap.mighty_zap_client import MightyZapClient
from toddlerbot.utils.math_utils import interpolate_pos
from toddlerbot.utils.misc_utils import log, precise_sleep


@dataclass
class MightyZapConfig:
    port: str
    init_pos: List[float]
    baudrate: int = 115200
    timeout: float = 0.1
    default_vel: float = 8000
    interp_method: str = "cubic"
    # Recommendable delay time is 5msec for data write, 10msec for data read.
    # Therefore, the maximum frequency is 200Hz.
    interp_freq: int = 100


@dataclass
class MightyZapState:
    time: float
    pos: float


class MightyZapController(BaseController):
    def __init__(self, config, motor_ids):
        super().__init__(config)

        self.config = config
        self.motor_ids = motor_ids
        self.last_pos = {id: 0.0 for id in self.motor_ids}
        self.lock = Lock()

        self.client = self.connect_to_client()
        self.initialize_motors()

    def connect_to_client(self):
        try:
            client = MightyZapClient(
                self.config.port, self.config.baudrate, self.config.timeout
            )
            log(f"Connected to the port: {self.config.port}", header="MightyZap")
            return client
        except Exception:
            raise ConnectionError("Could not connect to any MightyZap port.")

    def initialize_motors(self):
        log("Initializing motors...", header="MightyZap")
        self.set_pos(self.config.init_pos)
        precise_sleep(0.1)

    def close_motors(self):
        self.client.force_enable(self.motor_ids, [0] * len(self.motor_ids))
        self.client.close()

    # @profile
    def set_pos(self, pos, interp=True, vel=None, delta_t=None):
        def set_pos_helper(pos):
            rounded_pos = [round(p) for p in pos]
            # log(f"Goal: {rounded_pos}", header="MightyZap", level="debug")
            self.client.goal_position(self.motor_ids, rounded_pos)
            # for id, p in zip(self.motor_ids, rounded_pos):
            #     self.client.goal_position(id, p)``

        if interp:
            pos = np.array(pos)
            state_dict = self.get_motor_state()
            pos_start = np.array([state.pos for state in state_dict.values()])

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
                "mighty_zap",
                sleep_time=1 / self.config.interp_freq,
            )
        else:
            set_pos_helper(pos)

    # @profile
    def get_motor_state(self):
        state_dict = {}
        for id in self.motor_ids:
            pos = self.client.present_position(id)
            # log(f"ID: {id}, Present: {pos}", header="MightyZap", level="debug")
            if pos < 0:
                pos = self.last_pos[id]
                log(
                    f"Read the MightyZap {id} Position failed. "
                    + f"Use the last position {self.last_pos[id]}.",
                    header="MightyZap",
                    level="warning",
                )
            else:
                self.last_pos[id] = pos

            state_dict[id] = MightyZapState(time=time.time(), pos=pos)

        return state_dict


if __name__ == "__main__":
    motor_ids = [0, 1, 2, 3]
    init_pos = [1488] * len(motor_ids)
    controller = MightyZapController(
        MightyZapConfig(port="/dev/tty.usbserial-0001", init_pos=init_pos),
        motor_ids=motor_ids,
    )

    pos_max = 2000
    pos_min = 1000
    pos_ref_seq = [
        [pos_max, pos_max, pos_max, pos_max],
        [pos_max, pos_min, pos_max, pos_min],
        [pos_min, pos_max, pos_min, pos_max],
        [pos_min, pos_min, pos_min, pos_min],
    ]

    time_start = time.time()
    for pos_ref in pos_ref_seq:
        controller.set_pos(pos_ref)

        while True:
            state_dict = controller.get_motor_state()
            pos = [state.pos for state in state_dict.values()]

            if np.allclose(pos, pos_ref, atol=10):
                break

            precise_sleep(0.02)

        message = "Motor states:"
        for id, state in state_dict.items():
            message += f" {id}: {state.pos:.1f};"

        message += f" Time: {state.time - time_start:.4f}s"

        log(message, header="MightyZap", level="debug")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    controller.close_motors()

    log("Process completed successfully.", header="MightyZap")
