import time
from dataclasses import dataclass
from typing import List

import numpy as np

from toddlerbot.actuation import *
from toddlerbot.actuation.mighty_zap.mighty_zap_client import MightyZapClient


@dataclass
class MightyZapConfig:
    port: str
    init_pos: List[float]
    baudrate: int = 115200
    default_vel: float = 4000
    interp_method: str = "cubic"
    interp_freq: int = 1000


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
            client = MightyZapClient(self.config.port, self.config.baudrate)
            log(f"Connected to the port: {self.config.port}", header="MightyZap")
            return client
        except Exception as e:
            raise ConnectionError("Could not connect to any MightyZap port.")

    def initialize_motors(self):
        log("Initializing motors...", header="MightyZap")
        self.set_pos(self.config.init_pos)
        time.sleep(0.1)

    def close_motors(self):
        self.client.force_enable(self.motor_ids, [0] * len(self.motor_ids))
        self.client.close()

    # @profile
    def set_pos(self, pos, interp=True, vel=None, delta_t=None):
        def set_pos_helper(pos):
            rounded_pos = [round(p) for p in pos]
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
            if pos < 0:
                log(
                    f"Read the MightyZap {id} Position failed.",
                    header="MightyZap",
                    level="warning",
                )

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
