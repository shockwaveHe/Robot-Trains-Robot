import time
from dataclasses import dataclass
from typing import List

import numpy as np

from toddlerbot.actuation import BaseController, JointState
from toddlerbot.actuation.mighty_zap.mighty_zap_client import MightyZapClient
from toddlerbot.utils.file_utils import find_ports
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


class MightyZapController(BaseController):
    def __init__(self, config, motor_ids):
        super().__init__(config)

        self.config = config
        self.motor_ids = motor_ids

        self.connect_to_client()
        self.initialize_motors()

    def connect_to_client(self):
        try:
            if not isinstance(self.config.port, list):
                self.config.port = [self.config.port]

            self.clients = {}
            for motor_id, port in zip(self.motor_ids, self.config.port):
                client = MightyZapClient(
                    port, self.config.baudrate, self.config.timeout
                )
                self.clients[motor_id] = client
                log(f"Connected to the port: {port}", header="MightyZap")

            # self.executor = ThreadPoolExecutor(max_workers=len(self.clients))

        except Exception:
            raise ConnectionError("Could not connect to any MightyZap port.")

    def initialize_motors(self):
        log("Initializing motors...", header="MightyZap")
        for motor_id, client in self.clients.items():
            client.set_return_delay_time(motor_id, 0)
            precise_sleep(0.1)

        self.set_pos(self.config.init_pos)
        precise_sleep(0.1)

    def close_motors(self):
        for motor_id, client in self.clients.items():
            client.force_enable(motor_id, 0)

        # Need to be serparte from the previous for loop
        for client in self.clients.values():
            client.close()

    def set_pos_single(self, pos, motor_id):
        self.clients[motor_id].goal_position(motor_id, round(pos))

    def set_pos(self, pos, interp=False, vel=None, delta_t=None):
        def set_pos_helper(pos):
            for motor_id, pos in zip(self.motor_ids, pos):
                self.set_pos_single(pos, motor_id)

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

    # @profile()
    def get_motor_state_single(self, motor_id):
        # log(f"Start... {time.time()}", header="MightyZap", level="warning")
        joint_state = JointState(
            time=time.time(), pos=self.clients[motor_id].present_position(motor_id)
        )
        # log(f"End... {time.time()}", header="MightyZap", level="warning")
        return joint_state

    def get_motor_state(self):
        state_dict = {}
        for motor_id in self.motor_ids:
            state_dict[motor_id] = self.get_motor_state_single(motor_id)

        return state_dict


if __name__ == "__main__":
    pos_min = 1000
    pos_mid = 2000
    pos_max = 3000

    motor_ids = [0, 1, 2, 3]
    init_pos = [pos_mid] * len(motor_ids)
    controller = MightyZapController(
        MightyZapConfig(port=find_ports("USB Quad_Serial"), init_pos=init_pos),
        motor_ids=motor_ids,
    )

    pos_ref_seq = [
        [pos_max] * len(motor_ids),
        [pos_min] * len(motor_ids),
        [pos_max] * len(motor_ids),
        [pos_mid] * len(motor_ids),
    ]

    time_start = time.time()
    for pos_ref in pos_ref_seq:
        controller.set_pos(pos_ref)
        precise_sleep(0.02)

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

    controller.close_motors()

    log("Process completed successfully.", header="MightyZap")
