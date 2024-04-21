from dataclasses import dataclass
from threading import Lock

import numpy as np
from adafruit_servokit import ServoKit

from toddlerbot.actuation import BaseController
from toddlerbot.utils.misc_utils import log, precise_sleep


@dataclass
class EMAXConfig:
    init_pos: np.ndarray
    gear_ratio: np.ndarray
    default_vel: float = np.pi / 2
    interp_method: str = "cubic"


@dataclass
class EMAXState:
    time: float
    pos: float
    # vel: float
    # current: float


class EMAXController(BaseController):
    def __init__(self, config, motor_ids):
        super().__init__(config)

        self.config = config
        self.motor_ids = motor_ids
        self.lock = Lock()

        self.client = self.connect_to_client()
        self.initialize_motors()

    def connect_to_client(self):
        pass

    def initialize_motors(self):
        log("Initializing motors...", header="EMAX")
        self.client = ServoKit(channels=16, frequency=50)
        for id in self.motor_ids:
            self.client.servo[id].actuation_range = 180
            self.client.servo[id].set_pulse_width_range(600, 2450)
        precise_sleep(0.1)

    def close_motors(self):
        pass

    # @profile
    # Receive pos and directly control the robot
    def set_pos(self, pos, interp=True, vel=None, delta_t=None):
        for id in self.motor_ids:
            self.client.servo[id].angle = np.degrees(pos[id])

        # def set_pos_helper(pos):
        #     pos = np.array(pos)
        #     pos_drive = self.config.init_pos - pos / self.config.gear_ratio
        #     # with self.lock:
        #     self.client.write_desired_pos(self.motor_ids, pos_drive)

        # if interp:
        #     pos = np.array(pos)
        #     pos_start = np.array(
        #         [state.pos for state in self.get_motor_state().values()]
        #     )
        #     if vel is None and delta_t is None:
        #         delta_t = max(np.abs(pos - pos_start) / self.config.default_vel)
        #     elif delta_t is None:
        #         delta_t = max(np.abs(pos - pos_start) / vel)

        #     interpolate_pos(
        #         set_pos_helper,
        #         pos_start,
        #         pos,
        #         delta_t,
        #         self.config.interp_method,
        #         "emax",
        #     )
        # else:
        #     set_pos_helper(pos)

    # @profile
    def get_motor_state(self):
        pass
        # state_dict = {}
        # # with self.lock:
        # pos_arr = self.client.read_pos()
        # # pos_arr, vel_arr, current_arr = self.client.read_pos_vel_cur()
        # # if current_arr.max() > 700:
        # #     log(f"Current: {current_arr.max()}", header="EMAX", level="debug")
        # pos_arr_driven = (self.config.init_pos - pos_arr) * self.config.gear_ratio
        # for i, id in enumerate(self.motor_ids):
        #     state_dict[id] = EMAXState(
        #         time=time.time(),
        #         pos=pos_arr_driven[i],
        #         # vel=vel_arr[i],
        #         # current=current_arr[i],
        #     )

        # return state_dict


if __name__ == "__main__":
    motor_ids = [0, 1]
    controller = EMAXController(
        EMAXConfig(init_pos=np.array([0.0]), gear_ratio=np.array([1.0])),
        motor_ids=motor_ids,
    )

    i = 0
    while i < 30:
        controller.set_pos([np.pi] * len(motor_ids))
        i += 1

    precise_sleep(1)
    i = 0
    while i < 30:
        controller.set_pos([0.0] * len(motor_ids))
        i += 1

    precise_sleep(0.1)

    controller.close_motors()

    log("Process completed successfully.", header="EMAX")
