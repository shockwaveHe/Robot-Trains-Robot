import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

from toddlerbot.utils.math_utils import *
from toddlerbot.utils.misc_utils import log


# @profile
def interpolate_pos(
    set_pos, pos_start, pos, delta_t, interp_type, actuator_type, sleep_time=0.0
):
    time_start = time.time()
    time_curr = 0
    counter = 0
    while time_curr <= delta_t:
        time_curr = time.time() - time_start
        pos_interp = interpolate(
            pos_start, pos, delta_t, time_curr, interp_type=interp_type
        )
        set_pos(pos_interp)

        time_until_next_step = sleep_time - (time.time() - time_start - time_curr)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        counter += 1

    time_end = time.time()
    control_freq = counter / (time_end - time_start)
    log(
        f"Control frequency: {control_freq}",
        header="".join(x.title() for x in actuator_type.split("_")),
        level="debug",
    )


class BaseController(ABC):
    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def connect_to_client(self):
        pass

    @abstractmethod
    def initialize_motors(self):
        pass

    @abstractmethod
    def set_pos(self, pos):
        pass

    @abstractmethod
    def get_motor_state(self):
        pass

    @abstractmethod
    def close_motors(self):
        pass

    # @abstractmethod
    # def set_vel(self, vel):
    #     pass
