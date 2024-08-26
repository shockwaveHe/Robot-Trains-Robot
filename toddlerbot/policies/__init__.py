from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate


class BasePolicy(ABC):
    @abstractmethod
    def __init__(self, robot: Robot):
        self.robot = robot
        self.name = "base"
        self.control_dt = 6 * 0.002
        self.prep_duration = 0.0

    @abstractmethod
    def step(self, obs: Obs) -> npt.NDArray[np.float32]:
        pass

    def reset(
        self,
        time_curr: float,
        action_curr: npt.NDArray[np.float32],
        reset_action: npt.NDArray[np.float32],
        reset_duration: float,
        end_time: float = 0.0,
    ):
        reset_time = np.linspace(
            0,
            reset_duration,
            int(reset_duration / self.control_dt),
            endpoint=False,
            dtype=np.float32,
        )

        reset_pos = np.zeros((len(reset_time), action_curr.shape[0]), dtype=np.float32)
        for i, t in enumerate(reset_time):
            if t < reset_duration - end_time:
                pos = interpolate(
                    action_curr,
                    reset_action,
                    reset_duration - end_time,
                    t,
                )
            else:
                pos = reset_action

            reset_pos[i] = pos

        reset_time += time_curr + self.control_dt

        return reset_time, reset_pos
