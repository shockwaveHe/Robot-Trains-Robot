from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import numpy.typing as npt

from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate


class BasePolicy(ABC):
    @abstractmethod
    def __init__(self, robot: Robot):
        self.robot = robot
        self.name = "base"
        self.control_dt = 0.01

    @abstractmethod
    def run(
        self, obs_dict: Dict[str, npt.NDArray[np.float32]]
    ) -> npt.NDArray[np.float32]:
        pass

    def warm_up(
        self,
        warm_up_duration: float,
    ):
        warm_up_time = np.linspace(
            0, warm_up_duration, int(warm_up_duration / self.control_dt), endpoint=False
        )

        warm_up_pos = np.zeros(
            (len(warm_up_time), self.robot.action_dim), dtype=np.float32
        )

        return warm_up_time, warm_up_pos

    def reset(
        self,
        time_curr: float,
        action_curr: npt.NDArray[np.float32],
        reset_duration: float,
    ):
        reset_time = np.linspace(
            0, reset_duration, int(reset_duration / self.control_dt), endpoint=False
        )

        reset_pos = np.zeros((len(reset_time), action_curr.shape[0]), dtype=np.float32)
        for i, t in enumerate(reset_time):
            if t < reset_duration - 0.5:
                pos = interpolate(
                    action_curr,
                    np.zeros_like(action_curr),
                    reset_duration - 0.5,
                    t,
                )
            else:
                pos = np.zeros_like(action_curr)

            reset_pos[i] = pos

        reset_time += time_curr + self.control_dt

        return reset_time, reset_pos
