from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate


class BasePolicy(ABC):
    @abstractmethod
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        control_dt: float = 6 * 0.002,
        prep_duration: float = 2.0,
        n_steps_total: float = float("inf"),
    ):
        self.name = name
        self.robot = robot
        self.init_motor_pos = init_motor_pos
        self.control_dt = control_dt
        self.prep_duration = prep_duration
        self.n_steps_total = n_steps_total

    @abstractmethod
    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        pass

    def move(
        self,
        time_curr: float,
        action_curr: npt.NDArray[np.float32],
        action_next: npt.NDArray[np.float32],
        duration: float,
        end_time: float = 0.0,
    ):
        reset_time = np.linspace(
            0,
            duration,
            int(duration / self.control_dt),
            endpoint=False,
            dtype=np.float32,
        )

        reset_pos = np.zeros((len(reset_time), action_curr.shape[0]), dtype=np.float32)
        for i, t in enumerate(reset_time):
            if t < duration - end_time:
                pos = interpolate(
                    action_curr,
                    action_next,
                    duration - end_time,
                    t,
                )
            else:
                pos = action_next

            reset_pos[i] = pos

        reset_time += time_curr + self.control_dt

        return reset_time, reset_pos
