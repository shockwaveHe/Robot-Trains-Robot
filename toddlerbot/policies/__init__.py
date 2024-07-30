from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import numpy.typing as npt

from toddlerbot.sim.robot import Robot


class BasePolicy(ABC):
    @abstractmethod
    def __init__(self, robot: Robot):
        self.robot = robot
        self.name = "base"
        self.control_dt = 0.01

    @abstractmethod
    def run(
        self,
        obs_dict: Dict[str, npt.NDArray[np.float32]],
        last_action: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        pass
