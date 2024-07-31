from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import numpy.typing as npt

from toddlerbot.actuation import JointState


class BaseSim(ABC):
    @abstractmethod
    def __init__(self):
        self.name = "base"
        self.dt = 0.001
        self.start_time = 0.0
        self.visualizer = None

    @abstractmethod
    def get_joint_state(self) -> Dict[str, JointState]:
        pass

    @abstractmethod
    def set_motor_angles(self, motor_angles: Dict[str, float]):
        pass

    @abstractmethod
    def get_observation(
        self,
    ) -> Dict[str, npt.NDArray[np.float32]]:
        pass

    @abstractmethod
    def close(self):
        pass
