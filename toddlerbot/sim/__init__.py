from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import numpy as np
import numpy.typing as npt


@dataclass
class Obs:
    time: float
    u: npt.NDArray[np.float32]
    q: npt.NDArray[np.float32]
    dq: npt.NDArray[np.float32]
    lin_vel: npt.NDArray[np.float32] = np.zeros(3, dtype=np.float32)
    ang_vel: npt.NDArray[np.float32] = np.zeros(3, dtype=np.float32)
    euler: npt.NDArray[np.float32] = np.zeros(3, dtype=np.float32)


class BaseSim(ABC):
    @abstractmethod
    def __init__(self, name: str):
        self.name = name
        self.visualizer = None

    @abstractmethod
    def set_motor_angles(self, motor_angles: Dict[str, float]):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def get_observation(self) -> Obs:
        pass

    @abstractmethod
    def close(self):
        pass
