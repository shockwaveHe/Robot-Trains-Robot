from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt


@dataclass
class Obs:
    time: float
    motor_pos: npt.NDArray[np.float32]
    motor_vel: npt.NDArray[np.float32]
    motor_torque: npt.NDArray[np.float32]
    # lin_vel: npt.NDArray[np.float32] = np.zeros(3, dtype=np.float32)
    ang_vel: npt.NDArray[np.float32] = np.zeros(3, dtype=np.float32)
    euler: npt.NDArray[np.float32] = np.zeros(3, dtype=np.float32)
    joint_pos: Optional[npt.NDArray[np.float32]] = None
    joint_vel: Optional[npt.NDArray[np.float32]] = None


class BaseSim(ABC):
    @abstractmethod
    def __init__(self, name: str):
        self.name = name
        self.visualizer = None

    @abstractmethod
    def set_motor_angles(self, motor_angles: Dict[str, float]):
        pass

    @abstractmethod
    def set_motor_kps(self, motor_kps: Dict[str, float]):
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
