from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import numpy.typing as npt

from toddlerbot.actuation import JointState


class BaseSim(ABC):
    @abstractmethod
    def __init__(self):
        self.name = "base"
        self.dt = 0.001
        self.visualizer = None

    @abstractmethod
    def get_joint_state(self) -> Dict[str, JointState]:
        pass

    @abstractmethod
    def set_joint_angles(self, joint_angles: Dict[str, float]):
        pass

    @abstractmethod
    def get_observation(
        self,
    ) -> Dict[str, npt.NDArray[np.float32]]:
        pass

    def rollout(self, joint_angles_list: List[Dict[str, float]]):
        joint_state_list: List[Dict[str, JointState]] = []
        return joint_state_list

    @abstractmethod
    def close(self):
        pass
