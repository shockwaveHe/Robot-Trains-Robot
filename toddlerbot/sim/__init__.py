from abc import ABC, abstractmethod
from typing import Dict

from toddlerbot.actuation import JointState


class BaseSim(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_joint_state(self) -> Dict[str, JointState]:
        pass

    @abstractmethod
    def set_joint_angles(self, joint_angles: Dict[str, float]):
        pass

    @abstractmethod
    def close(self):
        pass
