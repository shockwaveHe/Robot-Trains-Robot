from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class JointState:
    time: float
    pos: float
    vel: float = 0.0


class BaseController(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def connect_to_client(self):
        pass

    @abstractmethod
    def initialize_motors(self):
        pass

    @abstractmethod
    def set_pos(self, pos: List[float]):
        pass

    @abstractmethod
    def get_motor_state(self) -> Dict[int, JointState]:
        pass

    @abstractmethod
    def close_motors(self):
        pass
