from abc import ABC, abstractmethod
from typing import Optional

from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType


class MotionReference(ABC):
    def __init__(self, name: str, motion_type: str, robot: Robot):
        self.name = name
        self.motion_type = motion_type
        self.robot = robot

    @abstractmethod
    def get_phase_signal(
        self, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        pass

    @abstractmethod
    def get_state_ref(
        self,
        path_pos: ArrayType,
        path_quat: ArrayType,
        time_curr: Optional[float | ArrayType] = None,
        command: Optional[ArrayType] = None,
    ) -> ArrayType:
        pass
