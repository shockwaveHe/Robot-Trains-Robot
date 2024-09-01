from abc import ABC, abstractmethod
from typing import Optional

from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType


class MotionReference(ABC):
    def __init__(self, name: str, motion_type: str, robot: Robot):
        self.name = name
        self.motion_type = motion_type
        self.robot = robot

    def get_joint_idx(self, joint_name: str) -> int:
        return self.robot.joint_ordering.index(joint_name)

    @abstractmethod
    def get_state_ref(
        self,
        path_pos: ArrayType,
        path_quat: ArrayType,
        phase: Optional[float | ArrayType] = None,
        command: Optional[ArrayType] = None,
        duration: Optional[float] = None,
    ) -> ArrayType:
        pass
