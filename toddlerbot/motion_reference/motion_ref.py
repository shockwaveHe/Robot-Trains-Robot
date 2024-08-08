from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.sim.robot import Robot


class MotionReference(ABC):
    def __init__(self, motion_type: str, robot: Robot):
        self.motion_type = motion_type
        self.robot = robot

    def get_joint_idx(self, joint_name: str) -> int:
        return self.robot.joint_ordering.index(joint_name)

    @abstractmethod
    def get_state(
        self,
        path_frame: npt.NDArray[np.float32],
        phase: Optional[float] = None,
        command: Optional[npt.NDArray[np.float32]] = None,
    ) -> npt.NDArray[np.float32] | Tuple[npt.NDArray[np.float32], float]:
        # pos: 3
        # quat: 4
        # linear_vel: 3
        # angular_vel: 3
        # joint_pos: 30
        # joint_vel: 30
        # left_contact: 1
        # right_contact: 1
        pass
