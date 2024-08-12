from abc import ABC, abstractmethod
from typing import Optional

import jax
import numpy as np
import numpy.typing as npt

from toddlerbot.sim.robot import Robot


class MotionReference(ABC):
    def __init__(self, motion_type: str, robot: Robot, use_jax: bool):
        self.motion_type = motion_type
        self.robot = robot
        self.use_jax = use_jax

    def get_joint_idx(self, joint_name: str) -> int:
        return self.robot.joint_ordering.index(joint_name)

    @abstractmethod
    def get_state_ref(
        self,
        path_pos: npt.NDArray[np.float32] | jax.Array,
        path_quat: npt.NDArray[np.float32] | jax.Array,
        phase: Optional[float | npt.NDArray[np.float32] | jax.Array] = None,
        command: Optional[npt.NDArray[np.float32] | jax.Array] = None,
    ) -> npt.NDArray[np.float32] | jax.Array:
        pass
