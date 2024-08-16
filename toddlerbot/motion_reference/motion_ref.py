from abc import ABC, abstractmethod
from typing import Optional

import jax

from toddlerbot.sim.robot import Robot


class MotionReference(ABC):
    def __init__(self, name: str, motion_type: str, robot: Robot, use_jax: bool):
        self.name = name
        self.motion_type = motion_type
        self.robot = robot
        self.use_jax = use_jax

    def get_joint_idx(self, joint_name: str) -> int:
        return self.robot.joint_ordering.index(joint_name)

    @abstractmethod
    def get_state_ref(
        self,
        path_pos: jax.Array,
        path_quat: jax.Array,
        phase: Optional[float | jax.Array] = None,
        command: Optional[jax.Array] = None,
    ) -> jax.Array:
        pass
