from abc import ABC, abstractmethod
from typing import Optional, Tuple

import jax.numpy as jnp
from jax import jit  # type: ignore

from toddlerbot.sim.robot import Robot


class MotionReference(ABC):
    def __init__(self, motion_type: str, robot: Robot):
        self.motion_type = motion_type
        self.robot = robot

    def get_joint_idx(self, joint_name: str) -> int:
        return self.robot.joint_ordering.index(joint_name)

    @abstractmethod
    @jit
    def get_state(
        self,
        path_frame: jnp.ndarray,
        phase: Optional[float] = None,
        command: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray | Tuple[jnp.ndarray, float]:
        # pos: 3
        # quat: 4
        # linear_vel: 3
        # angular_vel: 3
        # joint_pos: 30
        # joint_vel: 30
        # left_contact: 1
        # right_contact: 1
        pass
