from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt


class RefMotionGenerator(ABC):
    def __init__(self, motion_type: str, init_state: npt.NDArray[np.float32]):
        self.motion_type = motion_type
        self.init_state = init_state

    @abstractmethod
    def get_state(
        self,
        path_frame: npt.NDArray[np.float32],
        phase: Optional[npt.NDArray[np.float32]] = None,
        command: Optional[npt.NDArray[np.float32]] = None,
    ) -> (
        npt.NDArray[np.float32]
        | Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
    ):
        # pos: 3
        # quat: 4
        # linear_vel: 3
        # angular_vel: 3
        # joint_pos: 30
        # joint_vel: 30
        # left_contact: 1
        # right_contact: 1
        pass
