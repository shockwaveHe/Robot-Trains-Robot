from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import numpy.typing as npt


class BasePolicy(ABC):
    @abstractmethod
    def __init__(self):
        self.name = "base"

    @abstractmethod
    def run(
        self,
        obs_dict: Dict[str, npt.NDArray[np.float32]],
        last_action: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        pass
