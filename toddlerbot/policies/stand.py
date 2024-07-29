from typing import Dict

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy


class StandPolicy(BasePolicy):
    def __init__(self):
        super().__init__()
        self.name = "stand"

    def run(
        self,
        obs_dict: Dict[str, npt.NDArray[np.float32]],
        last_action: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        return np.zeros_like(last_action)
