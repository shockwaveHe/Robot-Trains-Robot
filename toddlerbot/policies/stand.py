import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot


class StandPolicy(BasePolicy):
    def __init__(self, robot: Robot):
        super().__init__(robot)
        self.name = "stand"

        self.default_action = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )

    def run(self, obs: Obs) -> npt.NDArray[np.float32]:
        return self.default_action
