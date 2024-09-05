import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate_action


class StandPolicy(BasePolicy):
    def __init__(self, robot: Robot, init_motor_pos: npt.NDArray[np.float32]):
        super().__init__(robot, init_motor_pos)
        self.name = "stand"

        self.default_motor_pos = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )

        self.prep_duration = 2.0
        self.prep_time, self.prep_action = self.move(
            -self.control_dt, init_motor_pos, self.default_motor_pos, self.prep_duration
        )

    def step(self, obs: Obs) -> npt.NDArray[np.float32]:
        if obs.time < self.prep_time[-1]:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return action

        return self.default_motor_pos
