import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate_action


class StandPolicy(BasePolicy):
    def __init__(self, robot: Robot, init_joint_pos: npt.NDArray[np.float32]):
        super().__init__(robot)
        self.name = "stand"

        self.default_action = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )

        self.prep_duration = 2.0
        init_action = np.array(
            list(
                robot.joint_to_motor_angles(
                    dict(zip(robot.joint_ordering, init_joint_pos))
                ).values()
            ),
            dtype=np.float32,
        )
        self.prep_time, self.prep_action = self.reset(
            -self.control_dt, init_action, self.default_action, self.prep_duration
        )

    def step(self, obs: Obs) -> npt.NDArray[np.float32]:
        if obs.time < self.prep_time[-1]:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )

            return action

        return self.default_action
