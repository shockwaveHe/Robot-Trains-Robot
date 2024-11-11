from typing import Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate_action


class RecordPolicy(BasePolicy, policy_name="record"):
    def __init__(
        self, name: str, robot: Robot, init_motor_pos: npt.NDArray[np.float32]
    ):
        super().__init__(name, robot, init_motor_pos)

        self.disable_motor_indices = np.concatenate(
            [self.leg_motor_indices, np.array([16, 23])]
        )

        self.is_prepared = False
        self.is_running = False
        self.toggle_motor = False

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        if not self.is_prepared:
            self.is_prepared = True
            self.prep_duration = 7.0 if is_real else 2.0
            self.prep_time, self.prep_action = self.move(
                -self.control_dt,
                self.init_motor_pos,
                self.default_motor_pos,
                self.prep_duration,
                end_time=5.0 if is_real else 0.0,
            )

        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return self.zero_command, action

        if not self.is_running:
            self.is_running = True
            self.toggle_motor = True

        action = self.default_motor_pos.copy()
        action[self.disable_motor_indices] = obs.motor_pos[self.disable_motor_indices]

        return self.zero_command, action
