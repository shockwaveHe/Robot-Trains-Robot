import os
import pickle
from typing import Dict, List

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot


class ReplayPolicy(BasePolicy, policy_name="replay"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        run_name: str,
    ):
        super().__init__(name, robot, init_motor_pos)
        # Use glob to find all pickle files matching the pattern
        pickle_file_path = os.path.join("results", run_name, "log_data.pkl")
        if not os.path.exists(pickle_file_path):
            raise ValueError("No data files found")

        with open(pickle_file_path, "rb") as f:
            data_dict = pickle.load(f)

        motor_angles_list: List[Dict[str, float]] = data_dict["motor_angles_list"]

        self.prep_duration = 7.0
        self.prep_time, self.prep_action = self.move(
            -self.control_dt,
            init_motor_pos,
            np.zeros_like(init_motor_pos),
            self.prep_duration,
            end_time=5.0,
        )

        replay_action = np.array(
            [list(motor_angles.values()) for motor_angles in motor_angles_list],
            dtype=np.float32,
        )
        self.action_arr = np.concatenate([self.prep_action, replay_action])

        self.n_steps_total = self.action_arr.shape[0]
        self.step_curr = 0

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        action = self.action_arr[self.step_curr]
        self.step_curr = self.step_curr + 1
        return action
