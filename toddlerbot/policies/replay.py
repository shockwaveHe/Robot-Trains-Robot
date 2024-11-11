import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate_action


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
        dataset_file_path = os.path.join("results", run_name, "dataset.lz4")
        picklet_file_path = os.path.join("results", run_name, "log_data.pkl")

        if os.path.exists(dataset_file_path):
            data_dict = self.convert_dataset(joblib.load(dataset_file_path))
        elif os.path.exists(picklet_file_path):
            data_dict = joblib.load(picklet_file_path)
        else:
            raise ValueError(
                f"No data files found in {dataset_file_path} or {picklet_file_path}"
            )

        motor_angles_list: List[Dict[str, float]] = data_dict["motor_angles_list"]
        self.time_arr = np.array(
            [data_dict["obs_list"][i].time for i in range(len(motor_angles_list))]
        )
        self.time_arr = self.time_arr - self.time_arr[0]  # make sure it starts from 0

        self.action_arr = np.array(
            [list(motor_angles.values()) for motor_angles in motor_angles_list],
            dtype=np.float32,
        )

        start_idx = 0
        for idx in range(len(self.action_arr)):
            if np.allclose(self.default_motor_pos, self.action_arr[idx], atol=1e-2):
                start_idx = idx
                print(f"Truncating dataset at index {start_idx}")
                break

        self.action_arr = self.action_arr[start_idx:]

        # reset motors to initial position
        self.prep_duration = 2.0
        self.prep_time, self.prep_action = self.move(
            -self.control_dt, init_motor_pos, self.action_arr[0], self.prep_duration
        )

    def convert_dataset(self, data_dict: Dict):
        # convert the dataset to the correct format
        # dataset is assumed to be logged on toddlerbot_arms

        converted_dict: Dict[str, List] = {"obs_list": [], "motor_angles_list": []}
        for i in range(data_dict["state_array"].shape[0]):
            motor_angles = {}
            obs = Obs(
                time=data_dict["state_array"][i, 0],
                motor_pos=np.zeros(14, dtype=np.float32),
                motor_vel=np.zeros(14, dtype=np.float32),
                motor_tor=np.zeros(14, dtype=np.float32),
            )
            for j, jname in enumerate(self.robot.joint_ordering):
                motor_angles[jname] = data_dict["state_array"][i, j + 1]

            converted_dict["obs_list"].append(obs)
            converted_dict["motor_angles_list"].append(motor_angles)

        return converted_dict

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )

        else:
            curr_idx = np.argmin(np.abs(self.time_arr - obs.time + self.prep_duration))
            action = self.action_arr[curr_idx]

        return self.zero_command, action
