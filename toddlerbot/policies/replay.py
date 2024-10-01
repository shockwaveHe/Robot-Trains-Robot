import os
import time
from typing import Dict, List, Optional

import joblib
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
        read_dataset: bool = True,
    ):
        super().__init__(name, robot, init_motor_pos)
        # Use glob to find all pickle files matching the pattern
        if read_dataset:
            data_file_path = os.path.join("results", run_name, "dataset.lz4")
        else:
            data_file_path = os.path.join("results", run_name, "log_data.pkl")
        if not os.path.exists(data_file_path):
            raise ValueError("No data files found")

        with open(data_file_path, "rb") as f:
            data_dict = joblib.load(f)

        if read_dataset:
            data_dict = self.convert_dataset(data_dict)

        motor_angles_list: List[Dict[str, float]] = data_dict["motor_angles_list"]
        self.t_action = np.array(
            [data_dict["obs_list"][i].time for i in range(len(motor_angles_list))]
        )
        self.t_action = self.t_action - self.t_action[0]  # make sure it starts from 0

        # reset motors to initial position
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
        self.replay_done = False
        self.init_motor_pos = init_motor_pos
        self.reset_time = None

    def convert_dataset(self, data_dict: Dict):
        # convert the dataset to the correct format
        # dataset is assumed to be logged on toddlerbot_arms

        converted_dict = {"obs_list": [], "motor_angles_list": []}
        for i in range(data_dict["state_array"].shape[0]):
            motor_angles = {}
            obs = Obs(
                time=data_dict["state_array"][i, 0],
                motor_pos=np.zeros(14),
                motor_vel=np.zeros(14),
                motor_tor=np.zeros(14),
            )
            for j, jname in enumerate(self.robot.joint_ordering):
                motor_angles[jname] = data_dict["state_array"][i, j + 1]
            converted_dict["obs_list"].append(obs)
            converted_dict["motor_angles_list"].append(motor_angles)
        return converted_dict

    def step(
        self,
        obs: Obs,
        is_real: bool = False,
        control_inputs: Optional[Dict[str, float]] = None,
    ) -> npt.NDArray[np.float32]:
        # action = self.action_arr[self.step_curr]
        # self.step_curr = self.step_curr + 1

        if self.replay_start_time < 1:
            self.replay_start_time = time.time()

        curr_idx = np.argmin(np.abs(self.t_action - (time.time() - self.replay_start)))
        action = self.action_arr[curr_idx]

        if (time.time() - self.replay_start) > len(self.t_action):
            print("Replay done")
            self.replay_done = True

        if self.replay_done:
            if self.reset_time is None:
                self.reset_time, self.reset_action = self.move(
                    -self.control_dt,
                    self.init_motor_pos,
                    obs.motor_pos,
                    self.prep_duration,
                    end_time=5.0,
                )
                self.reset_idx = 0
            action = self.reset_action[self.reset_idx]
            self.reset_idx = min(self.reset_idx + 1, len(self.reset_action) - 1)

        return action
