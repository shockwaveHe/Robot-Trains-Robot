import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.keyboard import Keyboard
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

        if "push_up" in run_name:
            motion_file_path = os.path.join("toddlerbot", "motion", f"{run_name}.pkl")
            if os.path.exists(motion_file_path):
                data_dict = joblib.load(motion_file_path)
            else:
                raise ValueError(f"No data files found in {motion_file_path}")

            self.time_arr = np.array(data_dict["time"])
            self.action_arr = np.array(data_dict["action_traj"], dtype=np.float32)
        else:
            # Use glob to find all pickle files matching the pattern
            dataset_file_path = os.path.join("results", run_name, "toddlerbot_0.lz4")
            pickle_file_path = os.path.join("results", run_name, "log_data.pkl")

            if os.path.exists(dataset_file_path):
                data_dict = self.convert_dataset(joblib.load(dataset_file_path))
            elif os.path.exists(pickle_file_path):
                data_dict = joblib.load(pickle_file_path)
            else:
                raise ValueError(
                    f"No data files found in {dataset_file_path} or {pickle_file_path}"
                )

            motor_angles_list: List[Dict[str, float]] = data_dict["motor_angles_list"]
            self.time_arr = np.array(
                [data_dict["obs_list"][i].time for i in range(len(motor_angles_list))]
            )
            self.time_arr = self.time_arr - self.time_arr[0]
            self.action_arr = np.array(
                [list(motor_angles.values()) for motor_angles in motor_angles_list],
                dtype=np.float32,
            )

        start_idx = 0
        for idx in range(len(self.action_arr)):
            if np.allclose(self.default_motor_pos, self.action_arr[idx], atol=1e-1):
                start_idx = idx
                print(f"Truncating dataset at index {start_idx}")
                break

        self.time_arr = self.time_arr[start_idx:]
        self.action_arr = self.action_arr[start_idx:]

        self.step_curr = 0
        self.keyframes: List[npt.NDArray[np.float32]] = []
        self.keyframe_saved = False
        self.is_prepared = False

        self.keyboard = None
        try:
            self.keyboard = Keyboard()

            def save(action: npt.NDArray[np.float32]):
                self.keyframes.append(action)
                print(f"Keyframe added at step {self.step_curr}")

            self.keyboard.register("save", save)

        except Exception:
            print("Keyboard is not available")

    def convert_dataset(self, data_dict: Dict):
        # convert the dataset to the correct format
        # dataset is assumed to be logged on toddlerbot_arms
        converted_dict: Dict[str, List] = {"obs_list": [], "motor_angles_list": []}
        for i in range(data_dict["time"].shape[0]):
            obs = Obs(
                time=data_dict["time"][i],
                motor_pos=np.zeros(14, dtype=np.float32),
                motor_vel=np.zeros(14, dtype=np.float32),
                motor_tor=np.zeros(14, dtype=np.float32),
            )
            motor_angles = dict(zip(self.robot.motor_ordering, data_dict["motor_pos"][i]))

            converted_dict["obs_list"].append(obs)
            converted_dict["motor_angles_list"].append(motor_angles)

        return converted_dict

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        if not self.is_prepared:
            self.is_prepared = True
            self.prep_duration = 7.0 if is_real else 2.0
            self.prep_time, self.prep_action = self.move(
                -self.control_dt,
                self.init_motor_pos,
                self.action_arr[0],
                self.prep_duration,
                end_time=5.0 if is_real else 0.0,
            )

        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return {}, action

        curr_idx = np.argmin(np.abs(self.time_arr - obs.time + self.prep_duration))
        action = self.action_arr[curr_idx]

        if self.keyboard is not None:
            key_inputs = self.keyboard.get_keyboard_input()
            for key in key_inputs:
                self.keyboard.check(key, action=action)

        self.step_curr += 1

        return {}, action
