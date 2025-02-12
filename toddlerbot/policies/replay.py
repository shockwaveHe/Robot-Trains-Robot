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

# This script replays a keyframe animation or recorded motion data.


class ReplayPolicy(BasePolicy, policy_name="replay"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        run_name: str,
    ):
        """Initializes the class with motion data and configuration.

        Args:
            name (str): The name of the instance.
            robot (Robot): The robot object associated with this instance.
            init_motor_pos (npt.NDArray[np.float32]): Initial motor positions.
            run_name (str): The name of the run, used to determine the motion file path.

        Raises:
            ValueError: If no data files are found for the specified run name.

        This constructor loads motion data from a specified file based on the `run_name`. If the run name includes "cuddle" or "push_up", it loads a motion file. Otherwise, it attempts to load data from a dataset or pickle file. The method also initializes various attributes related to motion timing and actions, and sets up a keyboard listener for saving keyframes.
        """
        super().__init__(name, robot, init_motor_pos)

        motion_file_path = os.path.join("motion", f"{run_name}.pkl")
        if os.path.exists(motion_file_path):
            data_dict = joblib.load(motion_file_path)

            self.time_arr = np.array(data_dict["time"])
            self.action_arr = np.array(data_dict["action_traj"], dtype=np.float32)

            if robot.has_gripper and self.action_arr.shape[1] < len(
                robot.motor_ordering
            ):
                self.action_arr = np.concatenate(
                    [self.action_arr, np.zeros((self.action_arr.shape[0], 2))], axis=1
                )

            if not robot.has_gripper and self.action_arr.shape[1] > len(
                robot.motor_ordering
            ):
                self.action_arr = self.action_arr[:, :-2]
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
        self.is_done = False
        self.time_start = self.prep_duration

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
        """Converts a dataset to the required format for processing with toddlerbot_arms.

        This function processes the input dataset dictionary by iterating over each time step,
        creating an observation object with initialized motor position, velocity, and torque arrays,
        and mapping motor positions to their respective motor names. The converted data is stored
        in a dictionary with lists of observations and motor angles.

        Args:
            data_dict (Dict): A dictionary containing the dataset with keys such as "time" and "motor_pos".

        Returns:
            Dict[str, List]: A dictionary with two keys: "obs_list" containing observation objects and
            "motor_angles_list" containing dictionaries of motor angles.
        """
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
            motor_angles = dict(
                zip(self.robot.motor_ordering, data_dict["motor_pos"][i])
            )

            converted_dict["obs_list"].append(obs)
            converted_dict["motor_angles_list"].append(motor_angles)

        return converted_dict

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Executes a single step in the simulation or real environment, returning the current action.

        This function determines the appropriate action to take based on the current observation and whether the environment is real or simulated. It handles the preparation phase if necessary and updates the action based on the current time and keyboard inputs.

        Args:
            obs (Obs): The current observation containing the time and other relevant data.
            is_real (bool, optional): Indicates if the environment is real. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing an empty dictionary and the action array for the current step.
        """
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

        curr_idx = np.argmin(np.abs(self.time_arr - obs.time + self.time_start))
        action = self.action_arr[curr_idx]

        if curr_idx == len(self.action_arr) - 1:
            self.is_done = True

        if self.keyboard is not None:
            key_inputs = self.keyboard.get_keyboard_input()
            for key in key_inputs:
                self.keyboard.check(key, action=action)

        self.step_curr += 1

        return {}, action
