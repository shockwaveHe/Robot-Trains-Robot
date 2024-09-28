import time

import joblib

# import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pynput import keyboard

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot

"""
Replay Policy loads the dataset logged and follows the same trajectory
"""

default_pose = np.array(
    [
        -0.60745645,
        -0.9265244,
        0.02147579,
        1.2348546,
        0.52922344,
        0.49394178,
        -1.125942,
        0.5123496,
        -0.96180606,
        -0.25003886,
        1.2195148,
        -0.35128164,
        -0.6504078,
        -1.1535536,
    ]
)


class ReplayFollowerPolicy(BasePolicy):
    def __init__(self, robot: Robot, log_path: str, replay_dest: str):
        super().__init__(name="replay_fixed", robot=robot, init_motor_pos=default_pose)

        self.log_path = log_path
        self.toggle_motor = False
        self.data_dict = joblib.load(log_path)
        self.data_dict["state_array"][:, 0] = (
            self.data_dict["state_array"][:, 0] - self.data_dict["state_array"][0, 0]
        )
        print(self.data_dict.keys())
        self.replay_start = 0

        self.log = False
        self.replay_done = False
        self.blend_percentage = 0.0
        self.default_pose = default_pose
        # Start a listener for the spacebar
        self._start_keyboard_listener()

    def _start_keyboard_listener(self):
        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    # if space is pressed, and is done playing, play next trajectory
                    # print(
                    #     f"\n\nLogging is now {'enabled' if self.log else 'disabled'}.\n\n"
                    # )
                    pass
            except AttributeError:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    def reset_slowly(self, obs_real):
        leader_action = (
            self.default_pose * self.blend_percentage
            + obs_real.motor_pos * (1 - self.blend_percentage)
        )
        self.blend_percentage += 0.002
        self.blend_percentage = min(1, self.blend_percentage)
        return leader_action

    def step(self, obs: Obs, obs_real: Obs) -> npt.NDArray[np.float32]:
        if self.replay_start < 1:
            self.replay_start = time.time()

        curr_idx = np.argmin(
            np.abs(
                self.data_dict["state_array"][:, 0] - (time.time() - self.replay_start)
            )
        )
        sim_action = self.data_dict["state_array"][curr_idx, 1:15]

        if (time.time() - self.replay_start) > self.data_dict["state_array"][-1, 0]:
            print("Replay done")
            self.replay_done = True

        if self.replay_done:
            leader_action = self.reset_slowly(obs_real)
            if self.blend_percentage >= 0.99:
                self.log = True
                self.toggle_motor = True
        else:
            leader_action = sim_action
        return sim_action, leader_action
