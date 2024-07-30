from typing import Dict, List

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import (
    get_random_sine_signal_config,
    get_sine_signal,
    interpolate_arr,
)
from toddlerbot.utils.misc_utils import set_seed


class RotateTorsoPolicy(BasePolicy):
    def __init__(self, robot: Robot):
        super().__init__(robot)
        self.name = "rotate_torso"

        set_seed(0)

        default_q = np.array(list(robot.init_joint_angles.values()), dtype=np.float32)

        warm_up_duration = 3.0
        sine_duraion = 3.0
        reset_duration = 2.0
        n_sine_signal = 2
        frequency_range = [0.2, 0.5]
        amplitude_min = np.pi / 12

        time_list: List[npt.NDArray[np.float32]] = []
        action_list: List[npt.NDArray[np.float32]] = []

        warm_up_time, warm_up_pos = self.warm_up(warm_up_duration)

        time_list.append(warm_up_time)
        action_list.append(warm_up_pos)

        for joint_name in ["waist_roll", "waist_yaw"]:
            joint_idx = robot.joint_ordering.index(joint_name)

            mean = (
                robot.joint_limits[joint_name][0] + robot.joint_limits[joint_name][1]
            ) / 2
            amplitude_max = robot.joint_limits[joint_name][1] - mean

            for _ in range(n_sine_signal):
                sine_signal_config = get_random_sine_signal_config(
                    sine_duraion,
                    self.control_dt,
                    mean,
                    frequency_range,
                    [amplitude_min, amplitude_max],
                )
                time, signal = get_sine_signal(sine_signal_config)
                if len(time_list) > 0:
                    time += time_list[-1][-1] + self.control_dt

                timed_pos = np.tile(default_q.copy(), (signal.shape[0], 1))
                timed_pos[:, joint_idx] = signal
                timed_action = np.zeros_like(timed_pos)
                for i, pos in enumerate(timed_pos):
                    joint_angles = dict(zip(robot.joint_ordering, pos))
                    motor_angles = robot.joint_to_motor_angles(joint_angles)
                    timed_action[i] = np.array(
                        list(motor_angles.values()), dtype=np.float32
                    )

                time_list.append(time)
                action_list.append(timed_action)

                reset_time, reset_pos = self.reset(
                    time[-1], timed_action[-1], reset_duration
                )

                time_list.append(reset_time)
                action_list.append(reset_pos)

        self.time_arr = np.concatenate(time_list)
        self.action_arr = np.concatenate(action_list)

    def run(
        self, obs_dict: Dict[str, npt.NDArray[np.float32]]
    ) -> npt.NDArray[np.float32]:
        time_curr = obs_dict["time"].item()
        return np.array(interpolate_arr(time_curr, self.time_arr, self.action_arr))
