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

        time_list: List[npt.NDArray[np.float32]] = []
        action_list: List[npt.NDArray[np.float32]] = []
        for joint_name in ["waist_roll", "waist_yaw"]:
            lower_limit = robot.config["joints"][joint_name]["lower_limit"]
            upper_limit = robot.config["joints"][joint_name]["upper_limit"]
            joint_idx = robot.joint_ordering.index(joint_name)

            mean = (lower_limit + upper_limit) / 2
            amplitude_max = upper_limit - mean

            for _ in range(3):
                sine_signal_config = get_random_sine_signal_config(
                    3.0, self.control_dt, 0.0, [0.5, 2], [np.pi / 12, amplitude_max]
                )
                t, signal = get_sine_signal(sine_signal_config)
                if len(time_list) > 0:
                    t += time_list[-1][-1] + self.control_dt

                timed_pos = np.tile(default_q.copy(), (signal.shape[0], 1))
                timed_pos[:, joint_idx] = signal
                timed_action = np.zeros_like(timed_pos)
                for i, pos in enumerate(timed_pos):
                    joint_angles = dict(zip(robot.joint_ordering, pos))
                    motor_angles = robot.joint_to_motor_angles(joint_angles)
                    timed_action[i] = np.array(
                        list(motor_angles.values()), dtype=np.float32
                    )

                time_list.append(t)
                action_list.append(timed_action)

        self.time_arr = np.concatenate(time_list)
        self.action_arr = np.concatenate(action_list)

    def run(
        self,
        obs_dict: Dict[str, npt.NDArray[np.float32]],
        last_action: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        return interpolate_arr(obs_dict["time"].item(), self.time_arr, self.action_arr)  # type: ignore
