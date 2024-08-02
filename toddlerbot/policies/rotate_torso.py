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

        prep_duration = 20.0
        warm_up_duration = 2.0
        sine_duraion = 6.0
        reset_duration = 2.0
        n_trials = 1

        time_list: List[npt.NDArray[np.float32]] = []
        action_list: List[npt.NDArray[np.float32]] = []

        prep_act = np.array(list(robot.init_motor_angles.values()), dtype=np.float32)
        prep_time, prep_action = self.warm_up(prep_act, prep_duration)

        time_list.append(prep_time)
        action_list.append(prep_action)

        warm_up_act = np.array(list(robot.init_motor_angles.values()), dtype=np.float32)
        warm_up_act[robot.motor_ordering.index("left_sho_roll")] = -np.pi / 12
        warm_up_act[robot.motor_ordering.index("right_sho_roll")] = -np.pi / 12
        warm_up_time, warm_up_action = self.warm_up(warm_up_act, warm_up_duration)
        warm_up_time += time_list[-1][-1] + self.control_dt

        time_list.append(warm_up_time)
        action_list.append(warm_up_action)

        for joint_name in ["waist_yaw", "waist_roll"]:
            joint_idx = robot.joint_ordering.index(joint_name)

            mean = (
                robot.joint_limits[joint_name][0] + robot.joint_limits[joint_name][1]
            ) / 2
            amplitude_min = np.pi / 12
            amplitude_max = robot.joint_limits[joint_name][1] - mean

            if joint_name == "waist_yaw":
                frequency_range = [0.2, 0.5]
            else:
                frequency_range = [0.2, 0.5]
                amplitude_min = np.pi / 24
                amplitude_max: float = np.pi / 12

            for i in range(n_trials):
                sine_signal_config = get_random_sine_signal_config(
                    sine_duraion,
                    self.control_dt,
                    mean,
                    frequency_range,
                    [amplitude_min, amplitude_max],
                )
                rotate_time, signal = get_sine_signal(sine_signal_config)
                if len(time_list) > 0:
                    rotate_time += time_list[-1][-1] + self.control_dt

                rotate_pos = np.tile(default_q.copy(), (signal.shape[0], 1))
                rotate_pos[:, joint_idx] = signal
                rotate_action = np.zeros_like(rotate_pos)
                for j, pos in enumerate(rotate_pos):
                    joint_angles = dict(zip(robot.joint_ordering, pos))
                    motor_angles = robot.joint_to_motor_angles(joint_angles)
                    sine_action = np.array(
                        list(motor_angles.values()), dtype=np.float32
                    )
                    rotate_action[j] = sine_action + warm_up_act

                time_list.append(rotate_time)
                action_list.append(rotate_action)

                if i == n_trials - 1:
                    if joint_name == "waist_roll":
                        reset_pos = default_q.copy()
                        reset_pos[robot.joint_ordering.index("waist_roll")] = (
                            -np.pi / 36
                        )
                        motor_angles = robot.joint_to_motor_angles(
                            dict(zip(robot.joint_ordering, reset_pos))
                        )
                        reset_act = np.array(
                            list(motor_angles.values()), dtype=np.float32
                        )
                    else:
                        reset_act = np.zeros_like(warm_up_act)
                else:
                    reset_act = warm_up_act.copy()

                reset_time, reset_action = self.reset(
                    time_list[-1][-1],
                    action_list[-1][-1],
                    reset_act,
                    reset_duration,
                )

                time_list.append(reset_time)
                action_list.append(reset_action)

        self.time_arr = np.concatenate(time_list)
        self.action_arr = np.concatenate(action_list)

    def run(
        self, obs_dict: Dict[str, npt.NDArray[np.float32]]
    ) -> npt.NDArray[np.float32]:
        time_curr = obs_dict["time"].item()
        return np.array(interpolate_arr(time_curr, self.time_arr, self.action_arr))
