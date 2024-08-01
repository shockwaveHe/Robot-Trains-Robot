from typing import Dict, List

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate, interpolate_arr
from toddlerbot.utils.misc_utils import set_seed


class SquatPolicy(BasePolicy):
    def __init__(self, robot: Robot):
        super().__init__(robot)
        self.name = "squat"

        set_seed(0)

        default_q = np.array(list(robot.init_joint_angles.values()), dtype=np.float32)

        warm_up_duration = 2.0
        squat_duration = 4.0
        reset_duration = 2.0
        n_trials = 2

        warm_up_action = np.array(
            list(robot.init_motor_angles.values()), dtype=np.float32
        )
        time_list: List[npt.NDArray[np.float32]] = []
        action_list: List[npt.NDArray[np.float32]] = []

        warm_up_time, warm_up_pos = self.warm_up(warm_up_action, warm_up_duration)

        time_list.append(warm_up_time)
        action_list.append(warm_up_pos)

        for i in range(n_trials):
            time = np.arange(0, squat_duration, self.control_dt, dtype=np.float32)  # type: ignore
            time += time_list[-1][-1] + self.control_dt

            pos_end = default_q.copy()
            for side in ["left", "right"]:
                knee_pos = np.random.uniform(
                    robot.joint_limits[f"{side}_knee_pitch"][0],
                    robot.joint_limits[f"{side}_knee_pitch"][1],
                )
                c = (
                    robot.data_dict["offsets"]["knee_to_ank_pitch_z"]
                    / robot.data_dict["offsets"]["hip_pitch_to_knee_z"]
                )
                ank_pitch_pos = np.arctan2(np.sin(knee_pos), np.cos(knee_pos) + c)
                hip_pitch_pos = knee_pos - ank_pitch_pos

                pos_end[robot.joint_ordering.index(f"{side}_hip_pitch")] = hip_pitch_pos
                pos_end[robot.joint_ordering.index(f"{side}_knee_pitch")] = knee_pos
                pos_end[robot.joint_ordering.index(f"{side}_ank_pitch")] = ank_pitch_pos

            joint_angles = dict(zip(robot.joint_ordering, pos_end))
            motor_angles = robot.joint_to_motor_angles(joint_angles)
            action_end = np.array(list(motor_angles.values()), dtype=np.float32)
            timed_action = np.tile(action_end.copy(), (time.shape[0], 1))
            for i, t in enumerate(time):
                action = interpolate(
                    np.zeros_like(action_end),
                    action_end,
                    squat_duration,
                    t,
                )
                timed_action[i] = action + warm_up_action

            time_list.append(time)
            action_list.append(timed_action)

            reset_time, reset_pos = self.reset(
                time[-1],
                timed_action[-1],
                warm_up_action if i < n_trials - 1 else np.zeros_like(timed_action[-1]),
                reset_duration,
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
