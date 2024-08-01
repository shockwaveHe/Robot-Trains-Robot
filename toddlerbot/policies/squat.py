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
        squat_duration = 3.0
        pause_duration = 1.0
        reset_duration = 3.0
        n_trials = 3

        warm_up_action = np.array(
            list(robot.init_motor_angles.values()), dtype=np.float32
        )
        warm_up_action[robot.motor_ordering.index("left_sho_yaw_drive")] = -np.pi / 2
        warm_up_action[robot.motor_ordering.index("right_sho_yaw_drive")] = np.pi / 2

        time_list: List[npt.NDArray[np.float32]] = []
        action_list: List[npt.NDArray[np.float32]] = []

        warm_up_time, warm_up_pos = self.warm_up(warm_up_action, warm_up_duration)

        time_list.append(warm_up_time)
        action_list.append(warm_up_pos)

        for i in range(n_trials):
            pos_end = default_q.copy()
            knee_angle = abs(
                np.random.uniform(
                    robot.joint_limits["left_knee_pitch"][0],
                    robot.joint_limits["left_knee_pitch"][1],
                )
            )
            c = (
                robot.data_dict["offsets"]["knee_to_ank_pitch_z"]
                / robot.data_dict["offsets"]["hip_pitch_to_knee_z"]
            )
            ank_pitch_angle = np.arctan2(np.sin(knee_angle), np.cos(knee_angle) + c)
            hip_pitch_angle = knee_angle - ank_pitch_angle

            pos_end[robot.joint_ordering.index("left_hip_pitch")] = -hip_pitch_angle
            pos_end[robot.joint_ordering.index("right_hip_pitch")] = hip_pitch_angle
            pos_end[robot.joint_ordering.index("left_knee_pitch")] = knee_angle
            pos_end[robot.joint_ordering.index("right_knee_pitch")] = -knee_angle
            pos_end[robot.joint_ordering.index("left_ank_pitch")] = -ank_pitch_angle
            pos_end[robot.joint_ordering.index("right_ank_pitch")] = -ank_pitch_angle

            pos_end[robot.joint_ordering.index("left_sho_pitch")] = np.pi / 6
            pos_end[robot.joint_ordering.index("right_sho_pitch")] = -np.pi / 6
            pos_end[robot.joint_ordering.index("left_elbow_roll")] = np.pi / 6
            pos_end[robot.joint_ordering.index("right_elbow_roll")] = np.pi / 6
            pos_end[robot.joint_ordering.index("left_wrist_roll_driven")] = -np.pi / 3
            pos_end[robot.joint_ordering.index("right_wrist_roll_driven")] = np.pi / 3

            joint_angles = dict(zip(robot.joint_ordering, pos_end))
            motor_angles = robot.joint_to_motor_angles(joint_angles)
            action_end = np.array(list(motor_angles.values()), dtype=np.float32)

            squat_time = np.arange(0, squat_duration, self.control_dt, dtype=np.float32)  # type: ignore
            squat_action = np.tile(action_end.copy(), (squat_time.shape[0], 1))
            for i, t in enumerate(squat_time):
                action = interpolate(
                    np.zeros_like(action_end),
                    action_end,
                    squat_duration,
                    t,
                )
                squat_action[i] = action + warm_up_action

            squat_time += time_list[-1][-1] + self.control_dt
            time_list.append(squat_time)
            action_list.append(squat_action)

            pause_time = np.arange(0, pause_duration, self.control_dt, dtype=np.float32)  # type: ignore
            pause_time += time_list[-1][-1] + self.control_dt
            pause_action = np.tile(action_list[-1][-1].copy(), (pause_time.shape[0], 1))

            time_list.append(pause_time)
            action_list.append(pause_action)

            reset_time, reset_pos = self.reset(
                time_list[-1][-1], action_list[-1][-1], warm_up_action, reset_duration
            )

            time_list.append(reset_time)
            action_list.append(reset_pos)

        reset_time, reset_pos = self.reset(
            time_list[-1][-1],
            action_list[-1][-1],
            np.zeros_like(action_list[-1][-1]),
            warm_up_duration,
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
