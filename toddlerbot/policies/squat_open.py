from typing import List

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate, interpolate_action
from toddlerbot.utils.misc_utils import set_seed


class SquatPolicy(BasePolicy):
    def __init__(self, robot: Robot):
        super().__init__(robot)
        self.name = "squat"
        self.control_dt = 0.02

        set_seed(0)

        self.default_q = np.array(
            list(robot.init_joint_angles.values()), dtype=np.float32
        )

        prep_duration = 10.0
        warm_up_duration = 1.0
        squat_duration = 4.0
        pause_duration = 2.0
        reset_duration = 3.0
        n_trials = 3

        time_list: List[npt.NDArray[np.float32]] = []
        action_list: List[npt.NDArray[np.float32]] = []

        prep_act = np.array(list(robot.init_motor_angles.values()), dtype=np.float32)
        prep_time, prep_action = self.warm_up(prep_act, prep_duration)

        time_list.append(prep_time)
        action_list.append(prep_action)

        warm_up_act = np.array(list(robot.init_motor_angles.values()), dtype=np.float32)
        warm_up_act[robot.motor_ordering.index("left_sho_roll")] = -np.pi / 24
        warm_up_act[robot.motor_ordering.index("right_sho_roll")] = -np.pi / 24
        # warm_up_act[robot.motor_ordering.index("left_sho_yaw_drive")] = -np.pi / 2
        # warm_up_act[robot.motor_ordering.index("right_sho_yaw_drive")] = np.pi / 2

        warm_up_time, warm_up_action = self.warm_up(warm_up_act, warm_up_duration)
        warm_up_time += time_list[-1][-1] + self.control_dt

        time_list.append(warm_up_time)
        action_list.append(warm_up_action)

        for i in range(n_trials):
            pos_end = self.default_q.copy()
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

            pos_end[robot.joint_ordering.index("left_sho_pitch")] = (
                np.pi / 10 if i == 0 else np.pi / 16
            )
            pos_end[robot.joint_ordering.index("right_sho_pitch")] = (
                -np.pi / 10 if i == 0 else -np.pi / 16
            )
            # pos_end[robot.joint_ordering.index("left_elbow_roll")] = np.pi / 6
            # pos_end[robot.joint_ordering.index("right_elbow_roll")] = np.pi / 6
            # pos_end[robot.joint_ordering.index("left_wrist_roll_driven")] = -np.pi / 3
            # pos_end[robot.joint_ordering.index("right_wrist_roll_driven")] = np.pi / 3

            joint_angles = dict(zip(robot.joint_ordering, pos_end))
            motor_angles = robot.joint_to_motor_angles(joint_angles)
            action_end = np.array(list(motor_angles.values()), dtype=np.float32)

            squat_time = np.arange(0, squat_duration, self.control_dt, dtype=np.float32)  # type: ignore
            squat_action = np.tile(action_end.copy(), (squat_time.shape[0], 1))  # type: ignore
            for i, t in enumerate(squat_time):
                action = interpolate(
                    np.zeros_like(action_end),
                    action_end,
                    squat_duration,
                    t,
                )
                squat_action[i] = action + warm_up_act

            squat_time += time_list[-1][-1] + self.control_dt
            time_list.append(squat_time)
            action_list.append(squat_action)

            pause_time, pause_action = self.move(
                time_list[-1][-1],
                action_list[-1][-1],
                action_list[-1][-1],
                pause_duration,
            )

            time_list.append(pause_time)
            action_list.append(pause_action)

            rise_act = warm_up_act.copy()
            rise_act[robot.motor_ordering.index("left_sho_pitch")] = 0.0
            rise_act[robot.motor_ordering.index("right_sho_pitch")] = 0.0

            rise_time, rise_action = self.move(
                time_list[-1][-1],
                action_list[-1][-1],
                rise_act,
                squat_duration,
            )

            time_list.append(rise_time)
            action_list.append(rise_action)

            reset_time, reset_action = self.move(
                time_list[-1][-1],
                action_list[-1][-1],
                warm_up_act,
                reset_duration,
            )

            time_list.append(reset_time)
            action_list.append(reset_action)

        rise_time, rise_action = self.move(
            time_list[-1][-1],
            action_list[-1][-1],
            np.zeros_like(action_list[-1][-1]),
            reset_duration,
        )

        time_list.append(rise_time)
        action_list.append(rise_action)

        self.time_arr = np.concatenate(time_list)  # type: ignore
        self.action_arr = np.concatenate(action_list)  # type: ignore

    def step(self, obs: Obs) -> npt.NDArray[np.float32]:
        action = np.array(interpolate_action(obs.time, self.time_arr, self.action_arr))

        # TODO: Fix this
        left_knee_pitch = obs.motor_pos[
            self.robot.joint_ordering.index("left_knee_pitch")
        ]
        right_knee_pitch = obs.motor_pos[
            self.robot.joint_ordering.index("right_knee_pitch")
        ]
        left_ank_pitch = obs.motor_pos[
            self.robot.joint_ordering.index("left_ank_pitch")
        ]
        right_ank_pitch = obs.motor_pos[
            self.robot.joint_ordering.index("right_ank_pitch")
        ]

        action[self.robot.joint_ordering.index("left_hip_pitch")] = (
            -left_ank_pitch - left_knee_pitch
        )
        action[self.robot.joint_ordering.index("right_hip_pitch")] = (
            right_ank_pitch - right_knee_pitch
        )

        # time_curr = obs_dict["time"].item()
        # action = np.array(interpolate_arr(time_curr, self.time_arr, self.action_arr))

        # c = (
        #     self.robot.data_dict["offsets"]["knee_to_ank_pitch_z"]
        #     / self.robot.data_dict["offsets"]["hip_pitch_to_knee_z"]
        # )
        # knee_angle = abs(
        #     obs_dict["q"][self.robot.joint_ordering.index("left_knee_pitch")]
        # )
        # ank_pitch_angle = np.arctan2(np.sin(knee_angle), np.cos(knee_angle) + c)
        # hip_pitch_angle = knee_angle - ank_pitch_angle

        # ank_act = self.robot.ankle_ik([0.0, -ank_pitch_angle])

        # action[self.robot.motor_ordering.index("left_hip_pitch")] = -hip_pitch_angle
        # action[self.robot.motor_ordering.index("right_hip_pitch")] = hip_pitch_angle
        # action[self.robot.motor_ordering.index("left_ank_act_1")] = ank_act[0]
        # action[self.robot.motor_ordering.index("left_ank_act_2")] = ank_act[1]
        # action[self.robot.motor_ordering.index("right_ank_act_1")] = -ank_act[0]
        # action[self.robot.motor_ordering.index("right_ank_act_2")] = -ank_act[1]

        return action
