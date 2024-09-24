# type: ignore

from typing import Dict, List

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.sysID import SysIDSpecs
from toddlerbot.utils.math_utils import (
    get_chirp_signal,
    interpolate_action,
)
from toddlerbot.utils.misc_utils import set_seed


class RotateTorsoOpenPolicy(BasePolicy, policy_name="rotate_torso_open"):
    def __init__(
        self, name: str, robot: Robot, init_motor_pos: npt.NDArray[np.float32]
    ):
        super().__init__(name, robot, init_motor_pos)
        set_seed(0)

        self.prep_duration = 7.0
        warm_up_duration = 2.0
        signal_duraion = 10.0
        reset_duration = 2.0

        # TODO: Update to use sine signals
        joint_sysID_specs = {
            "waist_roll": SysIDSpecs(
                amplitude_ratio=0.5,
                final_frequency=0.3,
                warm_up_angles={
                    "left_sho_roll": -np.pi / 6,
                    "right_sho_roll": -np.pi / 6,
                },
            ),
            "waist_yaw": SysIDSpecs(
                final_frequency=0.3,
                warm_up_angles={
                    "left_sho_roll": -np.pi / 6,
                    "right_sho_roll": -np.pi / 6,
                },
            ),
        }

        time_list: List[npt.NDArray[np.float32]] = []
        action_list: List[npt.NDArray[np.float32]] = []
        self.ckpt_dict: Dict[str, float] = {}

        prep_time, prep_action = self.move(
            -self.control_dt,
            init_motor_pos,
            np.zeros_like(init_motor_pos),
            self.prep_duration,
            end_time=5.0,
        )

        time_list.append(prep_time)
        action_list.append(prep_action)

        for joint_name, sysID_specs in joint_sysID_specs.items():
            joint_idx = robot.joint_ordering.index(joint_name)

            mean = (
                robot.joint_limits[joint_name][0] + robot.joint_limits[joint_name][1]
            ) / 2
            warm_up_act = np.zeros_like(init_motor_pos)
            warm_up_act[joint_idx] = mean

            if sysID_specs.warm_up_angles is not None:
                for name, angle in sysID_specs.warm_up_angles.items():
                    warm_up_act[robot.joint_ordering.index(name)] = angle

            if not np.allclose(warm_up_act, action_list[-1][-1], 1e-6):
                warm_up_time, warm_up_action = self.move(
                    time_list[-1][-1],
                    action_list[-1][-1],
                    warm_up_act,
                    warm_up_duration,
                )

                time_list.append(warm_up_time)
                action_list.append(warm_up_action)

            amplitude_max = robot.joint_limits[joint_name][1] - mean
            amplitude = sysID_specs.amplitude_ratio * amplitude_max

            rotate_time, signal = get_chirp_signal(
                signal_duraion,
                self.control_dt,
                0.0,
                sysID_specs.initial_frequency,
                sysID_specs.final_frequency,
                amplitude,
                sysID_specs.decay_rate,
            )
            rotate_time = np.asarray(rotate_time)
            signal = np.asarray(signal)

            rotate_time += time_list[-1][-1] + self.control_dt

            rotate_pos = np.zeros(
                (signal.shape[0], len(robot.joint_ordering)), np.float32
            )
            rotate_pos[:, joint_idx] = signal

            rotate_action = np.zeros_like(rotate_pos)
            for j, pos in enumerate(rotate_pos):
                signal_action = np.array(
                    list(
                        robot.joint_to_motor_angles(
                            dict(zip(robot.joint_ordering, pos))
                        ).values()
                    ),
                    dtype=np.float32,
                )
                rotate_action[j] = signal_action + warm_up_act

            time_list.append(rotate_time)
            action_list.append(rotate_action)

            reset_time, reset_action = self.move(
                time_list[-1][-1],
                action_list[-1][-1],
                warm_up_act,
                reset_duration,
                end_time=0.5,
            )

            time_list.append(reset_time)
            action_list.append(reset_action)
            self.ckpt_dict[joint_name] = time_list[-1][-1]

        self.time_arr = np.concatenate(time_list)
        self.action_arr = np.concatenate(action_list)
        self.n_steps_total = len(self.time_arr)

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        action = np.asarray(
            interpolate_action(obs.time, self.time_arr, self.action_arr)
        )
        return action
