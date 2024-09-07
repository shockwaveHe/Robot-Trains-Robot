from typing import Dict, List

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.sysID import SysIDSpecs
from toddlerbot.utils.math_utils import get_chirp_signal, interpolate_action
from toddlerbot.utils.misc_utils import set_seed


class SysIDFixedPolicy(BasePolicy):
    def __init__(self, robot: Robot, init_motor_pos: npt.NDArray[np.float32]):
        super().__init__("sysID_fixed", robot, init_motor_pos)

        set_seed(0)

        self.prep_duration = 2.0
        warm_up_duration = 2.0
        signal_duraion = 10.0
        reset_duration = 2.0

        if "sysID" in robot.name:
            joint_sysID_specs = {
                "joint_0": SysIDSpecs(
                    final_frequency=1.5, kp_list=list(range(300, 3000, 300))
                )
            }
        else:
            joint_sysID_specs = {
                # "neck_yaw_driven": SysIDSpecs(amplitude_max=np.pi / 2),
                # "neck_pitch_driven": SysIDSpecs(),
                "waist_roll": SysIDSpecs(
                    warm_up_angles={
                        "left_sho_roll": -np.pi / 6,
                        "right_sho_roll": -np.pi / 6,
                    },
                    kp_list=list(range(1200, 2400, 300)),
                ),
                "waist_yaw": SysIDSpecs(
                    warm_up_angles={
                        "left_sho_roll": -np.pi / 6,
                        "right_sho_roll": -np.pi / 6,
                    },
                    kp_list=list(range(1200, 2400, 300)),
                ),
                "hip_yaw_driven": SysIDSpecs(
                    amplitude_ratio=0.5,
                    warm_up_angles={
                        "left_sho_roll": -np.pi / 6,
                        "right_sho_roll": -np.pi / 6,
                    },
                    kp_list=list(range(1200, 2400, 300)),
                ),
                "hip_roll": SysIDSpecs(
                    warm_up_angles={
                        "left_sho_roll": -np.pi / 6,
                        "right_sho_roll": -np.pi / 6,
                    },
                    direction=-1,
                    kp_list=list(range(1800, 3000, 300)),
                ),
                "hip_pitch": SysIDSpecs(
                    warm_up_angles={
                        "left_sho_roll": -np.pi / 12,
                        "right_sho_roll": -np.pi / 12,
                        "left_hip_roll": np.pi / 8,
                        "right_hip_roll": np.pi / 8,
                    },
                    kp_list=list(range(1800, 3000, 300)),
                ),
                "knee_pitch": SysIDSpecs(
                    warm_up_angles={
                        "left_sho_roll": -np.pi / 12,
                        "right_sho_roll": -np.pi / 12,
                        "left_hip_roll": np.pi / 8,
                        "right_hip_roll": np.pi / 8,
                    },
                    direction=-1,
                    kp_list=list(range(2400, 3300, 300)),
                ),
                # "sho_yaw_driven": SysIDSpecs(
                #     amplitude_max=np.pi / 4,
                #     warm_up_angles={
                #         "left_sho_roll": -np.pi / 6,
                #         "right_sho_roll": -np.pi / 6,
                #     },
                #     direction=-1,
                # ),
                # "elbow_yaw_driven": SysIDSpecs(
                #     amplitude_max=np.pi / 4,
                #     warm_up_angles={
                #         "left_sho_roll": -np.pi / 6,
                #         "right_sho_roll": -np.pi / 6,
                #     },
                #     direction=-1,
                # ),
                # "wrist_pitch": SysIDSpecs(
                #     warm_up_angles={
                #         "left_sho_roll": -np.pi / 6,
                #         "right_sho_roll": -np.pi / 6,
                #     },
                # ),
                # "elbow_roll": SysIDSpecs(
                #     warm_up_angles={
                #         "left_sho_roll": -np.pi / 6,
                #         "right_sho_roll": -np.pi / 6,
                #         "left_sho_yaw_driven": -np.pi / 2,
                #         "right_sho_yaw_driven": -np.pi / 2,
                #     },
                # ),
                # "wrist_roll_driven": SysIDSpecs(
                #     warm_up_angles={
                #         "left_sho_roll": -np.pi / 6,
                #         "right_sho_roll": -np.pi / 6,
                #         "left_sho_yaw_driven": -np.pi / 2,
                #         "right_sho_yaw_driven": -np.pi / 2,
                #     },
                # ),
                "ank_roll": SysIDSpecs(kp_list=list(range(1200, 2400, 300))),
                "ank_pitch": SysIDSpecs(kp_list=list(range(1200, 2400, 300))),
                # "sho_pitch": SysIDSpecs(amplitude_max=np.pi / 4),
                # "sho_roll": SysIDSpecs(amplitude_max=np.pi / 4),
            }

        time_list: List[npt.NDArray[np.float32]] = []
        action_list: List[npt.NDArray[np.float32]] = []
        self.ckpt_dict: Dict[float, Dict[str, float]] = {}

        prep_time, prep_action = self.move(
            -self.control_dt,
            init_motor_pos,
            np.zeros_like(init_motor_pos),
            self.prep_duration,
        )

        time_list.append(prep_time)
        action_list.append(prep_action)

        for symm_joint_name, sysID_specs in joint_sysID_specs.items():
            if symm_joint_name in robot.joint_ordering:
                joint_names = [symm_joint_name]
                joint_idx = robot.joint_ordering.index(joint_names[0])
            else:
                joint_names = [f"left_{symm_joint_name}", f"right_{symm_joint_name}"]
                joint_idx = [
                    robot.joint_ordering.index(joint_names[0]),
                    robot.joint_ordering.index(joint_names[1]),
                ]

            mean = (
                robot.joint_limits[joint_names[0]][0]
                + robot.joint_limits[joint_names[0]][1]
            ) / 2
            warm_up_pos = np.zeros_like(init_motor_pos)

            if isinstance(joint_idx, int):
                warm_up_pos[joint_idx] = mean
            else:
                warm_up_pos[joint_idx[0]] = mean
                warm_up_pos[joint_idx[1]] = mean * sysID_specs.direction

            if sysID_specs.warm_up_angles is not None:
                for name, angle in sysID_specs.warm_up_angles.items():
                    warm_up_pos[robot.joint_ordering.index(name)] = angle

            warm_up_motor_angles = robot.joint_to_motor_angles(
                dict(zip(robot.joint_ordering, warm_up_pos))
            )
            warm_up_act = np.array(
                list(warm_up_motor_angles.values()), dtype=np.float32
            )

            amplitude_max = robot.joint_limits[joint_names[0]][1] - mean
            amplitude = sysID_specs.amplitude_ratio * amplitude_max

            def build_episode(kp: float):
                if not np.allclose(warm_up_act, action_list[-1][-1], 1e-6):
                    warm_up_time, warm_up_action = self.move(
                        time_list[-1][-1],
                        action_list[-1][-1],
                        warm_up_act,
                        warm_up_duration,
                    )

                    time_list.append(warm_up_time)
                    action_list.append(warm_up_action)

                rotate_time, signal = get_chirp_signal(
                    signal_duraion,
                    self.control_dt,
                    0.0,
                    sysID_specs.initial_frequency,
                    sysID_specs.final_frequency,
                    amplitude,
                )
                rotate_time = np.asarray(rotate_time)
                signal = np.asarray(signal)

                rotate_time += time_list[-1][-1] + self.control_dt

                rotate_pos = np.zeros(
                    (signal.shape[0], len(robot.joint_ordering)), np.float32
                )

                if isinstance(joint_idx, int):
                    rotate_pos[:, joint_idx] = signal
                else:
                    rotate_pos[:, joint_idx[0]] = signal
                    rotate_pos[:, joint_idx[1]] = signal * sysID_specs.direction

                rotate_action = np.zeros_like(rotate_pos)
                for j, pos in enumerate(rotate_pos):
                    rotate_motor_angles = robot.joint_to_motor_angles(
                        dict(zip(robot.joint_ordering, pos))
                    )
                    signal_action = np.array(
                        list(rotate_motor_angles.values()), dtype=np.float32
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

                self.ckpt_dict[time_list[-1][-1]] = dict(
                    zip(joint_names, [kp] * len(joint_names))
                )

            if sysID_specs.kp_list is None:
                build_episode(0.0)
            else:
                for kp in sysID_specs.kp_list:
                    build_episode(kp)

        self.time_arr = np.concatenate(time_list)  # type: ignore
        self.action_arr = np.concatenate(action_list)  # type: ignore
        self.n_steps_total = len(self.time_arr)

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        action = np.asarray(
            interpolate_action(obs.time, self.time_arr, self.action_arr)
        )
        return action
