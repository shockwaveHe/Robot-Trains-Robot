from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import get_chirp_signal, interpolate_action
from toddlerbot.utils.misc_utils import set_seed


# This script collects data for system identification of the motors.


@dataclass
class SysIDSpecs:
    """Dataclass for system identification specifications."""

    amplitude_list: List[float]
    initial_frequency: float = 0.1
    final_frequency: float = 10.0
    decay_rate: float = 0.1
    direction: float = 1
    kp_list: Optional[List[float]] = None
    warm_up_angles: Optional[Dict[str, float]] = None


class SysIDFixedPolicy(BasePolicy, policy_name="sysID"):
    """System identification policy for the toddlerbot robot."""

    def __init__(
        self, name: str, robot: Robot, init_motor_pos: npt.NDArray[np.float32]
    ):
        """Initializes the class with specified parameters and sets up system identification specifications for robot joints.

        Args:
            name (str): The name of the instance.
            robot (Robot): The robot object containing joint and motor information.
            init_motor_pos (npt.NDArray[np.float32]): Initial motor positions as a NumPy array.

        Attributes:
            prep_duration (float): Duration for preparation phase.
            ckpt_dict (Dict[float, Dict[str, float]]): Dictionary storing checkpoint data for joint names and their corresponding kp values.
            time_arr (npt.NDArray[np.float32]): Concatenated array of time steps for the entire process.
            action_arr (npt.NDArray[np.float32]): Concatenated array of actions corresponding to each time step.
            n_steps_total (int): Total number of steps in the time array.
        """
        super().__init__(name, robot, init_motor_pos)

        set_seed(0)

        self.prep_duration = 2.0
        warm_up_duration = 2.0
        signal_duraion = 10.0
        reset_duration = 2.0

        if "sysID" in robot.name:
            kp_list: List[float] = []
            if "330" in robot.name:
                kp_list = list(range(900, 2400, 300))
            else:
                kp_list = list(range(1500, 3600, 300))

            joint_sysID_specs = {
                "joint_0": SysIDSpecs(amplitude_list=[0.25, 0.5, 0.75], kp_list=kp_list)
            }
        else:
            XC330_kp_list = [1200.0, 1500.0, 1800.0]
            # XC430_kp_list = [1800.0, 2100.0, 2400.0]
            # XM430_kp_list = [2400.0, 2700.0, 3000.0]
            joint_sysID_specs = {
                # "neck_yaw_driven": SysIDSpecs(amplitude_max=np.pi / 2),
                # "neck_pitch": SysIDSpecs(),
                "ank_roll": SysIDSpecs(
                    amplitude_list=[0.25, 0.5, 0.75], kp_list=XC330_kp_list
                ),
                "ank_pitch": SysIDSpecs(
                    amplitude_list=[0.25, 0.5, 0.75], kp_list=XC330_kp_list
                ),
                # "knee": SysIDSpecs(
                #     amplitude_list=[0.25, 0.5, 0.75],
                #     warm_up_angles={
                #         "left_sho_roll": -np.pi / 12,
                #         "right_sho_roll": -np.pi / 12,
                #         "left_hip_roll": np.pi / 8,
                #         "right_hip_roll": np.pi / 8,
                #     },
                #     direction=-1,
                #     kp_list=XM430_kp_list,
                # ),
                # "hip_pitch": SysIDSpecs(
                #     amplitude_list=[0.25, 0.5, 0.75],
                #     warm_up_angles={
                #         "left_sho_roll": -np.pi / 12,
                #         "right_sho_roll": -np.pi / 12,
                #         "left_hip_roll": np.pi / 8,
                #         "right_hip_roll": np.pi / 8,
                #     },
                #     kp_list=XC430_kp_list,
                # ),
                # "hip_roll": SysIDSpecs(
                #     amplitude_list=[0.25, 0.5, 0.75],
                #     warm_up_angles={
                #         "left_sho_roll": -np.pi / 6,
                #         "right_sho_roll": -np.pi / 6,
                #     },
                #     direction=-1,
                #     kp_list=XC430_kp_list,
                # ),
                "hip_yaw_driven": SysIDSpecs(
                    amplitude_list=[0.25, 0.5, 0.75],
                    warm_up_angles={
                        "left_sho_roll": -np.pi / 6,
                        "right_sho_roll": -np.pi / 6,
                    },
                    kp_list=XC330_kp_list,
                ),
                "waist_roll": SysIDSpecs(
                    amplitude_list=[0.25, 0.5, 0.75],
                    warm_up_angles={
                        "left_sho_roll": -np.pi / 6,
                        "right_sho_roll": -np.pi / 6,
                    },
                    kp_list=XC330_kp_list,
                ),
                "waist_yaw": SysIDSpecs(
                    amplitude_list=[0.25, 0.5, 0.75],
                    warm_up_angles={
                        "left_sho_roll": -np.pi / 6,
                        "right_sho_roll": -np.pi / 6,
                    },
                    kp_list=XC330_kp_list,
                ),
                # "sho_yaw_driven": SysIDSpecs(
                #     warm_up_angles={
                #         "left_sho_roll": -np.pi / 6,
                #         "right_sho_roll": -np.pi / 6,
                #     },
                #     direction=-1,
                # ),
                # "elbow_yaw_driven": SysIDSpecs(
                #     warm_up_angles={
                #         "left_sho_roll": -np.pi / 6,
                #         "right_sho_roll": -np.pi / 6,
                #     },
                #     direction=-1,
                # ),
                # "wrist_pitch_driven": SysIDSpecs(
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
                # "wrist_roll": SysIDSpecs(
                #     warm_up_angles={
                #         "left_sho_roll": -np.pi / 6,
                #         "right_sho_roll": -np.pi / 6,
                #         "left_sho_yaw_driven": -np.pi / 2,
                #         "right_sho_yaw_driven": -np.pi / 2,
                #     },
                # ),
                # "sho_pitch": SysIDSpecs(),
                # "sho_roll": SysIDSpecs(),
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
            joint_idx: int | List[int] | None = None
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

            def build_episode(amplitude_ratio: float, kp: float):
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
                    amplitude_ratio * amplitude_max,
                    sysID_specs.decay_rate,
                )
                rotate_time = np.asarray(rotate_time)
                signal = np.asarray(signal)

                rotate_time += time_list[-1][-1] + self.control_dt

                rotate_pos = np.zeros((signal.shape[0], robot.nu), np.float32)

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
                for amplitude_ratio in sysID_specs.amplitude_list:
                    build_episode(amplitude_ratio, 0.0)
            else:
                for kp in sysID_specs.kp_list:
                    for amplitude_ratio in sysID_specs.amplitude_list:
                        build_episode(amplitude_ratio, kp)

        self.time_arr = np.concatenate(time_list)
        self.action_arr = np.concatenate(action_list)
        self.n_steps_total = len(self.time_arr)

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Executes a step in the environment by interpolating an action based on the given observation time.

        Args:
            obs (Obs): The observation containing the current time.
            is_real (bool, optional): Flag indicating whether the step is in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing an empty dictionary and the interpolated action as a NumPy array.
        """
        action = np.asarray(
            interpolate_action(obs.time, self.time_arr, self.action_arr)
        )
        return {}, action
