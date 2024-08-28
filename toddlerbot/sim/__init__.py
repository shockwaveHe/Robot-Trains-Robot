from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import numpy.typing as npt

from toddlerbot.actuation import JointState


@dataclass
class Obs:
    time: float
    u: npt.NDArray[np.float32]
    q: npt.NDArray[np.float32]
    dq: npt.NDArray[np.float32]
    euler: npt.NDArray[np.float32] = np.zeros(3, dtype=np.float32)
    ang_vel: npt.NDArray[np.float32] = np.zeros(3, dtype=np.float32)


def state_to_obs(
    motor_state_dict: Dict[str, JointState],
    joint_state_dict: Dict[str, JointState],
) -> Obs:
    time = list(joint_state_dict.values())[0].time

    a_obs: List[float] = []
    for motor_name in motor_state_dict:
        a_obs.append(motor_state_dict[motor_name].pos)

    q_obs: List[float] = []
    dq_obs: List[float] = []
    for joint_name in joint_state_dict:
        q_obs.append(joint_state_dict[joint_name].pos)
        dq_obs.append(joint_state_dict[joint_name].vel)

    obs = Obs(
        time=time,
        u=np.array(a_obs, dtype=np.float32),
        q=np.array(q_obs, dtype=np.float32),
        dq=np.array(dq_obs, dtype=np.float32),
    )
    return obs


class BaseSim(ABC):
    @abstractmethod
    def __init__(self, name: str):
        self.name = name
        self.visualizer = None

    @abstractmethod
    def set_motor_angles(self, motor_angles: Dict[str, float]):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def get_observation(self) -> Obs:
        pass

    @abstractmethod
    def close(self):
        pass
