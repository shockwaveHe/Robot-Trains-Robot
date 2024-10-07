from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt


@dataclass
class Obs:
    time: float
    motor_pos: npt.NDArray[np.float32]
    motor_vel: npt.NDArray[np.float32]
    motor_tor: npt.NDArray[np.float32]
    lin_vel: npt.NDArray[np.float32] = np.zeros(3, dtype=np.float32)
    ang_vel: npt.NDArray[np.float32] = np.zeros(3, dtype=np.float32)
    torso_pos: npt.NDArray[np.float32] = np.zeros(3, dtype=np.float32)
    torso_euler: npt.NDArray[np.float32] = np.zeros(3, dtype=np.float32)
    joint_pos: Optional[npt.NDArray[np.float32]] = None
    joint_vel: Optional[npt.NDArray[np.float32]] = None
    ee_force_data: Optional[npt.NDArray[np.float32]] = None # note: should read ee_force_data[0] when rigid connected
    ee_torque_data: Optional[npt.NDArray[np.float32]] = None
    arm_joint_pos: Optional[npt.NDArray[np.float32]] = None
    arm_joint_vel: Optional[npt.NDArray[np.float32]] = None
    mocap_pos: Optional[npt.NDArray[np.float32]] = None
    mocap_quat: Optional[npt.NDArray[np.float32]] = None
    


class BaseSim(ABC):
    @abstractmethod
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def set_motor_angles(self, motor_angles: Dict[str, float]):
        pass

    @abstractmethod
    def set_motor_kps(self, motor_kps: Dict[str, float]):
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
