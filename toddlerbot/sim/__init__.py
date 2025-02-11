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
    pos: npt.NDArray[np.float32] = np.zeros(3, dtype=np.float32)
    euler: npt.NDArray[np.float32] = np.zeros(3, dtype=np.float32)
    joint_pos: Optional[npt.NDArray[np.float32]] = None
    joint_vel: Optional[npt.NDArray[np.float32]] = None
    ee_force: Optional[npt.NDArray[np.float32]] = np.zeros(3, dtype=np.float64)
    ee_torque: Optional[npt.NDArray[np.float32]] = np.zeros(3, dtype=np.float64)
    arm_joint_pos: Optional[npt.NDArray[np.float32]] = None
    arm_joint_vel: Optional[npt.NDArray[np.float32]] = None
    mocap_pos: Optional[npt.NDArray[np.float32]] = None
    mocap_quat: Optional[npt.NDArray[np.float32]] = None
    arm_ee_pos: Optional[npt.NDArray[np.float32]] = np.zeros(3, dtype=np.float64)
    arm_ee_quat: Optional[npt.NDArray[np.float32]] = None
    arm_ee_vel: Optional[npt.NDArray[np.float32]] = np.zeros(3, dtype=np.float64)
    feet_y_dist: Optional[float] = None
    is_done: Optional[bool] = None
    last_action: Optional[npt.NDArray[np.float32]] = None
    last_last_action: Optional[npt.NDArray[np.float32]] = None
    state_ref: Optional[npt.NDArray[np.float32]] = None


class BaseSim(ABC):
    @abstractmethod
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def set_motor_target(self, motor_angles: Dict[str, float]):
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

    def reset(self) -> Obs:
        pass

    def is_done(self, obs: Obs) -> bool:
        pass
