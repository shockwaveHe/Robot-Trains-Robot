from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt


@dataclass
class Obs:
    """Observation data structure"""

    time: float
    motor_pos: npt.NDArray[np.float32]
    motor_vel: npt.NDArray[np.float32]
    motor_tor: npt.NDArray[np.float32]
    lin_vel: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    ang_vel: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    pos: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    euler: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
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
    hand_z_dist: Optional[npt.NDArray[np.float32]] = np.array([0.25, 0.25], dtype=np.float32)
    raw_action_mean: Optional[npt.NDArray[np.float32]] = 0.0
    base_action_mean: Optional[npt.NDArray[np.float32]] = 0.0
    is_done: Optional[bool] = None
    last_action: Optional[npt.NDArray[np.float32]] = None
    last_last_action: Optional[npt.NDArray[np.float32]] = None
    state_ref: Optional[npt.NDArray[np.float32]] = None


class BaseSim(ABC):
    """Base class for simulation environments"""

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


class DummySim(BaseSim):
    def __init__(self):
        super().__init__("dummy")

    def set_motor_target(self, motor_angles: Dict[str, float]):
        pass

    def set_motor_kps(self, motor_kps: Dict[str, float]):
        pass

    def step(self):
        pass

    def get_observation(self) -> Obs:
        return Obs(
            time=0.0,
            motor_pos=np.zeros(3, dtype=np.float32),
            motor_vel=np.zeros(3, dtype=np.float32),
            motor_tor=np.zeros(3, dtype=np.float32),
        )

    def close(self):
        pass

    def reset(self) -> Obs:
        return self.get_observation()

    def is_done(self, obs: Obs) -> bool:
        return False