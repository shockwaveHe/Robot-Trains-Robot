from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

from toddleroid.sim.robot import HumanoidRobot


class AbstractSim(ABC):
    @abstractmethod
    def __init__(self, robot: Optional[HumanoidRobot] = None):
        pass

    @abstractmethod
    def get_joint_name2qidx(self, robot: HumanoidRobot):
        pass

    @abstractmethod
    def get_link_pos(self, robot: HumanoidRobot, link_name: str):
        pass

    @abstractmethod
    def initialize_named_joint_angles(self, robot: HumanoidRobot):
        pass

    @abstractmethod
    def set_joint_angles(self, robot: HumanoidRobot, joint_angles: List[float]):
        pass

    @abstractmethod
    def simulate(
        self,
        step_func: Optional[Callable] = None,
        step_params: Optional[Tuple] = None,
        sleep_time: float = 0.0,
    ):
        pass
