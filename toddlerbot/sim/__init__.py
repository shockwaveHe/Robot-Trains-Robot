from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

from toddlerbot.sim.robot import HumanoidRobot


class BaseSim(ABC):
    @abstractmethod
    def __init__(self, robot: Optional[HumanoidRobot] = None):
        pass

    @abstractmethod
    def get_link_pos(self, robot: HumanoidRobot, link_name: str):
        pass

    @abstractmethod
    def get_com(self, robot: HumanoidRobot):
        pass

    @abstractmethod
    def get_zmp(self, robot: HumanoidRobot):
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
        vis_flags: Optional[List] = [],
    ):
        pass
