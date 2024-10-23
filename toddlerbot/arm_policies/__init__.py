from abc import ABC, abstractmethod
from typing import Dict, List, Type

import numpy as np
import numpy.typing as npt

from toddlerbot.sim import Obs
from toddlerbot.sim.arm import BaseArm

arm_policy_registry: Dict[str, Type["BaseArmPolicy"]] = {}


def get_arm_policy_class(arm_policy_name: str) -> Type["BaseArmPolicy"]:
    if arm_policy_name not in arm_policy_registry:
        raise ValueError(f"Unknown arm policy: {arm_policy_name}")

    return arm_policy_registry[arm_policy_name]


def get_arm_policy_names() -> List[str]:
    policy_names: List[str] = []
    for key in arm_policy_registry.keys():
        policy_names.append(key)
        policy_names.append(key + "_fixed")

    return policy_names


class BaseArmPolicy(ABC):
    # Automatic registration of subclasses
    def __init_subclass__(cls, arm_policy_name: str = "", **kwargs):
        super().__init_subclass__(**kwargs)
        if len(arm_policy_name) > 0:
            arm_policy_registry[arm_policy_name] = cls

    @abstractmethod
    def __init__(
        self,
        name: str,
        arm: BaseArm,
        init_joint_pos: npt.NDArray[np.float32],
        control_dt: float = 0.02,
    ):
        self.name = name
        self.arm = arm
        self.init_joint_pos = init_joint_pos
        self.control_dt = control_dt

    @abstractmethod
    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        pass
