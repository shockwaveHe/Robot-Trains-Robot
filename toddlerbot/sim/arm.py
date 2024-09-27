import json
import os
import pickle
from typing import Any, Dict, List, Optional, Type

import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

arm_registry: Dict[str, Type["BaseArm"]] = {}


def get_arm_class(arm_name: str) -> Type["BaseArm"]:
    if arm_name not in arm_registry:
        raise ValueError(f"Unknown arm type: {arm_name}")

    return arm_registry[arm_name]

class BaseArm(ABC):
    @abstractmethod
    def __init__(self, name: Optional[str] = "", arm_dofs: Optional[int] = None, arm_nbodies: Optional[int] = None):
        self.name = name
        self.arm_dofs = arm_dofs
        self.arm_nbodies = arm_nbodies
        self.joint_limits = {}
        self.joint_ordering = []

    # Automatic registration of subclasses
    def __init_subclass__(cls, arm_name: str = "", **kwargs):
        super().__init_subclass__(**kwargs)
        if len(arm_name) > 0:
            arm_registry[arm_name] = cls
    
class StandradBotArm(BaseArm, arm_name="standard_bot"):
    def __init__(
            self, 
            name: str = "standard_bot",
            arm_dofs: int = 6,
            arm_nbodies: int = 8,

        ):
        super().__init__(name, arm_dofs, arm_nbodies)
        self.joint_ordering = [
            "joint0",
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
        ]


class FrankaArm(BaseArm, arm_name="franka"):
    def __init__(
            self, 
            name: str = "franka",
            arm_dofs: int = 7,
            arm_nbodies: int = 9,
        ):
        super().__init__(name, arm_dofs, arm_nbodies)
        self.joint_ordering = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]