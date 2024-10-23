from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Type

arm_registry: Dict[str, Type["BaseArm"]] = {}

_HERE = Path(__file__).parent
_ROBOT_DESCRIPTIONS_PATH = _HERE.parent / "robot_descriptions"


def get_arm_class(arm_name: str) -> Type["BaseArm"]:
    if arm_name not in arm_registry:
        raise ValueError(f"Unknown arm type: {arm_name}")

    return arm_registry[arm_name]


class BaseArm(ABC):
    @abstractmethod
    def __init__(self, name: str, arm_dofs: int, arm_nbodies: int, xml_path: Path):
        self.name = name
        self.arm_dofs = arm_dofs
        self.arm_nbodies = arm_nbodies

        self.xml_path = xml_path
        self.joint_ordering = []  # type: ignore

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
        xml_path: Path = _ROBOT_DESCRIPTIONS_PATH
        / "standard_bot"
        / "standard_bot_scene.xml",
    ):
        super().__init__(name, arm_dofs, arm_nbodies, xml_path)
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
        xml_path: Path = _ROBOT_DESCRIPTIONS_PATH / "franka" / "franka_scene.xml",
    ):
        super().__init__(name, arm_dofs, arm_nbodies, xml_path)
        self.joint_ordering = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
