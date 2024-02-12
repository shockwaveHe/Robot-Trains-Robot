# File: toddleroid/robot_descriptions/robot_configs.py

import importlib
import os
from dataclasses import dataclass
from typing import Callable


@dataclass
class RobotConfig:
    com_height: float = 0.0
    act_params: dict = None
    constraint_pairs: list = None
    canonical_name2link_name: dict = None
    link_name2canonical_name: dict = None
    offsets: dict = None
    compute_leg_angles: Callable = None


@dataclass
class ActuatorParameters:
    type: str = ""
    damping: float = 0.0
    armature: float = 0.0
    kp: float = 0.0
    kv: float = 0.0


def load_robot_configs(
    base_path="toddleroid/robot_descriptions", ignore_dirs=["__pycache__"]
):
    configs = {}
    for robot_name in os.listdir(base_path):
        if robot_name in ignore_dirs:
            continue
        robot_path = os.path.join(base_path, robot_name)
        if os.path.isdir(robot_path):
            config_module_path = f"{robot_path.replace('/', '.')}.config"
            try:
                config_module = importlib.import_module(config_module_path)
                robot_config = getattr(config_module, f"{robot_name.lower()}_config")
                configs[robot_name] = robot_config
            except (ImportError, AttributeError):
                print(f"Warning: Could not load config for {robot_name}")
    return configs


robot_configs = load_robot_configs()
# print(robot_configs)
