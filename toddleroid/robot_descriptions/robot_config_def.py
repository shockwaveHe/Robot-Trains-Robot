from dataclasses import dataclass


@dataclass
class RobotConfig:
    canonical_name2link_name: dict = None
    link_name2canonical_name: dict = None
    joint_names: list = None
    named_lengths: dict = None
