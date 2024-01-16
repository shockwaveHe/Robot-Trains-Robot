from dataclasses import dataclass


@dataclass
class RobotConfig:
    left_foot_link: str
    right_foot_link: str
    magic_numbers: dict = None


robotis_op3_config = RobotConfig(
    left_foot_link="l_ank_roll_link", right_foot_link="r_ank_roll_link"
)

sustaina_op_config = RobotConfig(
    left_foot_link="left_foot_link",
    right_foot_link="right_foot_link",
    magic_numbers={
        "L1": 0.100,
        "L12": 0.057,
        "L2": 0.100,
        "L3": 0.053,
        "OFFSET_W": 0.044,
        "OFFSET_X": 0.0,
    },
)

# Configuration dictionary
robots_config = {
    "Robotis_OP3": robotis_op3_config,
    "Sustaina_OP": sustaina_op_config,
}
