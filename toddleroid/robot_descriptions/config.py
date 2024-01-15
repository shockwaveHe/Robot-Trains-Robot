from dataclasses import dataclass


@dataclass
class RobotConfig:
    left_foot_link: str
    right_foot_link: str


robotis_op3_config = RobotConfig(
    left_foot_link="l_ank_roll_link", right_foot_link="r_ank_roll_link"
)

another_robot_config = RobotConfig(
    left_foot_link="left_foot_link_name_other",
    right_foot_link="right_foot_link_name_other",
)

# Configuration dictionary
robots_config = {
    "Robotis_OP3": robotis_op3_config,
    "AnotherRobotModel": another_robot_config,
}
