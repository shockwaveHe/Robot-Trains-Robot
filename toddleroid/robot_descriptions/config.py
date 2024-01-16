from dataclasses import dataclass


@dataclass
class RobotConfig:
    left_foot_link: str
    right_foot_link: str
    joint_names: list = None
    magic_numbers: dict = None


robotis_op3_config = RobotConfig(
    left_foot_link="l_ank_roll_link", right_foot_link="r_ank_roll_link"
)

sustaina_op_config = RobotConfig(
    left_foot_link="left_foot_link",
    right_foot_link="right_foot_link",
    joint_names=[
        "waist_yaw_link",
        "waist_roll_link",
        "waist_pitch_link",
        "knee_pitch_link",
        "waist_pitch_mimic_link",
        "shin_pitch_link",
        "independent_pitch_link",
        "shin_pitch_mimic_link",
        "ankle_pitch_link",
        "ankle_roll_link",
    ],
    magic_numbers={
        "L1": 0.100,
        "L12": 0.057,
        "L2": 0.100,
        "L3": 0.053,
        "OFFSET_X": 0.0,
        "OFFSET_Y": 0.044,
    },
)

# Configuration dictionary
robots_config = {
    "Robotis_OP3": robotis_op3_config,
    "Sustaina_OP": sustaina_op_config,
}
