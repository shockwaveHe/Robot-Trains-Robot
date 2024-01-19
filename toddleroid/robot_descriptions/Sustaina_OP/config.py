from toddleroid.robot_descriptions.robot_config_def import *

canonical_name2link_name = {
    "left_foot_link": "left_foot_link",
    "right_foot_link": "right_foot_link",
}

link_name2canonical_name = {v: k for k, v in canonical_name2link_name.items()}


sustaina_op_config = RobotConfig(
    canonical_name2link_name=canonical_name2link_name,
    link_name2canonical_name=link_name2canonical_name,
    joint_names=[
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "knee_pitch_mimic_joint",
        "waist_pitch_mimic_joint",
        "knee_pitch_joint",
        "ankle_pitch_mimic_joint",
        "shin_pitch_mimic_joint",
        "ankle_pitch_joint",
        "ankle_roll_joint",
    ],
    named_lengths={
        "L1": 0.100,
        "L12": 0.057,
        "L2": 0.100,
        "L3": 0.053,
        "OFFSET_X": 0.0,
        "OFFSET_Y": 0.044,
    },
)
