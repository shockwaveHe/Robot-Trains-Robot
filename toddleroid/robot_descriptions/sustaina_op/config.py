import numpy as np

from toddleroid.robot_descriptions.robot_configs import RobotConfig

canonical_name2link_name = {
    "base_link": "base_link",
    "left_foot_link": "left_foot_link",
    "right_foot_link": "right_foot_link",
}

link_name2canonical_name = {v: k for k, v in canonical_name2link_name.items()}


sustaina_op_config = RobotConfig(
    com_height=0.3,
    canonical_name2link_name=canonical_name2link_name,
    link_name2canonical_name=link_name2canonical_name,
    joint_names=[
        "left_waist_yaw_joint",
        "left_waist_roll_joint",
        "left_waist_pitch_joint",
        "left_knee_pitch_mimic_joint",
        "left_waist_pitch_mimic_joint",
        "left_knee_pitch_joint",
        "left_ankle_pitch_mimic_joint",
        "left_shin_pitch_mimic_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_waist_yaw_joint",
        "right_waist_roll_joint",
        "right_waist_pitch_joint",
        "right_knee_pitch_mimic_joint",
        "right_waist_pitch_mimic_joint",
        "right_knee_pitch_joint",
        "right_ankle_pitch_mimic_joint",
        "right_shin_pitch_mimic_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ],
    offsets={
        "z_offset_hip": 0.053,
        "z_offset_thigh": 0.100,
        "z_offset_knee": 0.057,
        "z_offset_shin": 0.100,
        "x_offset_foot_to_ankle": 0.0,
        "y_offset_foot_to_ankle": 0.044,
        "left_offset_foot_to_sole": np.array([0.0, 0.01, -0.04]),
        "right_offset_foot_to_sole": np.array([0.0, -0.01, -0.04]),
        "y_offset_com_to_foot": 0.06,
    },
)
