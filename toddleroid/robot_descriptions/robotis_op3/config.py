import numpy as np

from toddleroid.robot_descriptions.robot_configs import RobotConfig

canonical_name2link_name = {
    "left_foot_link": "l_ank_roll_link",
    "right_foot_link": "r_ank_roll_link",
}

link_name2canonical_name = {v: k for k, v in canonical_name2link_name.items()}


robotis_op3_config = RobotConfig(
    com_height=0.36,
    canonical_name2link_name=canonical_name2link_name,
    link_name2canonical_name=link_name2canonical_name,
    joint_names=[
        "l_hip_yaw",
        "l_hip_roll",
        "l_hip_pitch",
        "l_knee",
        "l_ank_pitch",
        "l_ank_roll",
        "r_hip_yaw",
        "r_hip_roll",
        "r_hip_pitch",
        "r_knee",
        "r_ank_pitch",
        "r_ank_roll",
    ],
    offsets={
        "L1": 0.100,
        "L12": 0.057,
        "L2": 0.100,
        "L3": 0.053,
        "OFFSET_X": 0.0,
        "OFFSET_Y": 0.044,
        "left_offset_foot_to_sole": np.array([0.0, 0.0, 0.0]),
        "right_offset_foot_to_sole": np.array([0.0, -0.0, 0.0]),
        "y_offset_com_to_foot": 0.06,
    },
)
