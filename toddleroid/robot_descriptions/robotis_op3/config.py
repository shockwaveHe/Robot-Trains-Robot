import numpy as np

from toddleroid.robot_descriptions.robot_configs import RobotConfig

canonical_name2link_name = {
    "base_link": "body_link",
    "left_foot_link": "l_ank_roll_link",
    "right_foot_link": "r_ank_roll_link",
}

link_name2canonical_name = {v: k for k, v in canonical_name2link_name.items()}


robotis_op3_config = RobotConfig(
    com_height=0.3,
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
        "z_offset_hip": 0.028,
        "z_offset_thigh": 0.11,  # from the hip pitch joint to the knee joint
        "z_offset_knee": 0.0,
        "z_offset_shin": 0.11,  # from the knee joint to the ankle pitch joint
        "x_offset_foot_to_ankle": 0.0,
        "y_offset_foot_to_ankle": 0.044,
        "left_offset_foot_to_sole": np.array([0.0, 0.01, 0.0]),
        "right_offset_foot_to_sole": np.array([0.0, -0.01, 0.0]),
    },
    gains={
        "kp": 100,
        "kv": 10,
    },
)
