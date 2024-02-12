import numpy as np

from toddleroid.robot_descriptions.robot_configs import RobotConfig

canonical_name2link_name = {
    "body_link": "body_link",
    "left_foot_link": "l_ank_roll_link",
    "right_foot_link": "r_ank_roll_link",
}

link_name2canonical_name = {v: k for k, v in canonical_name2link_name.items()}


base_config = RobotConfig(
    com_height=0.3,
    canonical_name2link_name=canonical_name2link_name,
    link_name2canonical_name=link_name2canonical_name,
    joint_names=[
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ank_pitch",
        "left_ank_roll",
        # "left_ank_act1",
        # "left_ank_act2",
        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee",
        "right_ank_pitch",
        "right_ank_roll",
        # "right_ank_act1",
        # "right_ank_act2",
        "left_sho_pitch",
        "left_sho_roll",
        "left_elb",
        "right_sho_pitch",
        "right_sho_roll",
        "right_elb",
    ],
    offsets={
        # "z_offset_hip": 0.028,
        # "z_offset_thigh": 0.11,  # from the hip pitch joint to the knee joint
        # "z_offset_knee": 0.0,
        # "z_offset_shin": 0.11,  # from the knee joint to the ankle pitch joint
        # "x_offset_foot_to_ankle": 0.0,
        # "y_offset_foot_to_ankle": 0.044,
        # "left_offset_foot_to_sole": np.array([0.0, 0.01, 0.0]),
        # "right_offset_foot_to_sole": np.array([0.0, -0.01, 0.0]),
        # "x_offset_sole": 0.127,
        # "y_offset_sole": 0.076,
        # "z_offset_sole": 0.002,
    },
    gains={
        "kp": 100,
        "kv": 10,
    },
)
