import numpy as np

from toddleroid.robot_descriptions.robot_configs import ActuatorParameters, RobotConfig

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
    act_params={
        "left_hip_yaw": ActuatorParameters(
            type="position", damping=0.1, armature=2.977e-6, kp=2.0, kv=0.1
        ),
        "left_hip_roll": ActuatorParameters(
            type="position", damping=0.1, armature=2.700e-5, kp=2.0, kv=0.1
        ),
        "left_hip_pitch": ActuatorParameters(
            type="position", damping=0.1, armature=2.700e-5, kp=2.0, kv=0.1
        ),
        "left_knee": ActuatorParameters(
            type="position", damping=0.1, armature=1e-7, kp=2.0, kv=0.1
        ),
        "left_ank_pitch": ActuatorParameters(
            type="position", damping=0.1, armature=1e-7, kp=2.0, kv=0.1
        ),
        "left_ank_roll": ActuatorParameters(
            type="position", damping=0.1, armature=1e-7, kp=2.0, kv=0.1
        ),
        # "left_ank_act1": ActuatorParameters(type="position", damping=0.1, armature=1e-7, kp=10.0, kv=0.1),
        # "left_ank_act2": ActuatorParameters(type="position", damping=0.1, armature=1e-7, kp=10.0, kv=0.1),
        "right_hip_yaw": ActuatorParameters(
            type="position", damping=0.1, armature=2.977e-6, kp=2.0, kv=0.1
        ),
        "right_hip_roll": ActuatorParameters(
            type="position", damping=0.1, armature=2.700e-5, kp=2.0, kv=0.1
        ),
        "right_hip_pitch": ActuatorParameters(
            type="position", damping=0.1, armature=2.700e-5, kp=2.0, kv=0.1
        ),
        "right_knee": ActuatorParameters(
            type="position", damping=0.1, armature=1e-7, kp=2.0, kv=0.1
        ),
        "right_ank_pitch": ActuatorParameters(
            type="position", damping=0.1, armature=1e-7, kp=2.0, kv=0.1
        ),
        "right_ank_roll": ActuatorParameters(
            type="position", damping=0.1, armature=1e-7, kp=2.0, kv=0.1
        ),
        # "right_ank_act1": ActuatorParameters(type="position", damping=0.1, armature=1e-7, kp=10.0, kv=0.1),
        # "right_ank_act2": ActuatorParameters(type="position", damping=0.1, armature=1e-7, kp=10.0, kv=0.1),
        "left_sho_pitch": ActuatorParameters(
            type="position", damping=0.1, armature=1.735e-5, kp=5.0, kv=0.1
        ),
        "left_sho_roll": ActuatorParameters(
            type="position", damping=0.1, armature=1.735e-5, kp=5.0, kv=0.1
        ),
        "left_elb": ActuatorParameters(
            type="position", damping=0.1, armature=1.735e-5, kp=5.0, kv=0.1
        ),
        "right_sho_pitch": ActuatorParameters(
            type="position", damping=0.1, armature=1.735e-5, kp=5.0, kv=0.1
        ),
        "right_sho_roll": ActuatorParameters(
            type="position", damping=0.1, armature=1.735e-5, kp=5.0, kv=0.1
        ),
        "right_elb": ActuatorParameters(
            type="position", damping=0.1, armature=1.735e-5, kp=5.0, kv=0.1
        ),
    },
    constraint_pairs=[
        ("ank_act_rod_head", "ank_act_rod"),
        ("ank_act_rod_head_2", "ank_act_rod_2"),
        ("ank_act_rod_head_3", "ank_act_rod_3"),
        ("ank_act_rod_head_4", "ank_act_rod_4"),
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
)
