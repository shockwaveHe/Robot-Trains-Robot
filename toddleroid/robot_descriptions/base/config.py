import math

import numpy as np

from toddleroid.robot_descriptions.robot_configs import ActuatorParameters, RobotConfig

canonical_name2link_name = {
    "body_link": "body_link",
    "left_foot_link": "ank_roll_link",
    "right_foot_link": "ank_roll_link_2",
}

link_name2canonical_name = {v: k for k, v in canonical_name2link_name.items()}


def compute_leg_angles(target_foot_pos, target_foot_ori, side, offsets):
    # Decompose target position and orientation
    target_x, target_y, target_z = target_foot_pos
    ankle_roll, ankle_pitch, hip_yaw = target_foot_ori

    # Adjust positions based on offsets and compute new coordinates
    target_x += offsets["x_offset_ankle_to_foot"]
    target_y += (
        -offsets["y_offset_ankle_to_foot"]
        if side == "left"
        else offsets["y_offset_ankle_to_foot"]
    )
    target_z = (
        offsets["z_offset_thigh"]
        + offsets["z_offset_knee"]
        + offsets["z_offset_shin"]
        - target_z
    )

    transformed_x = target_x * math.cos(target_foot_ori[2]) + target_y * math.sin(
        target_foot_ori[2]
    )
    transformed_y = -target_x * math.sin(target_foot_ori[2]) + target_y * math.cos(
        target_foot_ori[2]
    )
    transformed_z = target_z

    hip_roll = math.atan2(transformed_y, transformed_z)

    leg_projected_yz_length = math.sqrt(transformed_y**2 + transformed_z**2)
    leg_length = math.sqrt(transformed_x**2 + leg_projected_yz_length**2)
    leg_pitch = math.atan2(transformed_x, leg_projected_yz_length)
    wrist_disp_cos = (
        leg_length**2 + offsets["z_offset_shin"] ** 2 - offsets["z_offset_thigh"] ** 2
    ) / (2 * leg_length * offsets["z_offset_shin"])
    wrist_disp = math.acos(min(max(wrist_disp_cos, -1.0), 1.0))
    ankle_disp = math.asin(
        offsets["z_offset_thigh"] / offsets["z_offset_shin"] * math.sin(wrist_disp)
    )
    hip_pitch = -leg_pitch - wrist_disp
    knee_pitch = wrist_disp + ankle_disp
    ankle_pitch += knee_pitch + hip_pitch

    angles_dict = {
        "hip_yaw": -hip_yaw,
        "hip_roll": -hip_roll,
        "hip_pitch": hip_pitch,
        "knee": -knee_pitch if side == "left" else knee_pitch,
        "ank_pitch": ankle_pitch if side == "left" else -ankle_pitch,
        "ank_roll": ankle_roll + hip_roll,
    }
    return angles_dict


base_config = RobotConfig(
    com_height=0.3,
    canonical_name2link_name=canonical_name2link_name,
    link_name2canonical_name=link_name2canonical_name,
    # act_params={
    #     "left_hip_yaw": ActuatorParameters(
    #         type="motor", damping=0.1, armature=2.977e-6, kp=10.0, kv=0.1
    #     ),
    #     "left_hip_roll": ActuatorParameters(
    #         type="motor", damping=0.1, armature=2.700e-5, kp=10.0, kv=0.1
    #     ),
    #     "left_hip_pitch": ActuatorParameters(
    #         type="motor", damping=0.1, armature=2.700e-5, kp=10.0, kv=0.1
    #     ),
    #     "left_knee": ActuatorParameters(
    #         type="motor", damping=0.1, armature=1e-5, kp=10.0, kv=0.1
    #     ),
    #     "left_ank_pitch": ActuatorParameters(
    #         type="motor", damping=0.1, armature=1e-7, kp=10.0, kv=0.1
    #     ),
    #     "left_ank_roll": ActuatorParameters(
    #         type="motor", damping=0.1, armature=1e-7, kp=10.0, kv=0.1
    #     ),
    #     # "left_ank_act1": ActuatorParameters(type="motor", damping=0.1, armature=1e-7, kp=10.0, kv=0.1),
    #     # "left_ank_act2": ActuatorParameters(type="motor", damping=0.1, armature=1e-7, kp=10.0, kv=0.1),
    #     "right_hip_yaw": ActuatorParameters(
    #         type="motor", damping=0.1, armature=2.977e-6, kp=10.0, kv=0.1
    #     ),
    #     "right_hip_roll": ActuatorParameters(
    #         type="motor", damping=0.1, armature=2.700e-5, kp=10.0, kv=0.1
    #     ),
    #     "right_hip_pitch": ActuatorParameters(
    #         type="motor", damping=0.1, armature=2.700e-5, kp=10.0, kv=0.1
    #     ),
    #     "right_knee": ActuatorParameters(
    #         type="motor", damping=0.1, armature=1e-5, kp=10.0, kv=0.1
    #     ),
    #     "right_ank_pitch": ActuatorParameters(
    #         type="motor", damping=0.1, armature=1e-7, kp=10.0, kv=0.1
    #     ),
    #     "right_ank_roll": ActuatorParameters(
    #         type="motor", damping=0.1, armature=1e-7, kp=10.0, kv=0.1
    #     ),
    #     # "right_ank_act1": ActuatorParameters(type="motor", damping=0.1, armature=1e-7, kp=10.0, kv=0.1),
    #     # "right_ank_act2": ActuatorParameters(type="motor", damping=0.1, armature=1e-7, kp=10.0, kv=0.1),
    #     "left_sho_pitch": ActuatorParameters(
    #         type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
    #     ),
    #     "left_sho_roll": ActuatorParameters(
    #         type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
    #     ),
    #     "left_elb": ActuatorParameters(
    #         type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
    #     ),
    #     "right_sho_pitch": ActuatorParameters(
    #         type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
    #     ),
    #     "right_sho_roll": ActuatorParameters(
    #         type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
    #     ),
    #     "right_elb": ActuatorParameters(
    #         type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
    #     ),
    # },
    act_params={
        "left_hip_yaw": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_hip_roll": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_hip_pitch": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_knee": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_ank_pitch": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_ank_roll": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        # "left_ank_act1": ActuatorParameters(type="motor", damping=0.1, armature=1e-7, kp=10.0, kv=0.1),
        # "left_ank_act2": ActuatorParameters(type="motor", damping=0.1, armature=1e-7, kp=10.0, kv=0.1),
        "right_hip_yaw": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_hip_roll": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_hip_pitch": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_knee": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_ank_pitch": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_ank_roll": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        # "right_ank_act1": ActuatorParameters(type="motor", damping=0.1, armature=1e-7, kp=10.0, kv=0.1),
        # "right_ank_act2": ActuatorParameters(type="motor", damping=0.1, armature=1e-7, kp=10.0, kv=0.1),
        "left_sho_pitch": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_sho_roll": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_elb": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_sho_pitch": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_sho_roll": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_elb": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
    },
    constraint_pairs=[
        ("ank_act_rod_head", "ank_act_rod"),
        ("ank_act_rod_head_2", "ank_act_rod_2"),
        ("ank_act_rod_head_3", "ank_act_rod_3"),
        ("ank_act_rod_head_4", "ank_act_rod_4"),
    ],
    offsets={
        "z_offset_hip": 0.0515,  # from the hip yaw joint to the hip pitch joint
        "z_offset_thigh": 0.1075,  # from the hip pitch joint to the knee joint
        "z_offset_knee": 0.0,
        "z_offset_shin": 0.1,  # from the knee joint to the ankle pitch joint
        "x_offset_ankle_to_foot": 0.0,
        "y_offset_ankle_to_foot": 0.044,
        "left_offset_foot_to_sole": np.array([0.0, 0.01, -0.03]),
        "right_offset_foot_to_sole": np.array([0.0, -0.01, -0.03]),
        "foot_size_x": 0.095,
        "foot_size_y": 0.03,
        "foot_size_z": 0.004,
    },
    compute_leg_angles=compute_leg_angles,
)
