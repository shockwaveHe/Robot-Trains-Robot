import math

import numpy as np

from toddlerbot.robot_descriptions.robot_configs import *

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
    hip_yaw = -hip_yaw

    target_z = (
        offsets["z_offset_thigh"]
        + offsets["z_offset_knee"]
        + offsets["z_offset_shin"]
        - target_z
    )

    transformed_x = target_x * math.cos(hip_yaw) + target_y * math.sin(hip_yaw)
    transformed_y = -target_x * math.sin(hip_yaw) + target_y * math.cos(hip_yaw)
    transformed_z = target_z

    hip_roll = math.atan2(
        transformed_y, transformed_z + offsets["z_offset_hip_roll_to_pitch"]
    )

    leg_projected_yz_length = math.sqrt(transformed_y**2 + transformed_z**2)
    leg_length = math.sqrt(transformed_x**2 + leg_projected_yz_length**2)
    leg_pitch = math.atan2(transformed_x, leg_projected_yz_length)
    hip_disp_cos = (
        leg_length**2 + offsets["z_offset_thigh"] ** 2 - offsets["z_offset_shin"] ** 2
    ) / (2 * leg_length * offsets["z_offset_thigh"])
    hip_disp = math.acos(min(max(hip_disp_cos, -1.0), 1.0))
    ankle_disp = math.asin(
        offsets["z_offset_thigh"] / offsets["z_offset_shin"] * math.sin(hip_disp)
    )
    hip_pitch = -leg_pitch - hip_disp
    knee_pitch = hip_disp + ankle_disp
    ankle_pitch += knee_pitch + hip_pitch

    angles_dict = {
        "hip_yaw": hip_yaw,
        "hip_roll": hip_roll if side == "left" else -hip_roll,
        "hip_pitch": -hip_pitch if side == "left" else hip_pitch,
        "knee": knee_pitch if side == "left" else -knee_pitch,
        "ank_pitch": ankle_pitch if side == "left" else -ankle_pitch,
        "ank_roll": ankle_roll - hip_roll,
    }
    return angles_dict


base_config = RobotConfig(
    canonical_name2link_name=canonical_name2link_name,
    link_name2canonical_name=link_name2canonical_name,
    motor_params={
        "left_hip_yaw": MotorParameters(
            brand="dynamixel",
            id=7,
            type="motor",
            damping=1.084,
            armature=0.045,
            kp=400.0,
            kv=40.0,
        ),
        "left_hip_roll": MotorParameters(
            brand="dynamixel",
            id=8,
            type="motor",
            damping=1.084,
            armature=0.045,
            kp=400.0,
            kv=40.0,
        ),
        "left_hip_pitch": MotorParameters(
            brand="dynamixel",
            id=9,
            type="motor",
            damping=1.084,
            armature=0.045,
            kp=400.0,
            kv=40.0,
        ),
        "left_knee": MotorParameters(
            brand="sunny_sky",
            id=1,
            type="motor",
            damping=1.084,
            armature=0.045,
            kp=400.0,
            kv=40.0,
        ),
        "left_ank_pitch": MotorParameters(
            brand="mighty_zap",
            id=1,
            type="motor",
            damping=1.084,
            armature=0.045,
            kp=400.0,
            kv=40.0,
        ),
        "left_ank_roll": MotorParameters(
            brand="mighty_zap",
            id=0,
            type="motor",
            damping=1.084,
            armature=0.045,
            kp=400.0,
            kv=40.0,
        ),
        "right_hip_yaw": MotorParameters(
            brand="dynamixel",
            id=10,
            type="motor",
            damping=1.084,
            armature=0.045,
            kp=400.0,
            kv=40.0,
        ),
        "right_hip_roll": MotorParameters(
            brand="dynamixel",
            id=11,
            type="motor",
            damping=1.084,
            armature=0.045,
            kp=400.0,
            kv=40.0,
        ),
        "right_hip_pitch": MotorParameters(
            brand="dynamixel",
            id=12,
            type="motor",
            damping=1.084,
            armature=0.045,
            kp=400.0,
            kv=40.0,
        ),
        "right_knee": MotorParameters(
            brand="sunny_sky",
            id=2,
            type="motor",
            damping=1.084,
            armature=0.045,
            kp=400.0,
            kv=40.0,
        ),
        "right_ank_pitch": MotorParameters(
            brand="mighty_zap",
            id=3,
            type="motor",
            damping=1.084,
            armature=0.045,
            kp=400.0,
            kv=40.0,
        ),
        "right_ank_roll": MotorParameters(
            brand="mighty_zap",
            id=2,
            type="motor",
            damping=1.084,
            armature=0.045,
            kp=400.0,
            kv=40.0,
        ),
        # "right_ank_act1": ActuatorParameters(type="motor", damping=0.1, armature=1e-7, kp=10.0, kv=0.1),
        # "right_ank_act2": ActuatorParameters(type="motor", damping=0.1, armature=1e-7, kp=10.0, kv=0.1),
        "left_sho_pitch": MotorParameters(
            brand="dynamixel",
            id=0,
            type="motor",
            damping=1.084,
            armature=0.045,
            kp=400.0,
            kv=40.0,
        ),
        "left_sho_roll": MotorParameters(
            brand="dynamixel",
            id=1,
            type="motor",
            damping=1.084,
            armature=0.045,
            kp=400.0,
            kv=40.0,
        ),
        "left_elb": MotorParameters(
            brand="dynamixel",
            id=2,
            type="motor",
            damping=1.084,
            armature=0.045,
            kp=400.0,
            kv=40.0,
        ),
        "right_sho_pitch": MotorParameters(
            brand="dynamixel",
            id=3,
            type="motor",
            damping=1.084,
            armature=0.045,
            kp=400.0,
            kv=40.0,
        ),
        "right_sho_roll": MotorParameters(
            brand="dynamixel",
            id=4,
            type="motor",
            damping=1.084,
            armature=0.045,
            kp=400.0,
            kv=40.0,
        ),
        "right_elb": MotorParameters(
            brand="dynamixel",
            id=5,
            type="motor",
            damping=1.084,
            armature=0.045,
            kp=400.0,
            kv=40.0,
        ),
    },
    constraint_pairs=[
        ("12lf_rod_end", "12lf_rod"),
        ("12lf_rod_end_2", "12lf_rod_2"),
        ("12lf_rod_end_3", "12lf_rod_3"),
        ("12lf_rod_end_4", "12lf_rod_4"),
    ],
    com=[-0.014, 0, 0.323],
    foot_size=[0.095, 0.03, 0.004],
    offsets={
        "z_offset_hip_roll_to_pitch": 0.024,  # from the hip roll joint to the hip pitch joint
        "z_offset_thigh": 0.098,  # from the hip pitch joint to the knee joint
        "z_offset_knee": 0.0,
        "z_offset_shin": 0.1,  # from the knee joint to the ankle roll joint
        "y_offset_com_to_foot": 0.035,  # from the hip center to the foot
        # Below are for the ankle IK
        # Implemented based on page 3 of the following paper:
        # http://link.springer.com/10.1007/978-3-319-93188-3_49
        # Notations are from the paper.
        "s1": [-0.0295, 0.0199, 0.085],
        "f1E": [-0.0371, 0.01, -0.0125],
        "nE": [1, 0, 0],
        "r": 0.01,
        "mighty_zap_len": 0.0752,
    },
    compute_leg_angles=compute_leg_angles,
)
