import math

import numpy as np

from toddlerbot.robot_descriptions.robot_configs import ActuatorParameters, RobotConfig

canonical_name2link_name = {
    "body_link": "body_link",
    "left_foot_link": "l_ank_roll_link",
    "right_foot_link": "r_ank_roll_link",
}

link_name2canonical_name = {v: k for k, v in canonical_name2link_name.items()}


def compute_leg_angles(target_foot_pos, target_foot_ori, side, offsets):
    # Decompose target position and orientation
    target_x, target_y, target_z = target_foot_pos
    ankle_roll, ankle_pitch, hip_yaw = target_foot_ori

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
        "hip_roll": -hip_roll,
        "hip_pitch": hip_pitch if side == "left" else -hip_pitch,
        "knee": knee_pitch if side == "left" else -knee_pitch,
        "ank_pitch": ankle_pitch if side == "left" else -ankle_pitch,
        "ank_roll": ankle_roll - hip_roll,
    }
    return angles_dict


robotis_op3_config = RobotConfig(
    canonical_name2link_name=canonical_name2link_name,
    link_name2canonical_name=link_name2canonical_name,
    act_params={
        "l_hip_yaw": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "l_hip_roll": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "l_hip_pitch": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "l_knee": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "l_ank_pitch": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "l_ank_roll": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "r_hip_yaw": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "r_hip_roll": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "r_hip_pitch": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "r_knee": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "r_ank_pitch": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "r_ank_roll": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "l_sho_pitch": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "l_sho_roll": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "l_el": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "r_sho_pitch": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "r_sho_roll": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "r_el": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "head_pan": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "head_tilt": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
    },
    com_z=0.3,
    foot_size=[0.127, 0.076, 0.002],
    offsets={
        "z_offset_hip_roll_to_pitch": 0.0,
        "z_offset_thigh": 0.11,  # from the hip pitch joint to the knee joint
        "z_offset_knee": 0.0,
        "z_offset_shin": 0.11,  # from the knee joint to the ankle pitch joint
        "y_offset_com_to_foot": 0.044,
    },
    compute_leg_angles=compute_leg_angles,
)
