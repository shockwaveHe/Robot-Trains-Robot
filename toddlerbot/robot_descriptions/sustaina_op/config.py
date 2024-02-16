import math

import numpy as np

from toddlerbot.robot_descriptions.robot_configs import ActuatorParameters, RobotConfig

canonical_name2link_name = {
    "body_link": "base_link",
    "left_foot_link": "left_foot_link",
    "right_foot_link": "right_foot_link",
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

    adjusted_leg_height_sq = transformed_y**2 + transformed_z**2 - transformed_x**2
    adjusted_leg_height = (
        math.sqrt(max(0.0, adjusted_leg_height_sq)) - offsets["z_offset_knee"]
    )
    leg_pitch = math.atan2(transformed_x, adjusted_leg_height)
    leg_length = math.sqrt(transformed_x**2 + adjusted_leg_height**2)
    knee_disp_cos = leg_length / (offsets["z_offset_thigh"] + offsets["z_offset_shin"])
    knee_disp = math.acos(min(max(-1.0, knee_disp_cos), 1.0))
    hip_pitch = -leg_pitch - knee_disp
    knee_pitch = -leg_pitch + knee_disp

    angles_dict = {
        "waist_yaw_joint": hip_yaw,
        "waist_roll_joint": hip_roll,
        "waist_pitch_joint": hip_pitch,
        "knee_pitch_mimic_joint": -hip_pitch,
        "waist_pitch_mimic_joint": hip_pitch,
        "knee_pitch_joint": knee_pitch,
        "ankle_pitch_mimic_joint": -knee_pitch,
        "shin_pitch_mimic_joint": knee_pitch,
        "ankle_pitch_joint": ankle_pitch,
        "ankle_roll_joint": ankle_roll - hip_roll,
    }
    return angles_dict


sustaina_op_config = RobotConfig(
    com_height=0.3,
    canonical_name2link_name=canonical_name2link_name,
    link_name2canonical_name=link_name2canonical_name,
    act_params={
        "left_waist_yaw_joint": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_waist_roll_joint": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_waist_roll_mimic_joint": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_waist_pitch_joint": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_knee_pitch_mimic_joint": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_waist_pitch_mimic_joint": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_knee_pitch_joint": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_ankle_pitch_mimic_joint": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_shin_pitch_mimic_joint": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_ankle_pitch_joint": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_ankle_roll_joint": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_waist_yaw_joint": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_waist_roll_joint": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_waist_roll_mimic_joint": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_waist_pitch_joint": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_knee_pitch_mimic_joint": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_waist_pitch_mimic_joint": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_knee_pitch_joint": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_ankle_pitch_mimic_joint": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_shin_pitch_mimic_joint": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_ankle_pitch_joint": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_ankle_roll_joint": ActuatorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
    },
    offsets={
        "z_offset_hip": 0.053,
        "z_offset_thigh": 0.1,
        "z_offset_knee": 0.057,
        "z_offset_shin": 0.1,
        "x_offset_ankle_to_foot": 0.0,
        "y_offset_ankle_to_foot": 0.044,
        "left_offset_foot_to_sole": np.array([0.0, 0.01, -0.04]),
        "right_offset_foot_to_sole": np.array([0.0, -0.01, -0.04]),
    },
    compute_leg_angles=compute_leg_angles,
)
