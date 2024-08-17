import math

import numpy as np

from toddlerbot.robot_descriptions.robot_configs import MotorParameters, RobotConfig

canonical_name2link_name = {"body_link": "base_link"}


def compute_leg_angles(target_foot_pos, target_foot_ori, side, offsets):
    # Decompose target position and orientation
    target_x, target_y, target_z = target_foot_pos
    ankle_roll, ankle_pitch, hip_yaw = target_foot_ori

    target_z = (
        offsets["hip_pitch_to_knee_z"]
        + offsets["z_offset_knee"]
        + offsets["knee_to_ank_roll_z"]
        - target_z
    )

    transformed_x = target_x * math.cos(hip_yaw) + target_y * math.sin(hip_yaw)
    transformed_y = -target_x * math.sin(hip_yaw) + target_y * math.cos(hip_yaw)
    transformed_z = target_z

    hip_roll = math.atan2(transformed_y, transformed_z + offsets["hip_roll_to_pitch_z"])

    adjusted_leg_height_sq = transformed_y**2 + transformed_z**2 - transformed_x**2
    adjusted_leg_height = (
        math.sqrt(max(0.0, adjusted_leg_height_sq)) - offsets["z_offset_knee"]
    )
    leg_pitch = math.atan2(transformed_x, adjusted_leg_height)
    leg_length = math.sqrt(transformed_x**2 + adjusted_leg_height**2)
    knee_disp_cos = leg_length / (
        offsets["hip_pitch_to_knee_z"] + offsets["knee_to_ank_roll_z"]
    )
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
    canonical_name2link_name=canonical_name2link_name,
    motor_params={
        "left_waist_yaw_joint": MotorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_waist_roll_joint": MotorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_waist_roll_mimic_joint": MotorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_waist_pitch_joint": MotorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_knee_pitch_mimic_joint": MotorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_waist_pitch_mimic_joint": MotorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_knee_pitch_joint": MotorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_ankle_pitch_mimic_joint": MotorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_shin_pitch_mimic_joint": MotorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_ankle_pitch_joint": MotorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "left_ankle_roll_joint": MotorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_waist_yaw_joint": MotorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_waist_roll_joint": MotorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_waist_roll_mimic_joint": MotorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_waist_pitch_joint": MotorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_knee_pitch_mimic_joint": MotorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_waist_pitch_mimic_joint": MotorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_knee_pitch_joint": MotorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_ankle_pitch_mimic_joint": MotorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_shin_pitch_mimic_joint": MotorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_ankle_pitch_joint": MotorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
        "right_ankle_roll_joint": MotorParameters(
            type="motor", damping=1.084, armature=0.045, kp=100.0, kv=10.0
        ),
    },
    com=[0, 0, 0.3],
    offsets={
        "hip_roll_to_pitch_z": 0.0,
        "hip_pitch_to_knee_z": 0.1,
        "z_offset_knee": 0.057,
        "knee_to_ank_roll_z": 0.1,
        "foot_to_com_y": 0.044,
    },
    compute_leg_angles=compute_leg_angles,
)
