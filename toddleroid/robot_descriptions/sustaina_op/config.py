import numpy as np

from toddleroid.robot_descriptions.robot_configs import ActuatorParameters, RobotConfig

canonical_name2link_name = {
    "body_link": "base_link",
    "left_foot_link": "left_foot_link",
    "right_foot_link": "right_foot_link",
}

link_name2canonical_name = {v: k for k, v in canonical_name2link_name.items()}


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
        "x_offset_foot_to_ankle": 0.0,
        "y_offset_foot_to_ankle": 0.044,
        "left_offset_foot_to_sole": np.array([0.0, 0.01, -0.04]),
        "right_offset_foot_to_sole": np.array([0.0, -0.01, -0.04]),
    },
)
