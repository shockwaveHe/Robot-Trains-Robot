import numpy as np

from toddleroid.robot_descriptions.robot_configs import ActuatorParameters, RobotConfig

canonical_name2link_name = {
    "body_link": "body_link",
    "left_foot_link": "l_ank_roll_link",
    "right_foot_link": "r_ank_roll_link",
}

link_name2canonical_name = {v: k for k, v in canonical_name2link_name.items()}


robotis_op3_config = RobotConfig(
    com_height=0.3,
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
    offsets={
        "z_offset_hip": 0.028,
        "z_offset_thigh": 0.11,  # from the hip pitch joint to the knee joint
        "z_offset_knee": 0.0,
        "z_offset_shin": 0.11,  # from the knee joint to the ankle pitch joint
        "x_offset_ankle_to_foot": 0.0,
        "y_offset_ankle_to_foot": 0.044,
        "left_offset_foot_to_sole": np.array([0.0, 0.01, 0.0]),
        "right_offset_foot_to_sole": np.array([0.0, -0.01, 0.0]),
        "x_offset_sole": 0.127,
        "y_offset_sole": 0.076,
        "z_offset_sole": 0.002,
    },
)
