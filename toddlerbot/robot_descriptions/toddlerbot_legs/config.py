import math

from toddlerbot.robot_descriptions.robot_configs import MotorParameters, RobotConfig

canonical_name2link_name = {
    "body_link": "body_link",
    "knee_link": "calf_link",
    "foot_link": "ank_roll_link",
}


# UPDATE: the function to compute leg angles
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

    hip_roll = math.atan2(target_y, target_z + offsets["z_offset_hip_roll_to_pitch"])

    leg_projected_yz_length = math.sqrt(target_y**2 + target_z**2)
    leg_length = math.sqrt(target_x**2 + leg_projected_yz_length**2)
    leg_pitch = math.atan2(target_x, leg_projected_yz_length)
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
        "hip_yaw": -hip_yaw,
        "hip_roll": hip_roll,
        "hip_pitch": -hip_pitch if side == "left" else hip_pitch,
        "knee": knee_pitch if side == "left" else -knee_pitch,
        "ank_pitch": ankle_pitch if side == "left" else -ankle_pitch,
        "ank_roll": ankle_roll - hip_roll,
    }
    return angles_dict


toddlerbot_legs_config = RobotConfig(
    canonical_name2link_name=canonical_name2link_name,
    # UPDATE: the motor parameters for the robot
    motor_params={
        "left_hip_yaw": MotorParameters(
            brand="dynamixel",
            id=7,
            type="motor",
            default_angle=0.0,
            damping=0.077,
            armature=0.0034,
            frictionloss=0.0434,
            kp=3.125,
            kv=0.0,
        ),
        "left_hip_roll": MotorParameters(
            brand="dynamixel",
            id=8,
            type="motor",
            default_angle=0.0,
            damping=3.122,
            armature=0.1,
            frictionloss=0.0002,
            kp=26.0,
            kv=0.0,
        ),
        "left_hip_pitch": MotorParameters(
            brand="dynamixel",
            id=9,
            type="motor",
            default_angle=0.325,
            damping=1.286,
            armature=0.0032,
            frictionloss=0.0,
            kp=13.0,
            kv=0.0,
        ),
        "left_knee": MotorParameters(
            brand="sunny_sky",
            id=1,
            type="motor",
            damping=1.142,
            default_angle=0.65,
            armature=0.0,
            frictionloss=0.0319,
            kp=40.0,
            kv=0.0,
        ),
        "left_ank_pitch": MotorParameters(
            brand="mighty_zap",
            id=0,
            type="motor",
            default_angle=0.325,
            damping=0.523,
            armature=0.0115,
            frictionloss=0.0005,
            kp=10.0,
            kv=0.0,
        ),
        "left_ank_roll": MotorParameters(
            brand="mighty_zap",
            id=1,
            type="motor",
            default_angle=0.0,
            damping=0.583,
            armature=0.0158,
            frictionloss=0.0003,
            kp=10.0,
            kv=0.0,
        ),
        "right_hip_yaw": MotorParameters(
            brand="dynamixel",
            id=10,
            type="motor",
            default_angle=0.0,
            damping=0.077,
            armature=0.0034,
            frictionloss=0.0434,
            kp=3.125,
            kv=0.0,
        ),
        "right_hip_roll": MotorParameters(
            brand="dynamixel",
            id=11,
            type="motor",
            default_angle=0.0,
            damping=3.122,
            armature=0.1,
            frictionloss=0.0002,
            kp=26.0,
            kv=0.0,
        ),
        "right_hip_pitch": MotorParameters(
            brand="dynamixel",
            id=12,
            type="motor",
            default_angle=-0.325,
            damping=1.286,
            armature=0.0032,
            frictionloss=0.0,
            kp=13.0,
            kv=0.0,
        ),
        "right_knee": MotorParameters(
            brand="sunny_sky",
            id=2,
            type="motor",
            default_angle=-0.65,
            damping=1.142,
            armature=0.0,
            frictionloss=0.0319,
            kp=40.0,
            kv=0.0,
        ),
        "right_ank_pitch": MotorParameters(
            brand="mighty_zap",
            id=2,
            type="motor",
            default_angle=-0.325,
            damping=0.523,
            armature=0.0115,
            frictionloss=0.0005,
            kp=10.0,
            kv=0.0,
        ),
        "right_ank_roll": MotorParameters(
            brand="mighty_zap",
            id=3,
            type="motor",
            default_angle=0.0,
            damping=0.583,
            armature=0.0158,
            frictionloss=0.0003,
            kp=10.0,
            kv=0.0,
        ),
        # "right_ank_act1": ActuatorParameters(type="motor", damping=0.1, armature=1e-7, kp=10.0, kv=0.1),
        # "right_ank_act2": ActuatorParameters(type="motor", damping=0.1, armature=1e-7, kp=10.0, kv=0.1),
    },
    # UPDATE: the constraint pairs for the robot
    constraint_pairs=[
        ("12lf_rod_end", "12lf_rod"),
        ("12lf_rod_end_2", "12lf_rod_2"),
        ("12lf_rod_end_3", "12lf_rod_3"),
        ("12lf_rod_end_4", "12lf_rod_4"),
    ],
    compute_leg_angles=compute_leg_angles,
)
