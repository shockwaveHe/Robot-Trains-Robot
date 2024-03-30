import argparse
import os
import textwrap
import xml.etree.ElementTree as ET


def find_root_link_name(root):
    child_links = {joint.find("child").get("link") for joint in root.findall("joint")}
    all_links = {link.get("name") for link in root.findall("link")}

    # The root link is the one not listed as a child
    root_link = all_links - child_links
    if root_link:
        return root_link.pop()
    else:
        raise ValueError("Could not find root link in URDF")


def get_config(robot_name):
    robot_dir = os.path.join("toddlerbot", "robot_descriptions", robot_name)
    urdf_path = os.path.join(robot_dir, f"{robot_name}.urdf")
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    body_link_name = find_root_link_name(root)

    # Define the content of the Python script as a multi-line string
    script_content = textwrap.dedent(
        f"""\
    import math

    import numpy as np

    from toddlerbot.robot_descriptions.robot_configs import *

    canonical_name2link_name = {{"body_link": "{body_link_name}"}}


    # UPDATE: the function to compute leg angles
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

        angles_dict = {{
            "hip_yaw": hip_yaw,
            "hip_roll": hip_roll if side == "left" else -hip_roll,
            "hip_pitch": -hip_pitch if side == "left" else hip_pitch,
            "knee": knee_pitch if side == "left" else -knee_pitch,
            "ank_pitch": ankle_pitch if side == "left" else -ankle_pitch,
            "ank_roll": ankle_roll - hip_roll,
        }}
        return angles_dict


    {robot_name}_config = RobotConfig(
        canonical_name2link_name=canonical_name2link_name,
        # UPDATE: the motor parameters for the robot
        motor_params={{
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
        }},
        # UPDATE: the constraint pairs for the robot
        constraint_pairs=[
            ("12lf_rod_end", "12lf_rod"),
            ("12lf_rod_end_2", "12lf_rod_2"),
            ("12lf_rod_end_3", "12lf_rod_3"),
            ("12lf_rod_end_4", "12lf_rod_4"),
        ],
        compute_leg_angles=compute_leg_angles,
    )"""
    )

    # Define the file path
    config_path = os.path.join(robot_dir, "config.py")

    # Write the content to the file
    with open(config_path, "w") as file:
        file.write(script_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the config.")
    parser.add_argument(
        "--robot-name",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    args = parser.parse_args()

    get_config(args.robot_name)
