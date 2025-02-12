import argparse
import json
import os
import platform
import xml.etree.ElementTree as ET
from typing import Any, Dict

import numpy as np


def get_default_config(
    robot_name: str,
    root: ET.Element,
    general_config: Dict[str, Any],
    motor_config: Dict[str, Dict[str, Any]],
    joint_dyn_config: Dict[str, Dict[str, float]],
):
    """Generates a default configuration dictionary for a robot based on its URDF structure and provided configurations.

    Args:
        robot_name (str): The name of the robot, used to determine specific configurations.
        root (ET.Element): The root element of the robot's URDF XML structure.
        general_config (Dict[str, Any]): General configuration settings for the robot.
        motor_config (Dict[str, Dict[str, Any]]): Configuration settings for each motor, including control parameters.
        joint_dyn_config (Dict[str, Dict[str, float]]): Dynamic configuration settings for each joint, such as damping and friction.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary containing the default configuration for the robot, including general settings and detailed joint configurations.
    """
    config_dict: Dict[str, Dict[str, Any]] = {"general": general_config, "joints": {}}

    # Define the URDF file path
    is_neck_closed_loop = False
    is_waist_closed_loop = False
    is_knee_closed_loop = False
    is_ankle_closed_loop = False
    for joint in root.findall("joint"):
        joint_name = joint.get("name")
        joint_type = joint.get("type")
        if joint_name is None or joint_type is None:
            continue

        if "neck" in joint_name and "act" in joint_name and "fixed" not in joint_type:
            is_neck_closed_loop = True
        if "waist" in joint_name and "act" in joint_name and "fixed" not in joint_type:
            is_waist_closed_loop = True
        if "knee" in joint_name and "act" in joint_name and "fixed" not in joint_type:
            is_knee_closed_loop = True
        if "ank" in joint_name and "act" in joint_name and "fixed" not in joint_type:
            is_ankle_closed_loop = True

    config_dict["general"]["is_neck_closed_loop"] = is_neck_closed_loop
    config_dict["general"]["is_waist_closed_loop"] = is_waist_closed_loop
    config_dict["general"]["is_knee_closed_loop"] = is_knee_closed_loop
    config_dict["general"]["is_ankle_closed_loop"] = is_ankle_closed_loop

    if is_waist_closed_loop:
        config_dict["general"]["waist_roll_backlash"] = 0.03
        config_dict["general"]["waist_yaw_backlash"] = 0.001
        config_dict["general"]["offsets"]["waist_roll_coef"] = 0.29166667
        config_dict["general"]["offsets"]["waist_yaw_coef"] = 0.20833333

    if is_ankle_closed_loop:
        config_dict["general"]["ank_solimp_0"] = 0.9999
        config_dict["general"]["ank_solref_0"] = 0.004
        config_dict["general"]["offsets"]["ank_act_arm_y"] = 0.00582666
        config_dict["general"]["offsets"]["ank_act_arm_r"] = 0.02
        config_dict["general"]["offsets"]["ank_long_rod_len"] = 0.05900847
        config_dict["general"]["offsets"]["ank_short_rod_len"] = 0.03951266
        config_dict["general"]["offsets"]["ank_rev_r"] = 0.01

    # toddlerbot_arm joints should start with id 16
    if "arms" in robot_name:
        init_id = 16
    else:
        init_id = 0

    for joint in root.findall("joint"):
        joint_type = joint.get("type")
        if joint_type is None or joint_type == "fixed":
            continue

        joint_name = joint.get("name")
        if joint_name is None:
            continue

        joint_limit = joint.find("limit")
        if joint_limit is None:
            raise ValueError(f"Joint {joint_name} does not have a limit tag.")
        else:
            lower_limit = float(joint_limit.get("lower", -np.pi))
            upper_limit = float(joint_limit.get("upper", np.pi))

        is_passive = False
        transmission = "none"

        if "drive" in joint_name:
            transmission = "gear"
            if "driven" in joint_name:
                is_passive = True

        elif "neck" in joint_name and is_neck_closed_loop:
            transmission = "linkage"
            if "act" not in joint_name:
                is_passive = True

        elif "waist" in joint_name and is_waist_closed_loop:
            transmission = "waist"
            if "act" not in joint_name:
                is_passive = True

        elif "knee" in joint_name and is_knee_closed_loop:
            transmission = "linkage"
            if "act" not in joint_name:
                is_passive = True

        elif "ank" in joint_name and is_ankle_closed_loop:
            transmission = "ankle"
            if "act" not in joint_name:
                is_passive = True

        elif "gripper" in joint_name:
            transmission = "rack_and_pinion"
            if "pinion" in joint_name:
                is_passive = True

        group = "none"
        group_keywords = {
            "neck": ["neck"],
            "waist": ["waist"],
            "arm": ["sho", "elbow", "wrist", "gripper"],
            "leg": ["hip", "knee", "ank"],
        }
        for name, keywords in group_keywords.items():
            if any(keyword in joint_name for keyword in keywords):
                group = name
                break

        joint_dict = {
            "is_passive": is_passive,
            "transmission": transmission,
            "group": group,
            "lower_limit": lower_limit,
            "upper_limit": upper_limit,
            "damping": 0.0,
            "armature": 0.0,
            "frictionloss": 0.0,
        }

        if is_passive:
            if joint_name in joint_dyn_config:
                for param_name in joint_dyn_config[joint_name]:
                    joint_dict[param_name] = joint_dyn_config[joint_name][param_name]
        else:
            if joint_name not in motor_config:
                raise ValueError(f"{joint_name} not found in the motor config!")

            motor_name = motor_config[joint_name]["motor"]
            joint_dict["id"] = list(motor_config.keys()).index(joint_name) + init_id
            joint_dict["type"] = "dynamixel"
            joint_dict["spec"] = motor_name
            # joint_dict["control_mode"] = "extended_position"
            joint_dict["control_mode"] = (
                "current_based_position"
                if "gripper" in joint_name
                else "extended_position"
            )
            joint_dict["init_pos"] = (
                motor_config[joint_name]["init_pos"]
                if "init_pos" in motor_config[joint_name]
                else 0.0
            )
            joint_dict["default_pos"] = (
                motor_config[joint_name]["default_pos"]
                if "default_pos" in motor_config[joint_name]
                else 0.0
            )
            joint_dict["kp_real"] = motor_config[joint_name]["kp"]
            joint_dict["ki_real"] = motor_config[joint_name]["ki"]
            joint_dict["kd_real"] = motor_config[joint_name]["kd"]
            joint_dict["kff2_real"] = motor_config[joint_name]["kff2"]
            joint_dict["kff1_real"] = motor_config[joint_name]["kff1"]
            joint_dict["kp_sim"] = motor_config[joint_name]["kp"] / 128
            joint_dict["kd_sim"] = 0.0

            if motor_name in joint_dyn_config:
                for param_name in joint_dyn_config[motor_name]:
                    joint_dict[param_name] = joint_dyn_config[motor_name][param_name]
            elif joint_name in joint_dyn_config:
                for param_name in joint_dyn_config[joint_name]:
                    joint_dict[param_name] = joint_dyn_config[joint_name][param_name]

            if transmission == "gear" or transmission == "rack_and_pinion":
                if "gear_ratio" in motor_config[joint_name]:
                    joint_dict["gear_ratio"] = motor_config[joint_name]["gear_ratio"]
                else:
                    joint_dict["gear_ratio"] = 1.0

        config_dict["joints"][joint_name] = joint_dict

    joints_list = list(config_dict["joints"].items())

    # Sort the list of joints first by id (if exists, otherwise use a large number) and then by name
    sorted_joints_list = sorted(
        joints_list, key=lambda item: (item[1].get("id", float("inf")), item[0])
    )

    # Create a new ordered dictionary from the sorted list
    config_dict["joints"] = dict(sorted_joints_list)

    return config_dict


def main() -> None:
    """Main function to generate and save configuration files for a specified robot.

    This function parses command-line arguments to determine the robot's name and sets up
    various configuration parameters based on the robot's characteristics. It generates
    general, motor, joint dynamics, and collision configurations, saving them to JSON files
    in the appropriate directories.

    Raises:
        ValueError: If the motor configuration file is not found for the specified robot.
    """
    parser = argparse.ArgumentParser(description="Get the config.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    # baud is either 3 or 4 int
    # parser.add_argument(
    #     "--baud",
    #     type=int,
    #     default=4,
    #     help="The baudrate of motors, unit in Mbps",
    # )
    args = parser.parse_args()

    # if macos, use 3
    if platform.system() == "Darwin":
        baud = 2
    else:
        baud = 2

    robot_dir = os.path.join("toddlerbot", "descriptions", args.robot)
    general_config: Dict[str, Any] = {
        "is_fixed": True,
        "has_imu": False,
        "has_dynamixel": True,
        "dynamixel_baudrate": baud * 1000000,
        "solref": [0.004, 1],
    }
    if "sysID" not in args.robot and "arms" not in args.robot:
        general_config["is_fixed"] = False
        general_config["has_imu"] = True
        if "gripper" in args.robot:
            general_config["ee_name"] = "gripper_connector"
        else:
            general_config["ee_name"] = "hand"
        general_config["foot_name"] = "ank_roll_link"
        general_config["offsets"] = {
            "torso_z": 0.33605,
            "default_torso_z": 0.3267,
            "foot_to_com_x": 0.0027,
            "foot_to_com_y": 0.0364,
            "hip_roll_to_pitch_z": 0.024,
            "hip_pitch_to_knee_z": 0.096,
            "knee_to_ank_pitch_z": 0.1,
            # "imu_x": 0.0282,
            # "imu_y": 0.0,
            # "imu_z": 0.105483,
            # "imu_zaxis": "-1 0 0",
        }

    # if general_config["has_imu"]:
    #     imu_config_path = os.path.join(robot_dir, "config_imu.json")
    #     if os.path.exists(imu_config_path):
    #         with open(imu_config_path, "r") as f:
    #             general_config["imu"] = json.load(f)
    #     else:
    #         raise ValueError(f"{imu_config_path} not found!")

    # This one needs to be ORDERED
    motor_config_path = os.path.join(robot_dir, "config_motors.json")
    if os.path.exists(motor_config_path):
        with open(motor_config_path, "r") as f:
            motor_config = json.load(f)
    elif "sysID" in args.robot:
        motor_config = {
            "joint_0": {"motor": str(args.robot.split("_")[-1]), "init_pos": 0.0}
        }
    else:
        raise ValueError(f"{motor_config_path} not found!")

    joint_dyn_config: Dict[str, Dict[str, float]] = {}
    if "sysID" not in args.robot:
        motor_name_list = [
            str(motor_config["motor"]) for motor_config in motor_config.values()
        ]
        for motor_name in motor_name_list:
            sysID_result_path = os.path.join(
                "toddlerbot",
                "descriptions",
                f"sysID_{motor_name}",
                "config_dynamics.json",
            )
            with open(sysID_result_path, "r") as f:
                sysID_result = json.load(f)

            joint_dyn_config[motor_name] = sysID_result["joint_0"]

    dynamics_config_path = os.path.join(robot_dir, "config_dynamics.json")
    if os.path.exists(dynamics_config_path):
        with open(dynamics_config_path, "r") as f:
            passive_joint_dyn_config = json.load(f)

        for joint_name, joint_config in passive_joint_dyn_config.items():
            joint_dyn_config[joint_name] = joint_config

    urdf_path = os.path.join(robot_dir, f"{args.robot}.urdf")
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    config_dict = get_default_config(
        args.robot, root, general_config, motor_config, joint_dyn_config
    )

    config_file_path = os.path.join(robot_dir, "config.json")
    with open(config_file_path, "w") as f:
        f.write(json.dumps(config_dict, indent=4))
        print(f"Config file saved to {config_file_path}")

    collision_config_file_path = os.path.join(robot_dir, "config_collision.json")
    if not os.path.exists(collision_config_file_path):
        collision_config = {}
    else:
        with open(collision_config_file_path, "r") as f:
            collision_config = json.load(f)

    link_names = []
    for link in root.findall("link"):
        link_name = link.get("name")
        link_names.append(link_name)

        if link_name is not None and link_name not in collision_config:
            collision_config[link_name] = {"has_collision": False}

    collision_config_updated = {
        link_name: value
        for link_name, value in collision_config.items()
        if link_name in link_names
    }

    with open(collision_config_file_path, "w") as f:
        f.write(json.dumps(collision_config_updated, indent=4))
        print(f"Collision config file saved to {collision_config_file_path}")


if __name__ == "__main__":
    main()
