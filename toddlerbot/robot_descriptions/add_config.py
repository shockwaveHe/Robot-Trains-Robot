import argparse
import json
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict

# TODO: Convert to CSV and upload to google sheet with python
# TODO: Verify that the XC330 is well system identified


def get_default_config(
    root: ET.Element,
    general_config: Dict[str, Any],
    motor_config: Dict[str, Dict[str, Any]],
    joint_dyn_config: Dict[str, Dict[str, float]],
    kp: float = 2400.0,
    kd: float = 2400.0,
):
    # Define the URDF file path
    config_dict: Dict[str, Dict[str, Any]] = {"general": general_config}

    is_waist_closed_loop = False
    is_knee_closed_loop = False
    is_ankle_closed_loop = False
    config_dict["joints"] = {}
    id = 0
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
            lower_limit = float(joint_limit.get("lower"))  # type: ignore
            upper_limit = float(joint_limit.get("upper"))  # type: ignore

        is_passive = False
        transmission = "none"
        if "drive" in joint_name:
            transmission = "gears"
            if "driven" in joint_name:
                is_passive = True

        if "waist" in joint_name:
            is_waist_closed_loop = True
            transmission = "waist"
            if "act" not in joint_name:
                is_passive = True

        if "knee" in joint_name:
            is_knee_closed_loop = True
            transmission = "knee"
            if "act" not in joint_name:
                is_passive = True

        if "ank" in joint_name:
            is_ankle_closed_loop = True
            transmission = "ankle"
            if "act" not in joint_name:
                is_passive = True

        group = "default"
        upper_body_keywords = ["neck", "waist", "sho", "elbow", "wrist", "gripper"]
        lower_body_keywords = ["hip", "knee", "ank"]
        for keyword in upper_body_keywords:
            if keyword in joint_name:
                group = "upper_body"
                break
        for keyword in lower_body_keywords:
            if keyword in joint_name:
                group = "lower_body"
                break

        joint_dict = {
            "is_passive": is_passive,
            "transmission": transmission,
            "group": group,
            "default_pos": 0.0,
            "lower_limit": lower_limit,
            "upper_limit": upper_limit,
            "damping": 0.0,
            "armature": 0.0,
            "frictionloss": 0.0,
        }

        if is_passive:
            if transmission == "gears":
                # TODO: Make sure the gear ratio doesn't amplify the dynamics paramters
                joint_drive_name = joint_name.replace("_driven", "_drive")
                motor_name = motor_config[joint_drive_name]["motor"]
                for param_name in ["damping", "armature", "frictionloss"]:
                    joint_dict[param_name] = joint_dyn_config[motor_name][param_name]
            else:
                pass
                # TODO: sysID the joint dynamics params for passive joints
        else:
            if joint_name not in motor_config:
                raise ValueError(f"{joint_name} not found in the motor config!")

            motor_name = motor_config[joint_name]["motor"]
            joint_dict["id"] = list(motor_config.keys()).index(joint_name)
            joint_dict["type"] = "dynamixel"
            joint_dict["spec"] = motor_name
            joint_dict["control_mode"] = (
                "current_based_position"
                if motor_name == "XC330"  # or motor_name == "XM430"
                else "extended_position"
            )
            # joint_dict["control_mode"] = "extended_position"
            joint_dict["init_pos"] = (
                motor_config[joint_name]["init_pos"]
                if "init_pos" in motor_config[joint_name]
                else 0.0
            )
            joint_dict["kp_real"] = kp
            joint_dict["ki_real"] = 0.0
            joint_dict["kd_real"] = kd
            joint_dict["kff2_real"] = 0.0
            joint_dict["kff1_real"] = 0.0
            joint_dict["kp_sim"] = kp / 128
            joint_dict["kd_sim"] = 0.0

            if motor_name in joint_dyn_config:
                for param_name in ["damping", "armature", "frictionloss"]:
                    joint_dict[param_name] = joint_dyn_config[motor_name][param_name]

            if transmission == "gears":
                if "gear_ratio" in motor_config[joint_name]:
                    joint_dict["gear_ratio"] = motor_config[joint_name]["gear_ratio"]
                else:
                    joint_dict["gear_ratio"] = 1.0

            id += 1

        config_dict["joints"][joint_name] = joint_dict

    config_dict["general"]["is_waist_closed_loop"] = is_waist_closed_loop
    config_dict["general"]["is_knee_closed_loop"] = is_knee_closed_loop
    config_dict["general"]["is_ankle_closed_loop"] = is_ankle_closed_loop

    return config_dict


def main():
    parser = argparse.ArgumentParser(description="Get the config.")
    parser.add_argument(
        "--robot-name",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    args = parser.parse_args()

    robot_dir = os.path.join("toddlerbot", "robot_descriptions", args.robot_name)

    if "sysID" in args.robot_name:
        general_config = {
            "is_fixed": True,
            "use_torso_site": False,
            "has_imu": False,
            "has_dynamixel": True,
            "dynamixel_baudrate": 4000000,
            "has_sunny_sky": False,
        }
    else:
        general_config = {
            "is_fixed": False,
            "use_torso_site": True,
            "has_imu": True,
            "has_dynamixel": True,
            "dynamixel_baudrate": 4000000,
            "has_sunny_sky": False,
            "offsets": {
                "torso_z": 0.3442,
                "waist_roll_coef": 0.29166667,
                "waist_yaw_coef": 0.41666667,
                "ank_act_arm_y": 0.00582666,
                "ank_act_arm_r": 0.02,
                "ank_long_rod_len": 0.05900847,
                "ank_short_rod_len": 0.03951266,
                "ank_rev_r": 0.01,
            },
        }

    # This one needs to be ORDERED
    motor_config_path = os.path.join(robot_dir, "config_motors.json")
    if os.path.exists(motor_config_path):
        with open(motor_config_path, "r") as f:
            motor_config = json.load(f)
    elif "sysID" in args.robot_name:
        motor_name = args.robot_name.split("_")[-1]
        motor_config = {"joint_0": {"motor": motor_name, "init_pos": 0.0}}
    else:
        motor_config: Dict[str, Dict[str, Any]] = {}

    joint_dyn_config: Dict[str, Dict[str, float]] = {}
    if "sysID" not in args.robot_name:
        motor_list = [motor_config["motor"] for motor_config in motor_config.values()]
        for motor_name in motor_list:
            sysID_result_path = os.path.join(
                "toddlerbot", "robot_descriptions", f"sysID_{motor_name}", "config.json"
            )
            with open(sysID_result_path, "r") as f:
                sysID_result = json.load(f)

            joint_dyn_config[motor_name] = {}
            for param_name in ["damping", "armature", "frictionloss"]:
                joint_dyn_config[motor_name][param_name] = sysID_result["joints"][
                    "joint_0"
                ][param_name]

    urdf_path = os.path.join(robot_dir, f"{args.robot_name}.urdf")
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    config_dict = get_default_config(
        root, general_config, motor_config, joint_dyn_config
    )

    config_file_path = os.path.join(robot_dir, "config.json")
    with open(config_file_path, "w") as f:
        f.write(json.dumps(config_dict, indent=4))

    collision_config_file_path = os.path.join(robot_dir, "config_collision.json")
    if not os.path.exists(collision_config_file_path):
        collision_config = {}
        for link in root.findall("link"):
            link_name = link.get("name")

            if link_name is None:
                continue

            collision_config[link_name] = {"has_collision": False}

        with open(collision_config_file_path, "w") as f:
            f.write(json.dumps(collision_config, indent=4))


if __name__ == "__main__":
    main()
