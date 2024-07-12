import argparse
import json
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict

# TODO: Update the ids of the dynamixels
# TODO: Use sysID results to update damping, armature, frictionloss
# TODO: Convert to CSV and upload to google sheet with python
# TODO: Double check default_pos


def get_default_config(root: ET.Element, kp: float = 2400.0, kd: float = 2400.0):
    # Define the URDF file path
    config_dict: Dict[str, Dict[str, Any]] = {}

    config_dict["general"] = {
        "is_fixed": False,
        "use_torso_site": False,
        "has_imu": False,
        "constraint_pairs": [],
        "has_dynamixel": True,
        "has_sunny_sky": False,
        "dynamixel_baudrate": 4000000,
    }

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
        if "driven" in joint_name:
            is_passive = True

        is_closed_loop = False
        if "waist" in joint_name or "ank" in joint_name:
            is_closed_loop = True
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
            "group": group,
            "is_closed_loop": is_closed_loop,
            "damping": 0.0,
            "armature": 0.0,
            "frictionloss": 0.0,
            "default_pos": 0.0,
            "lower_limit": lower_limit,
            "upper_limit": upper_limit,
        }

        if is_passive:
            joint_dict["gear_ratio"] = 1.0
        else:
            joint_dict["id"] = id
            joint_dict["type"] = "dynamixel"
            joint_dict["spec"] = "XC430"
            joint_dict["control_mode"] = "position"
            joint_dict["init_pos"] = 0.0
            joint_dict["kp_real"] = kp
            joint_dict["ki_real"] = 0.0
            joint_dict["kd_real"] = kd
            joint_dict["kff2_real"] = 0.0
            joint_dict["kff1_real"] = 0.0
            joint_dict["kp_sim"] = kp / 128
            joint_dict["kd_sim"] = 0.0

            id += 1

        config_dict["joints"][joint_name] = joint_dict

    config_dict["links"] = {}
    for link in root.findall("link"):
        link_name = link.get("name")

        if link_name is None:
            continue

        link_dict = {
            "has_collision": False,
            "collision_type": "box",  # box, mesh
            "collision_scale": [1.0, 1.0, 1.0],
        }
        config_dict["links"][link_name] = link_dict

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
    urdf_path = os.path.join(robot_dir, f"{args.robot_name}.urdf")
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    config_dict = get_default_config(root)
    config_file_path = os.path.join(robot_dir, "config.json")
    with open(config_file_path, "w") as f:
        f.write(json.dumps(config_dict, indent=4))


if __name__ == "__main__":
    main()
