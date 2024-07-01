import argparse
import json
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict


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
    for id, joint in enumerate(root.findall("joint")):
        joint_name = joint.get("name")

        joint_limit = joint.find("limit")
        if joint_limit is None:
            raise ValueError(f"Joint {joint_name} does not have a limit tag.")
        else:
            lower_limit = float(joint_limit.get("lower"))  # type: ignore
            upper_limit = float(joint_limit.get("upper"))  # type: ignore

        if joint_name is None:
            continue

        joint_dict = {
            "id": id,
            "group": "default",
            "type": "dynamixel",
            "control_mode": "position",
            "is_indirect": False,
            "has_closed_loop": False,
            "damping": 0.0,
            "armature": 0.0,
            "frictionloss": 0.0,
            "kp_real": kp,
            "ki_real": 0.0,
            "kd_real": kd,
            "kp_sim": kp / 128,
            "kd_sim": 0.0,  # TODO: Try kd / 16
            "kff2_real": 0.0,
            "kff1_real": 0.0,
            "gear_ratio": 1.0,
            "init_pos": 0.0,
            "default_pos": 0.0,
            "lower_limit": lower_limit,
            "upper_limit": upper_limit,
        }
        config_dict["joints"][joint_name] = joint_dict

    config_dict["links"] = {}
    for link in root.findall("link"):
        link_name = link.get("name")

        if link_name is None:
            continue

        link_dict = {
            "has_collision": True,
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
