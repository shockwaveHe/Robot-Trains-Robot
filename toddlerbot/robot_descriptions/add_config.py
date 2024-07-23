import argparse
import json
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict

# TODO: Convert to CSV and upload to google sheet with python


# This dict needs to be ORDERED
joint_motor_dict: Dict[str, str] = {
    "neck_yaw_drive": "XC330",
    "neck_pitch_drive": "XC330",
    "waist_act_1": "XC330",
    "waist_act_2": "XC330",
    "left_hip_yaw_drive": "XC330",
    "left_hip_roll": "2XC430",
    "left_hip_pitch": "2XC430",
    "left_knee": "XM430",
    "left_ank_act_1": "XC330",
    "left_ank_act_2": "XC330",
    "right_hip_yaw_drive": "XC330",
    "right_hip_roll": "2XC430",
    "right_hip_pitch": "2XC430",
    "right_knee": "XM430",
    "right_ank_act_1": "XC330",
    "right_ank_act_2": "XC330",
    "left_sho_pitch": "XC430",
    "left_sho_roll": "2XL430",
    "left_sho_yaw_drive": "2XL430",
    "left_elbow_roll": "2XL430",
    "left_elbow_yaw_drive": "2XL430",
    "left_wrist_pitch_drive": "2XL430",
    "left_wrist_roll": "2XL430",
    "right_sho_pitch": "XC430",
    "right_sho_roll": "2XL430",
    "right_sho_yaw_drive": "2XL430",
    "right_elbow_roll": "2XL430",
    "right_elbow_yaw_drive": "2XL430",
    "right_wrist_pitch_drive": "2XL430",
    "right_wrist_roll": "2XL430",
}

joint_gear_ratio_dict: Dict[str, float] = {
    "neck_yaw_driven": 1.3846153846153846,
    "neck_pitch_driven": 1.1,
    "waist_roll": 2.4,
    "waist_yaw": 2.4,
    "left_hip_yaw_driven": 1.105263157894737,
    "right_hip_yaw_driven": 1.105263157894737,
}

joint_dyn_params_dict: Dict[str, Dict[str, float]] = {}
motor_list = ["XC330", "XC430", "2XC430", "2XL430"]
for motor_name in motor_list:
    sysID_result_path = os.path.join(
        "toddlerbot", "robot_descriptions", f"sysID_{motor_name}", "config.json"
    )
    with open(sysID_result_path, "r") as f:
        sysID_result = json.load(f)

    joint_dyn_params_dict[motor_name] = {}
    for param_name in ["damping", "armature", "frictionloss"]:
        joint_dyn_params_dict[motor_name][param_name] = sysID_result["joints"][
            "joint_0"
        ][param_name]

joint_dyn_params_dict["XM430"] = {
    "damping": 1.084,
    "armature": 0.045,
    "frictionloss": 0.03,
}
joint_dyn_params_dict["left_ank_pitch"] = joint_dyn_params_dict["right_ank_pitch"] = (
    joint_dyn_params_dict["XC330"]
)
joint_dyn_params_dict["left_ank_roll"] = joint_dyn_params_dict["right_ank_roll"] = (
    joint_dyn_params_dict["XC330"]
)
joint_dyn_params_dict["waist_roll"] = joint_dyn_params_dict["XC330"]
joint_dyn_params_dict["waist_yaw"] = joint_dyn_params_dict["XC330"]


def get_default_config(root: ET.Element, kp: float = 2400.0, kd: float = 2400.0):
    # Define the URDF file path
    config_dict: Dict[str, Dict[str, Any]] = {}

    config_dict["general"] = {
        "is_fixed": False,
        "use_torso_site": True,
        "has_imu": True,
        "has_dynamixel": True,
        "dynamixel_baudrate": 4000000,
        "has_sunny_sky": False,
    }

    has_waist = False
    has_ankle = False
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
        if "driven" in joint_name:
            is_passive = True
            transmission = "gears"

        if "waist" in joint_name:
            has_waist = True
            transmission = "waist"
            if "act" not in joint_name:
                is_passive = True

        if "ank" in joint_name:
            has_ankle = True
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
            "transimission": transmission,
            "group": group,
            "default_pos": 0.0,
            "lower_limit": lower_limit,
            "upper_limit": upper_limit,
            "damping": 0.0,
            "armature": 0.0,
            "frictionloss": 0.0,
        }

        if is_passive:
            if joint_name in joint_gear_ratio_dict:
                joint_dict["gear_ratio"] = joint_gear_ratio_dict[joint_name]
            else:
                joint_dict["gear_ratio"] = 1.0

            if transmission == "gears":
                # TODO: Make sure the gear ratio doesn't amplify the dynamics paramters
                joint_drive_name = joint_name.replace("_driven", "_drive")
                motor_name = joint_motor_dict[joint_drive_name]
                for param_name in ["damping", "armature", "frictionloss"]:
                    joint_dict[param_name] = joint_dyn_params_dict[motor_name][
                        param_name
                    ]
            else:
                if joint_name in joint_dyn_params_dict:
                    for param_name in ["damping", "armature", "frictionloss"]:
                        joint_dict[param_name] = joint_dyn_params_dict[joint_name][
                            param_name
                        ]
        else:
            if joint_name not in joint_motor_dict:
                raise ValueError(f"{joint_name} not found in the spec dict!")

            motor_name = joint_motor_dict[joint_name]
            joint_dict["id"] = list(joint_motor_dict.keys()).index(joint_name)
            joint_dict["type"] = "dynamixel"
            joint_dict["spec"] = motor_name
            joint_dict["control_mode"] = (
                "current_based_position"
                if motor_name == "XC330" or motor_name == "XM430"
                else "extended_position"
            )
            joint_dict["init_pos"] = 0.0
            joint_dict["kp_real"] = kp
            joint_dict["ki_real"] = 0.0
            joint_dict["kd_real"] = kd
            joint_dict["kff2_real"] = 0.0
            joint_dict["kff1_real"] = 0.0
            joint_dict["kp_sim"] = kp / 128
            joint_dict["kd_sim"] = 0.0

            for param_name in ["damping", "armature", "frictionloss"]:
                joint_dict[param_name] = joint_dyn_params_dict[motor_name][param_name]

            id += 1

        config_dict["joints"][joint_name] = joint_dict

    config_dict["general"]["has_waist"] = has_waist
    config_dict["general"]["has_ankle"] = has_ankle

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
