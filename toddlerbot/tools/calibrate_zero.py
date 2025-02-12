import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np

from toddlerbot.actuation.dynamixel_control import (
    DynamixelConfig,
    DynamixelController,
)
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_ports
from toddlerbot.utils.misc_utils import log


# This script is used to calibrate the zero points of the Dynamixel motors.


def calibrate_dynamixel(port: str, robot: Robot, group: str):
    """Calibrates the Dynamixel motors for a specified group of joints in a robot.

    This function retrieves the necessary configuration and state information for the Dynamixel motors associated with a specified group of joints in the robot. It then calculates the initial positions for these motors and updates the robot's joint attributes accordingly.

    Args:
        port (str): The communication port used to connect to the Dynamixel motors.
        robot (Robot): The robot instance containing joint and configuration data.
        group (str): The group of joints to be calibrated.
    """
    dynamixel_ids = robot.get_joint_attrs("type", "dynamixel", "id", group)
    dynamixel_config = DynamixelConfig(
        port=port,
        baudrate=robot.config["general"]["dynamixel_baudrate"],
        control_mode=robot.get_joint_attrs("type", "dynamixel", "control_mode", group),
        kP=robot.get_joint_attrs("type", "dynamixel", "kp_real", group),
        kI=robot.get_joint_attrs("type", "dynamixel", "ki_real", group),
        kD=robot.get_joint_attrs("type", "dynamixel", "kd_real", group),
        kFF2=robot.get_joint_attrs("type", "dynamixel", "kff2_real", group),
        kFF1=robot.get_joint_attrs("type", "dynamixel", "kff1_real", group),
        # gear_ratio=robot.get_joint_attrs("type", "dynamixel", "gear_ratio", group),
        init_pos=[],
    )

    controller = DynamixelController(dynamixel_config, dynamixel_ids)

    transmission_list = robot.get_joint_attrs(
        "type", "dynamixel", "transmission", group
    )
    joint_group_list = robot.get_joint_attrs("type", "dynamixel", "group", group)
    state_dict = controller.get_motor_state(retries=-1)
    init_pos: Dict[int, float] = {}
    for transmission, joint_group, (id, state) in zip(
        transmission_list, joint_group_list, state_dict.items()
    ):
        if transmission == "none" and joint_group == "arm":
            init_pos[id] = np.pi / 4 * round(state.pos / (np.pi / 4))
        else:
            init_pos[id] = state.pos

    robot.set_joint_attrs("type", "dynamixel", "init_pos", init_pos, group)

    controller.close_motors()


def main(robot: Robot, parts: List[str]):
    """Calibrates the robot's motors based on specified parts and updates the configuration.

    This function prompts the user to confirm the installation of calibration parts,
    calibrates the Dynamixel motors if present, and updates the motor configuration
    file with the initial positions of the specified parts.

    Args:
        robot (Robot): The robot instance containing configuration and joint attributes.
        parts (List[str]): A list of parts to calibrate. Can include specific parts
            like 'left_arm', 'right_arm', or 'all' to calibrate all parts.

    Raises:
        ValueError: If an invalid part is specified in the `parts` list.
        FileNotFoundError: If the motor configuration file is not found.
    """
    while True:
        response = input("Have you installed the calibration parts? (y/n) > ")
        response = response.strip().lower()
        if response == "y" or response[0] == "y":
            break
        if response == "n" or response[0] == "n":
            return

        print("Please answer 'yes' or 'no'.")

    executor = ThreadPoolExecutor()

    has_dynamixel = robot.config["general"]["has_dynamixel"]

    future_dynamixel = None
    if has_dynamixel:
        dynamixel_ports: List[str] = find_ports("USB <-> Serial Converter")
        future_dynamixel = executor.submit(
            calibrate_dynamixel, dynamixel_ports[0], robot, "all"
        )

    if future_dynamixel is not None:
        future_dynamixel.result()

    executor.shutdown(wait=True)

    # Generate the motor mask based on the specified parts
    all_parts = {
        "left_arm": [16, 17, 18, 19, 20, 21, 22],
        "right_arm": [23, 24, 25, 26, 27, 28, 29],
        "left_gripper": [30],
        "right_gripper": [31],
        "hip": [2, 3, 4, 5, 6, 10, 11, 12],
        "knee": [7, 13],
        "left_ankle": [8, 9],
        "right_ankle": [14, 15],
        "neck": [0, 1],
    }
    if "all" in parts:
        motor_mask = list(range(robot.nu))
    else:
        motor_mask = []
        for part in parts:
            if part not in all_parts:
                raise ValueError(f"Invalid part: {part}")

            motor_mask.extend(all_parts[part])

    motor_names = robot.get_joint_attrs("is_passive", False)
    motor_pos_init = np.array(robot.get_joint_attrs("is_passive", False, "init_pos"))
    motor_angles = {}
    for i, (name, pos) in enumerate(zip(motor_names, motor_pos_init)):
        if i in motor_mask:
            motor_angles[name] = round(pos, 4)

    log(f"Motor angles for selected parts: {motor_angles}", header="Calibration")

    motor_config_path = os.path.join(robot.root_path, "config_motors.json")
    if os.path.exists(motor_config_path):
        with open(motor_config_path, "r") as f:
            motor_config = json.load(f)

        for i, (name, pos) in enumerate(zip(motor_names, motor_pos_init)):
            if i in motor_mask:
                motor_config[name]["init_pos"] = float(pos)

        with open(motor_config_path, "w") as f:
            json.dump(motor_config, f, indent=4)
    else:
        raise FileNotFoundError(f"Could not find {motor_config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the zero point calibration.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    parser.add_argument(
        "--parts",
        type=str,
        default="all",
        help="Specify parts to calibrate. Use 'all' or a subset of [left_arm, right_arm, left_gripper, right_gripper, hip, knee, left_ankle, right_ankle, neck], split by space.",
    )
    args = parser.parse_args()

    # Parse parts into a list
    parts = args.parts.split(" ") if args.parts != "all" else ["all"]

    robot = Robot(args.robot)

    main(robot, parts)
