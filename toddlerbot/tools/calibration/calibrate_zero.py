import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np

from toddlerbot.actuation.dynamixel.dynamixel_control import (
    DynamixelConfig,
    DynamixelController,
)
from toddlerbot.actuation.sunny_sky.sunny_sky_control import (
    SunnySkyConfig,
    SunnySkyController,
)
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_ports
from toddlerbot.utils.misc_utils import log


def calibrate_dynamixel(port: str, robot: Robot, group: str):
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


def calibrate_sunny_sky(port: str):
    sunny_sky_ids = robot.get_joint_attrs("type", "sunny_sky", "id")
    sunny_sky_config = SunnySkyConfig(
        port=port,
        kP=robot.get_joint_attrs("type", "sunny_sky", "kp_real"),
        kD=robot.get_joint_attrs("type", "sunny_sky", "kd_real"),
        i_ff=robot.get_joint_attrs("type", "sunny_sky", "i_ff_real"),
        gear_ratio=robot.get_joint_attrs("type", "sunny_sky", "gear_ratio"),
        joint_limit=robot.get_joint_attrs("type", "sunny_sky", "joint_limit"),
        init_pos=[],
    )
    controller = SunnySkyController(sunny_sky_config, sunny_sky_ids)

    init_pos: Dict[int, float] = controller.calibrate_motors()

    robot.set_joint_attrs("type", "sunny_sky", "init_pos", init_pos)

    controller.close_motors()


def main(robot: Robot):
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
    has_sunny_sky = robot.config["general"]["has_sunny_sky"]

    future_dynamixel = None
    if has_dynamixel:
        dynamixel_ports: List[str] = find_ports("USB <-> Serial Converter")
        future_dynamixel = executor.submit(
            calibrate_dynamixel, dynamixel_ports[0], robot, "all"
        )

    future_sunny_sky = None
    if has_sunny_sky:
        sunny_sky_ports: List[str] = find_ports("Feather")
        future_sunny_sky = executor.submit(calibrate_sunny_sky, sunny_sky_ports[0])

    if future_sunny_sky is not None:
        future_sunny_sky.result()
    if future_dynamixel is not None:
        future_dynamixel.result()

    executor.shutdown(wait=True)

    motor_names = robot.get_joint_attrs("is_passive", False)
    motor_pos_init = robot.get_joint_attrs("is_passive", False, "init_pos")
    motor_angles = {
        name: round(pos, 4) for name, pos in zip(motor_names, motor_pos_init)
    }
    log(f"Motor angles: {motor_angles}", header="Calibration")

    motor_config_path = os.path.join(robot.root_path, "config_motors.json")
    if os.path.exists(motor_config_path):
        with open(motor_config_path, "r") as f:
            motor_config = json.load(f)

        for motor_name, init_pos in zip(motor_names, motor_pos_init):
            motor_config[motor_name]["init_pos"] = float(init_pos)

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
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    args = parser.parse_args()

    robot = Robot(args.robot)

    main(robot)
