import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from toddlerbot.actuation.dynamixel.dynamixel_control import (
    DynamixelConfig,
    DynamixelController,
)
from toddlerbot.actuation.sunny_sky.sunny_sky_control import (
    SunnySkyConfig,
    SunnySkyController,
)
from toddlerbot.sim.robot import HumanoidRobot
from toddlerbot.utils.file_utils import find_ports


def calibrate_dynamixel(port: str, robot: HumanoidRobot, group: str):
    dynamixel_ids = robot.get_attrs("type", "dynamixel", "id", group)
    dynamixel_config = DynamixelConfig(
        port=port,
        control_mode=robot.get_attrs("type", "dynamixel", "control_mode", group),
        kP=robot.get_attrs("type", "dynamixel", "kp_real", group),
        kI=robot.get_attrs("type", "dynamixel", "ki_real", group),
        kD=robot.get_attrs("type", "dynamixel", "kd_real", group),
        kFF2=robot.get_attrs("type", "dynamixel", "kff2_real", group),
        kFF1=robot.get_attrs("type", "dynamixel", "kff1_real", group),
        gear_ratio=robot.get_attrs("type", "dynamixel", "gear_ratio", group),
        init_pos=[],
    )

    controller = DynamixelController(dynamixel_config, dynamixel_ids)
    init_pos: Dict[int, float] = controller.calibrate_motors(
        robot.get_attrs("type", "dynamixel", "is_indirect", group)
    )

    robot.set_attrs("type", "dynamixel", "init_pos", init_pos, group)

    controller.close_motors()


def calibrate_sunny_sky(port: str):
    sunny_sky_ids = robot.get_attrs("type", "sunny_sky", "id")
    sunny_sky_config = SunnySkyConfig(
        port=port,
        kP=robot.get_attrs("type", "sunny_sky", "kp_real"),
        kD=robot.get_attrs("type", "sunny_sky", "kd_real"),
        i_ff=robot.get_attrs("type", "sunny_sky", "i_ff_real"),
        gear_ratio=robot.get_attrs("type", "sunny_sky", "gear_ratio"),
        joint_limit=robot.get_attrs("type", "sunny_sky", "joint_limit"),
        init_pos=[],
    )
    controller = SunnySkyController(sunny_sky_config, sunny_sky_ids)

    init_pos: Dict[int, float] = controller.calibrate_motors()

    robot.set_attrs("type", "sunny_sky", "init_pos", init_pos)

    controller.close_motors()


def main(robot: HumanoidRobot):
    while True:
        response = input("Have you installed the calibration parts? (Y/N): ")
        response = response.strip().lower()
        if response == "y" or response[0] == "y":
            break
        if response == "n" or response[0] == "n":
            return

        print("Please answer 'yes' or 'no'.")

    dynamixel_ports: List[str] = find_ports("USB <-> Serial Converter")
    sunny_sky_ports: List[str] = find_ports("Feather")

    n_ports = len(dynamixel_ports) + len(sunny_sky_ports)
    executor = ThreadPoolExecutor(max_workers=n_ports)

    future_dynamixel = executor.submit(
        calibrate_dynamixel, dynamixel_ports[0], robot, "all"
    )

    future_sunny_sky = executor.submit(calibrate_sunny_sky, sunny_sky_ports[0])

    future_sunny_sky.result()
    future_dynamixel.result()

    robot.write_robot_config()

    executor.shutdown(wait=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the zero point calibration.")
    parser.add_argument(
        "--robot-name",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    args = parser.parse_args()

    robot = HumanoidRobot(args.robot_name)

    main(robot)
