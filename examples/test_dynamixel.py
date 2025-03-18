import argparse
import time
from typing import List

from toddlerbot.actuation.dynamixel_control import (
    DynamixelConfig,
    DynamixelController,
)
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_ports

# This script is used to test the control frequency of the Dynamixel motors.


def get_dynamixel_controller(
    robot: Robot, group: str = "", id_list: List[int] = []
) -> DynamixelController:
    dynamixel_ports: List[str] = find_ports("USB <-> Serial Converter")

    dynamixel_ids: List[int] = []
    if len(id_list) > 0:
        dynamixel_ids = id_list
    elif len(group) > 0:
        dynamixel_ids = robot.get_joint_attrs("type", "dynamixel", "id", group=group)
    else:
        dynamixel_ids = robot.get_joint_attrs("type", "dynamixel", "id")

    control_mode: List[str] = []
    kP: List[float] = []
    kI: List[float] = []
    kD: List[float] = []
    kFF2: List[float] = []
    kFF1: List[float] = []
    init_pos: List[float] = []
    for dynamixel_id in dynamixel_ids:
        for joint_config in robot.config["joints"].values():
            if (
                joint_config["type"] == "dynamixel"
                and joint_config["id"] == dynamixel_id
            ):
                control_mode.append(joint_config["control_mode"])
                kP.append(joint_config["kp_real"])
                kI.append(joint_config["ki_real"])
                kD.append(joint_config["kd_real"])
                kFF2.append(joint_config["kff2_real"])
                kFF1.append(joint_config["kff1_real"])
                init_pos.append(joint_config["init_pos"])
                break

    dynamixel_config = DynamixelConfig(
        port=dynamixel_ports[0],
        baudrate=robot.config["general"]["dynamixel_baudrate"],
        control_mode=control_mode,
        kP=kP,
        kI=kI,
        kD=kD,
        kFF2=kFF2,
        kFF1=kFF1,
        init_pos=init_pos,
    )
    dynamixel_controller = DynamixelController(dynamixel_config, dynamixel_ids)

    return dynamixel_controller


# @profile()
def main(robot: Robot):
    # Specify the Dynamixel IDs you want to control
    id_list = [*range(0, 30)]

    dynamixel_controller = get_dynamixel_controller(robot, id_list=id_list, group="all")

    step_idx = 0
    step_time_list: List[float] = []
    try:
        while True:
            step_start = time.time()

            motor_state = dynamixel_controller.get_motor_state()
            print(motor_state)

            step_idx += 1

            step_time = time.time() - step_start
            step_time_list.append(step_time)
            print(f"Latency: {step_time * 1000:.2f} ms")

    except KeyboardInterrupt:
        pass

    finally:
        time.sleep(1)

        dynamixel_controller.close_motors()

        print(
            f"Average Latency: {sum(step_time_list) / len(step_time_list) * 1000:.2f} ms"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the dynamixel test.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    args = parser.parse_args()

    robot = Robot(args.robot)

    main(robot)
