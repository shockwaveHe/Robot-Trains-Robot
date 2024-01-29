import argparse
import random
import time

import pybullet as p

from toddleroid.sim.pybullet_sim import PyBulletSim
from toddleroid.sim.robot import HumanoidRobot
from toddleroid.utils.data_utils import round_floats


def main():
    parser = argparse.ArgumentParser(description="Run the walking simulation.")
    parser.add_argument(
        "--robot-name",
        type=str,
        default="sustaina_op",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    parser.add_argument(
        "--sleep-time",
        type=float,
        default=0.0,
        help="Time to sleep between steps.",
    )
    args = parser.parse_args()

    random.seed(0)
    sim = PyBulletSim()
    # A 0.3725 offset moves the robot slightly up from the ground
    robot = HumanoidRobot(args.robot_name)
    sim.load_robot(robot)
    sim.put_robot_on_ground(robot)

    joint_angles = []
    joint_names = []
    for idx in range(p.getNumJoints(robot.id)):
        if p.getJointInfo(robot.id, idx)[3] > -1:
            joint_angles += [0]
            joint_names += [p.getJointInfo(robot.id, idx)[1].decode("UTF-8")]

    while p.isConnected():
        if robot.name == "sustaina_op":
            print(f"joint_angles: {round_floats(joint_angles[7:], 6)}")
        elif robot.name == "robotis_op3":
            print(f"joint_angles: {round_floats(joint_angles, 6)}")
        else:
            raise ValueError("Unknown robot name")

        for idx in range(p.getNumJoints(robot.id)):
            qIndex = p.getJointInfo(robot.id, idx)[3]
            if qIndex > -1:
                p.setJointMotorControl2(
                    robot.id, idx, p.POSITION_CONTROL, joint_angles[qIndex - 7]
                )
        p.stepSimulation()
        if args.sleep_time > 0:
            time.sleep(args.sleep_time)


if __name__ == "__main__":
    main()
