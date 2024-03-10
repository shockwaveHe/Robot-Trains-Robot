import argparse
import random

from toddlerbot.real.motor_control import MotorController
from toddlerbot.sim.mujoco_sim import MujoCoSim
from toddlerbot.sim.pybullet_sim import PyBulletSim
from toddlerbot.sim.robot import HumanoidRobot
from toddlerbot.utils.math_utils import round_floats


def main():
    parser = argparse.ArgumentParser(description="Run the walking simulation.")
    parser.add_argument(
        "--robot-name",
        type=str,
        default="base",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    parser.add_argument(
        "--sim",
        type=str,
        default="mujoco",
        help="The simulator to use.",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        default=False,
        help="Do sim2real control.",
    )
    parser.add_argument(
        "--sleep-time",
        type=float,
        default=0.0,
        help="Time to sleep between steps.",
    )
    args = parser.parse_args()

    random.seed(0)
    # A 0.3725 offset moves the robot slightly up from the ground
    robot = HumanoidRobot(args.robot_name)

    if args.sim == "pybullet":
        sim = PyBulletSim(robot)
    elif args.sim == "mujoco":
        sim = MujoCoSim(robot)
    else:
        raise ValueError("Unknown simulator")

    joint_angles = sim.initialize_joint_angles(robot)

    if args.real:
        motor_controller = MotorController()
        motor_controller.set_joint_angles(list(joint_angles.values()))

    else:

        def step_func():
            print(f"joint_angles: {round_floats(list(joint_angles.values()), 6)}")
            sim.set_joint_angles(robot, joint_angles)

        sim.simulate(step_func, sleep_time=args.sleep_time)


if __name__ == "__main__":
    main()
