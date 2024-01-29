import argparse
import random

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
    # A 0.3725 offset moves the robot slightly up from the ground
    robot = HumanoidRobot(args.robot_name)
    sim = PyBulletSim(robot)

    joint_angles, joint_names = sim.initialize_named_joint_angles(robot)

    def step_func():
        if robot.name == "sustaina_op":
            print(f"joint_angles: {round_floats(joint_angles[7:], 6)}")
        elif robot.name == "robotis_op3":
            print(f"joint_angles: {round_floats(joint_angles, 6)}")
        else:
            raise ValueError("Unknown robot name")

        sim.set_joint_angles(robot, joint_angles)

    sim.simulate(step_func, sleep_time=args.sleep_time)


if __name__ == "__main__":
    main()
