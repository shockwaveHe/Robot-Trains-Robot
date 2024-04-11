import argparse

import numpy as np
import pybullet as p

from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.pybullet_sim import PyBulletSim
from toddlerbot.sim.robot import HumanoidRobot
from toddlerbot.utils.math_utils import round_floats


def main():
    parser = argparse.ArgumentParser(description="Run the walking simulation.")
    parser.add_argument(
        "--robot-name",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    parser.add_argument(
        "--sim",
        type=str,
        default="pybullet",
        choices=["pybullet", "mujoco"],
        help="The simulator to use.",
    )
    parser.add_argument(
        "--sleep-time",
        type=float,
        default=0.0,
        help="Time to sleep between steps.",
    )
    args = parser.parse_args()

    robot = HumanoidRobot(args.robot_name)
    if args.sim == "pybullet":
        sim = PyBulletSim(robot, fixed=True)
        controls = {}
        for idx in range(p.getNumJoints(robot.id)):
            name = p.getJointInfo(robot.id, idx)[1].decode("UTF-8")
            qIndex = p.getJointInfo(robot.id, idx)[3]
            if not "passive" in name and qIndex > -1:
                low, high = p.getJointInfo(robot.id, idx)[8:10]
                print(f"{name} has limits ({low}, {high})")
                controls[name] = p.addUserDebugParameter(name, low, high, 0)
    elif args.sim == "mujoco":
        sim = MuJoCoSim(robot)
    else:
        raise ValueError("Unknown simulator")

    sim_step_idx = 0

    # This function requires its parameters to be the same as its return values.
    def step_func(sim_step_idx):
        if args.sim == "pybullet":
            joint_angles = robot.initialize_joint_angles()
            sim.set_joint_angles(robot, joint_angles)
        else:
            raise ValueError("Only pybullet is supported for now.")

        sim_step_idx += 1

        return (sim_step_idx,)

    sim.simulate(step_func, (sim_step_idx,), sleep_time=args.sleep_time)


if __name__ == "__main__":
    main()
