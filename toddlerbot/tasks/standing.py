import argparse
import os
import time

from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.pybullet_sim import PyBulletSim
from toddlerbot.sim.real_world import RealWorld
from toddlerbot.sim.robot import HumanoidRobot
from toddlerbot.utils.vis_plot import *


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
        default="mujoco",
        help="The simulator to use.",
    )
    parser.add_argument(
        "--sleep-time",
        type=float,
        default=0.0,
        help="Time to sleep between steps.",
    )
    args = parser.parse_args()

    exp_name = f"stand_{args.robot_name}_{args.sim}"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    exp_folder_path = f"results/{time_str}_{exp_name}"

    # A 0.3725 offset moves the robot slightly up from the ground
    robot = HumanoidRobot(args.robot_name)

    if args.sim == "pybullet":
        sim = PyBulletSim(robot)
    elif args.sim == "mujoco":
        sim = MuJoCoSim(robot)
    elif args.sim == "real":
        sim = RealWorld(robot)
    else:
        raise ValueError("Unknown simulator")

    joint_angles = robot.initialize_joint_angles()

    time_start = time.time()
    time_seq_ref = []
    time_seq_dict = {}
    joint_angle_ref_dict = {}
    joint_angle_dict = {}

    def step_func():
        time_ref = time.time() - time_start
        time_seq_ref.append(time_ref)
        for name, angle in joint_angles.items():
            if name not in joint_angle_ref_dict:
                joint_angle_ref_dict[name] = []
            joint_angle_ref_dict[name].append(angle)

        joint_state_dict = sim.get_joint_state(robot)
        for name, joint_state in joint_state_dict.items():
            if name not in time_seq_dict:
                time_seq_dict[name] = []
                joint_angle_dict[name] = []

            time_seq_dict[name].append(joint_state.time - time_start)
            joint_angle_dict[name].append(joint_state.pos)

        sim.set_joint_angles(robot, joint_angles)

    try:
        sim.simulate(step_func, sleep_time=args.sleep_time)
    finally:
        os.makedirs(exp_folder_path, exist_ok=True)

        plot_joint_tracking(
            time_seq_dict,
            time_seq_ref,
            joint_angle_dict,
            joint_angle_ref_dict,
            robot.config.motor_params,
            save_path=exp_folder_path,
        )


if __name__ == "__main__":
    main()
