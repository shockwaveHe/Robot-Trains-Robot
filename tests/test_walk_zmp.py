import argparse

import matplotlib.pyplot as plt
import numpy as np

from toddlerbot.algorithms.zmp_walk import ZMPWalk
from toddlerbot.sim.robot import Robot


def main(robot, command_x, command_y, command_z, cycle_time, foot_step_height):
    # Create a mock Robot instance with the necessary attributes for testing

    # Initialize the ZMPWalk instance with user-defined or default parameters
    zmp_walk = ZMPWalk(
        robot=robot,
        cycle_time=cycle_time,
        foot_step_height=foot_step_height,
        control_cost_Q=1.0,
        control_cost_R=1e-1,
    )

    # Create a command array for testing the build_lookup_table method
    command = np.array([command_x, command_y, command_z], dtype=np.float32)

    path_pos = np.zeros(3, dtype=np.float32)
    path_quat = np.array([1, 0, 0, 0], dtype=np.float32)

    desired_zmp, com_ref, leg_joint_pos_ref, stance_mask_ref = zmp_walk.plan(
        path_pos, path_quat, command
    )
    # Plotting the X vs Y trajectory
    plt.figure(figsize=(8, 8))
    plt.plot(com_ref[:, 0], com_ref[:, 1], linestyle="-")
    plt.plot(desired_zmp[:, 0], desired_zmp[:, 1], linestyle="--")

    # Adding labels and title
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Center of Mass (CoM) Trajectory")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ZMPWalk motion planning.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="Robot name to use for the test",
    )
    parser.add_argument(
        "--command_x", type=float, default=1.0, help="X-axis command input"
    )
    parser.add_argument(
        "--command_y", type=float, default=0.0, help="Y-axis command input"
    )
    parser.add_argument(
        "--command_z", type=float, default=0.0, help="Z-axis (rotation) command input"
    )
    parser.add_argument(
        "--cycle_time",
        type=float,
        default=1.0,
        help="Cycle time for each walking cycle",
    )
    parser.add_argument(
        "--foot_step_height",
        type=float,
        default=0.05,
        help="Foot step height for the walk",
    )

    args = parser.parse_args()

    robot = Robot(args.robot)

    main(
        robot,
        args.command_x,
        args.command_y,
        args.command_z,
        args.cycle_time,
        args.foot_step_height,
    )
