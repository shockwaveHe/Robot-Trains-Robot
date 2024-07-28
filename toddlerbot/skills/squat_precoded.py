import argparse
import time
from typing import Dict, List, Tuple

import numpy as np

from toddlerbot.sim.real_world import RealWorld
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import resample_trajectory
from toddlerbot.utils.misc_utils import dump_profiling_data, log, precise_sleep, profile


# @profile()
def main(robot: Robot):
    sim = RealWorld(robot, debug=True)

    control_dt = 0.01
    curr_joint_angles = {
        name: state.pos for name, state in sim.get_joint_state(retries=-1).items()
    }
    zero_joint_angles: Dict[str, float] = {name: 0 for name in robot.config}
    default_joint_angles = robot.initialize_motor_angles()
    middle_joint_angles: Dict[str, float] = {}
    for name in robot.config:
        middle_joint_angles[name] = (
            zero_joint_angles[name]
            + (default_joint_angles[name] - zero_joint_angles[name]) / 2
        )

    middle_joint_angles["left_sho_roll"] = np.pi / 6
    middle_joint_angles["right_sho_roll"] = np.pi / 6

    joint_angles_traj: List[Tuple[float, Dict[str, float]]] = []
    joint_angles_traj.append((0.0, curr_joint_angles))
    joint_angles_traj.append((1.0, zero_joint_angles))
    joint_angles_traj.append((4.0, zero_joint_angles))
    joint_angles_traj.append((4.5, middle_joint_angles))
    joint_angles_traj.append((5.0, default_joint_angles))
    joint_angles_traj.append((8.0, default_joint_angles))
    # joint_angles_traj.append((8.5, middle_joint_angles))
    # joint_angles_traj.append((9.0, zero_joint_angles))
    joint_angles_traj = resample_trajectory(
        joint_angles_traj,
        desired_interval=control_dt,
        interp_type="cubic",
    )

    step_idx = 0
    step_time_list: List[float] = []
    try:
        while True:
            step_start = time.time()

            _, joint_angles = joint_angles_traj[
                min(step_idx, len(joint_angles_traj) - 1)
            ]

            _ = sim.get_joint_state()
            sim.set_motor_angles(joint_angles)
            step_idx += 1

            step_time = time.time() - step_start
            step_time_list.append(step_time)
            log(f"Latency: {step_time * 1000:.2f} ms", header="Test", level="debug")
            time_until_next_step = control_dt - step_time
            if time_until_next_step > 0:
                precise_sleep(time_until_next_step)

    except KeyboardInterrupt:
        pass

    finally:
        time.sleep(1)

        sim.close()

        log(
            f"Average Latency: {sum(step_time_list) / len(step_time_list) * 1000:.2f} ms",
            header="Test",
            level="info",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the contorl frequency test.")
    parser.add_argument(
        "--robot-name",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    args = parser.parse_args()

    robot = Robot(args.robot_name)

    main(robot)
