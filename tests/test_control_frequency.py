import argparse
import time
from typing import List

from toddlerbot.sim.real_world import RealWorld
from toddlerbot.sim.robot import HumanoidRobot
from toddlerbot.utils.misc_utils import dump_profiling_data, log, precise_sleep, profile


# @profile()
def main(robot: HumanoidRobot):
    sim = RealWorld(robot, debug=True)

    step_idx = 0
    step_time_list: List[float] = []
    try:
        while True:
            step_start = time.time()

            _ = sim.get_joint_state()
            sim.set_joint_angles({name: 0 for name in robot.config})
            step_idx += 1

            step_time = time.time() - step_start
            step_time_list.append(step_time)
            log(f"Latency: {step_time * 1000:.2f} ms", header="Test", level="debug")

    except KeyboardInterrupt:
        pass

    finally:
        time.sleep(1)

        sim.close()

        # dump_profiling_data("profile_output.lprof")

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

    robot = HumanoidRobot(args.robot_name)

    main(robot)
