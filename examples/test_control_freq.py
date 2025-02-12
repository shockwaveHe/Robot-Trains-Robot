import argparse
import time
from typing import List

from toddlerbot.sim.real_world import RealWorld
from toddlerbot.sim.robot import Robot

# This script is used to test the control frequency in the real world.


# @profile()
def main(robot: Robot):
    sim = RealWorld(robot)

    step_idx = 0
    step_time_list: List[float] = []
    try:
        while True:
            step_start = time.time()

            _ = sim.get_observation()
            sim.set_motor_target(robot.init_motor_angles)

            step_idx += 1

            step_time = time.time() - step_start
            step_time_list.append(step_time)

            print(f"Latency: {step_time * 1000:.2f} ms")

    except KeyboardInterrupt:
        pass

    finally:
        time.sleep(1)

        sim.close()

        print(
            f"Average Latency: {sum(step_time_list) / len(step_time_list) * 1000:.2f} ms"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the contorl frequency test.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    args = parser.parse_args()

    robot = Robot(args.robot)

    main(robot)
