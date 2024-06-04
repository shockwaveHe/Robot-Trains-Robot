import argparse
import time

from toddlerbot.sim.real_world import RealWorld
from toddlerbot.sim.robot import HumanoidRobot
from toddlerbot.utils.misc_utils import dump_profiling_data, log, precise_sleep, profile


@profile()
def main(robot: HumanoidRobot):
    sim = RealWorld(robot, debug=True)

    step_idx = 0
    sim_dt = 0.001
    try:
        while step_idx < 100:
            step_start = time.time()

            _ = sim.get_joint_state()
            sim.set_joint_angles({name: 0 for name in robot.config})
            step_idx += 1

            step_time = time.time() - step_start
            log(
                f"Control Frequency: {1 / step_time:.2f} Hz",
                header="Test",
                level="debug",
            )
            time_until_next_step = sim_dt - step_time
            if time_until_next_step > 0:
                precise_sleep(time_until_next_step)

    except KeyboardInterrupt:
        pass

    finally:
        sim.close()

        dump_profiling_data("profile_output.lprof")


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
