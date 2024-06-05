import argparse
import time
from typing import List

import numpy as np

from toddlerbot.sim.real_world import RealWorld
from toddlerbot.sim.robot import HumanoidRobot
from toddlerbot.utils.misc_utils import dump_profiling_data, log, precise_sleep, profile


# @profile()
def main(robot: HumanoidRobot):
    ankle_motor_pos_left = robot.ankle_ik([0.0, 0.0], "left")
    ankle_motor_pos_right = robot.ankle_ik([0.0, 0.0], "right")
    log(
        f"Ankle pos: {[0.0, 0.0]}, motor pos: {ankle_motor_pos_left}",
        header="Test",
        level="info",
    )
    log(
        f"Ankle pos: {[0.0, 0.0]}, motor pos: {ankle_motor_pos_right}",
        header="Test",
        level="info",
    )

    ankle_motor_pos_left = robot.ankle_ik([np.pi / 4, 0.0], "left")
    ankle_motor_pos_right = robot.ankle_ik([np.pi / 4, 0.0], "right")
    log(
        f"Ankle pos: {[np.pi / 4, 0.0]}, motor pos: {ankle_motor_pos_left}",
        header="Test",
        level="info",
    )
    log(
        f"Ankle pos: {[np.pi / 4, 0.0]}, motor pos: {ankle_motor_pos_right}",
        header="Test",
        level="info",
    )

    # sim = RealWorld(robot, debug=True)

    # control_dt = 0.01
    # default_q = np.array(list(cfg.init_state.default_joint_angles.values()))

    # if sim.name == "isaac":
    #     # TODO: Isaac has a bug when warming up
    #     sim.reset_dof_state(default_q)
    #     sim.run_simulation(headless=True)

    # else:
    #     if hasattr(sim, "run_simulation"):
    #         sim.run_simulation(headless=True)

    #     zero_joint_angles, initial_joint_angles = robot.initialize_joint_angles()
    #     joint_angles_traj = []
    #     joint_angles_traj.append((0.0, zero_joint_angles))
    #     joint_angles_traj.append((0.5, initial_joint_angles))
    #     joint_angles_traj.append((1.5, cfg.init_state.default_joint_angles))
    #     joint_angles_traj.append((2.0, cfg.init_state.default_joint_angles))
    #     joint_angles_traj = resample_trajectory(
    #         joint_angles_traj,
    #         desired_interval=control_dt,
    #         interp_type="cubic",
    #     )
    #     step_idx = 0
    #     time_start = time.time()
    #     while time.time() - time_start < joint_angles_traj[-1][0]:
    #         step_start = time.time()

    #         _, joint_angles = joint_angles_traj[
    #             min(step_idx, len(joint_angles_traj) - 1)
    #         ]
    #         sim.set_joint_angles(joint_angles)

    #         step_idx += 1

    #         time_until_next_step = control_dt - (time.time() - step_start)
    #         if time_until_next_step > 0:
    #             precise_sleep(time_until_next_step)

    # step_idx = 0
    # sim_dt = 0.001
    # step_time_list: List[float] = []
    # try:
    #     while True:
    #         step_start = time.time()

    #         _ = sim.get_joint_state()
    #         sim.set_joint_angles({name: 0 for name in robot.config})
    #         step_idx += 1

    #         step_time = time.time() - step_start
    #         step_time_list.append(step_time)
    #         log(f"Latency: {step_time * 1000:.2f} ms", header="Test", level="debug")
    #         time_until_next_step = sim_dt - step_time
    #         if time_until_next_step > 0:
    #             precise_sleep(time_until_next_step)

    # except KeyboardInterrupt:
    #     pass

    # finally:
    #     time.sleep(1)

    #     sim.close()

    #     dump_profiling_data("profile_output.lprof")

    #     log(
    #         f"Average Latency: {sum(step_time_list) / len(step_time_list) * 1000:.2f} ms",
    #         header="Test",
    #         level="info",
    #     )


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
