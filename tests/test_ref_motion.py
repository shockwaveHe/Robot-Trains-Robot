import argparse
import os
import time

import numpy as np
from tqdm import tqdm

from toddlerbot.ref_motion import MotionReference
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.misc_utils import dump_profiling_data


def test_walk_ref(robot: Robot, sim: MuJoCoSim, walk_ref: MotionReference):
    exp_name: str = f"{robot.name}_{walk_ref.name}_{sim.name}_test"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    exp_folder_path = f"results/{exp_name}_{time_str}"
    os.makedirs(exp_folder_path, exist_ok=True)

    path_pos = np.zeros(3, dtype=np.float32)
    path_quat = np.array([1, 0, 0, 0], dtype=np.float32)

    command_list = [
        np.array([0, 0, 0], dtype=np.float32),
        np.array([0.3, 0, 0], dtype=np.float32),
        np.array([0, -0.1, 0], dtype=np.float32),
        np.array([0.0, 0, 0.2], dtype=np.float32),
    ]
    duration = 10
    try:
        for command in command_list:
            for phase in tqdm(
                np.arange(0, duration, sim.control_dt),  # type: ignore
                desc="Running Ref Motion",
            ):
                state = walk_ref.get_state_ref(
                    path_pos, path_quat, phase, command, duration
                )
                joint_angles = np.asarray(state[13 : 13 + len(robot.joint_ordering)])  # type: ignore
                motor_angles = robot.joint_to_motor_angles(
                    dict(zip(robot.joint_ordering, joint_angles))
                )
                sim.set_motor_angles(motor_angles)
                sim.step()

    except KeyboardInterrupt:
        print("KeyboardInterrupt: Stopping the simulation...")

    finally:
        if hasattr(sim, "save_recording"):
            assert isinstance(sim, MuJoCoSim)
            sim.save_recording(exp_folder_path, sim.control_dt, 2)

        sim.close()

        prof_path = os.path.join(exp_folder_path, "profile_output.lprof")
        dump_profiling_data(prof_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the walking simulation.")
    parser.add_argument(
        "--robot",
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
        "--vis",
        type=str,
        default="render",
        help="The visualization type.",
    )
    parser.add_argument(
        "--ref",
        type=str,
        default="zmp",
        help="The name of the task.",
    )
    args = parser.parse_args()

    robot = Robot("toddlerbot")
    if args.sim == "mujoco":
        from toddlerbot.sim.mujoco_sim import MuJoCoSim

        sim = MuJoCoSim(robot, vis_type=args.vis, fixed_base=True)
        sim.load_keyframe()
    else:
        raise ValueError("Unknown simulator")

    if args.ref == "simple":
        from toddlerbot.ref_motion.walk_simple_ref import WalkSimpleReference

        walk_ref = WalkSimpleReference(
            robot,
            default_joint_pos=np.array(list(robot.default_joint_angles.values())),  # type: ignore
        )

    elif args.ref == "limp":
        from toddlerbot.ref_motion.walk_lipm_ref import WalkLIPMReference

        walk_ref = WalkLIPMReference(
            robot,
            default_joint_pos=np.array(list(robot.default_joint_angles.values())),  # type: ignore
        )

    else:
        from toddlerbot.ref_motion.walk_zmp_ref import WalkZMPReference

        walk_ref = WalkZMPReference(
            robot,
            default_joint_pos=np.array(list(robot.default_joint_angles.values())),  # type: ignore
        )

    test_walk_ref(robot, sim, walk_ref)
