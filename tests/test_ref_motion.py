import argparse
import os
import time
from typing import List

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from toddlerbot.ref_motion import MotionReference
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.misc_utils import dump_profiling_data


def test_motion_ref(
    robot: Robot,
    sim: MuJoCoSim,
    motion_ref: MotionReference,
    command_list: List[npt.NDArray[np.float32]],
):
    exp_name: str = f"{robot.name}_{motion_ref.name}_{sim.name}_test"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    exp_folder_path = f"results/{exp_name}_{time_str}"
    os.makedirs(exp_folder_path, exist_ok=True)

    path_pos = np.zeros(3, dtype=np.float32)
    path_quat = np.array([1, 0, 0, 0], dtype=np.float32)
    try:
        for command in command_list:
            for _ in tqdm(
                range(5), desc="Running Ref Motion"
            ):  # Run the same command for 10 cycles
                for phase in np.arange(0, 1, sim.control_dt):  # type: ignore
                    _, state = motion_ref.get_state_ref(
                        path_pos, path_quat, phase, command
                    )
                    joint_angles = np.asarray(
                        state[13 : 13 + len(robot.joint_ordering)]
                    )  # type: ignore
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
        default="walk_simple",
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

    if args.ref == "walk_simple":
        from toddlerbot.envs.walk_env import WalkCfg
        from toddlerbot.ref_motion.walk_simple_ref import WalkSimpleReference

        cfg = WalkCfg()
        motion_ref = WalkSimpleReference(
            robot,
            cfg.action.cycle_time,
            default_joint_pos=np.array(list(robot.default_joint_angles.values())),  # type: ignore
        )

    elif args.ref == "walk_zmp":
        from toddlerbot.envs.walk_env import WalkCfg
        from toddlerbot.ref_motion.walk_zmp_ref import WalkZMPReference

        cfg = WalkCfg()
        motion_ref = WalkZMPReference(
            robot,
            cfg.action.cycle_time,
            [
                cfg.commands.lin_vel_x_range,
                cfg.commands.lin_vel_y_range,
                cfg.commands.ang_vel_yaw_range,
            ],
            default_joint_pos=np.array(list(robot.default_joint_angles.values())),  # type: ignore
        )

    elif args.ref == "squat":
        from toddlerbot.envs.squat_env import SquatCfg
        from toddlerbot.ref_motion.squat_ref import SquatReference

        cfg = SquatCfg()
        motion_ref = SquatReference(
            robot,
            cfg.action.episode_time,
            default_joint_pos=np.array(list(robot.default_joint_angles.values())),  # type: ignore
        )

    else:
        raise ValueError("Unknown ref motion")

    if "walk" in args.ref:
        command_list = [
            np.array([0.3, 0, 0], dtype=np.float32),
            # np.array([0, -0.1, 0], dtype=np.float32),
            np.array([0.0, 0, 0.2], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.float32),
        ]

    elif args.ref == "squat":
        command_list = [
            np.array([1, 0, 0], dtype=np.float32),
            np.array([-1, 0, 0], dtype=np.float32),
        ]

    else:
        command_list = [np.zeros(3, dtype=np.float32)]

    test_motion_ref(robot, sim, motion_ref, command_list)
