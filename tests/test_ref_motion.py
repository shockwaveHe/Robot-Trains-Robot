import argparse
import os
import time
from typing import List

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from toddlerbot.envs.balance_env import BalanceCfg
from toddlerbot.envs.mjx_config import MJXConfig
from toddlerbot.envs.rotate_torso_env import RotateTorsoCfg
from toddlerbot.envs.squat_env import SquatCfg
from toddlerbot.envs.walk_env import WalkCfg
from toddlerbot.ref_motion import MotionReference
from toddlerbot.ref_motion.balance_ref import BalanceReference
from toddlerbot.ref_motion.rotate_torso_ref import RotateTorsoReference
from toddlerbot.ref_motion.squat_ref import SquatReference
from toddlerbot.ref_motion.walk_simple_ref import WalkSimpleReference
from toddlerbot.ref_motion.walk_zmp_ref import WalkZMPReference
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.misc_utils import dump_profiling_data


def test_motion_ref(
    robot: Robot,
    sim: MuJoCoSim,
    motion_ref: MotionReference,
    command_list: List[npt.NDArray[np.float32]],
    time_total: float = 5.0,
    vis_type: str = "render",
):
    exp_name: str = f"{robot.name}_{motion_ref.name}_{sim.name}_test"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    exp_folder_path = f"results/{exp_name}_{time_str}"
    os.makedirs(exp_folder_path, exist_ok=True)

    path_pos = np.zeros(3, dtype=np.float32)
    path_quat = np.array([1, 0, 0, 0], dtype=np.float32)
    try:
        for command in command_list:
            for time_curr in tqdm(
                np.arange(0, time_total, sim.control_dt),
                desc="Running Ref Motion",
            ):
                state = motion_ref.get_state_ref(
                    path_pos, path_quat, time_curr, command
                )
                joint_angles = np.asarray(state[13 : 13 + robot.nu])
                # motor_angles = robot.joint_to_motor_angles(
                #     dict(zip(robot.joint_ordering, joint_angles))
                # )
                # sim.set_motor_angles(motor_angles)
                # sim.step()
                sim.set_joint_angles(dict(zip(robot.joint_ordering, joint_angles)))
                sim.forward()

                if vis_type == "view":
                    time.sleep(sim.control_dt)

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

    robot = Robot(args.robot)
    if args.sim == "mujoco":
        from toddlerbot.sim.mujoco_sim import MuJoCoSim

        sim = MuJoCoSim(robot, vis_type=args.vis, fixed_base=True)
        sim.load_keyframe()
    else:
        raise ValueError("Unknown simulator")

    cfg: MJXConfig | None = None
    motion_ref: MotionReference | None = None

    if args.ref == "walk_simple":
        cfg = WalkCfg()
        motion_ref = WalkSimpleReference(robot, cfg.action.cycle_time)

    elif args.ref == "walk_zmp":
        cfg = WalkCfg()
        motion_ref = WalkZMPReference(
            robot,
            cfg.commands.command_list,
            cfg.action.cycle_time,
            cfg.sim.timestep * cfg.action.n_frames,
        )

    elif args.ref == "squat":
        cfg = SquatCfg()
        motion_ref = SquatReference(robot)

    elif args.ref == "rotate_torso":
        cfg = RotateTorsoCfg()
        motion_ref = RotateTorsoReference(robot)

    elif args.ref == "balance":
        cfg = BalanceCfg()
        motion_ref = BalanceReference(robot)

    else:
        raise ValueError("Unknown ref motion")

    if "walk" in args.ref:
        command_list = [
            np.array([0.2, 0, 0], dtype=np.float32),
            np.array([-0.1, 0, 0], dtype=np.float32),
            np.array([0, 0.1, 0], dtype=np.float32),
            np.array([0, 0.0, 0.5], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.float32),
        ]

    elif "squat" in args.ref:
        command_list = [
            np.array([0.05, 0, 0], dtype=np.float32),
            np.array([-0.05, 0, 0], dtype=np.float32),
        ]

    elif "rotate_torso" in args.ref:
        command_list = [
            np.array([0.2, 0], dtype=np.float32),
            np.array([0, 1.0], dtype=np.float32),
        ]

    else:
        command_list = [
            np.array([0.0], dtype=np.float32),
            np.array([0.5], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        ]

    test_motion_ref(robot, sim, motion_ref, command_list, vis_type=args.vis)
