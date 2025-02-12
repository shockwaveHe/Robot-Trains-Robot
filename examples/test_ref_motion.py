import argparse
import time
from typing import List

import numpy as np
from tqdm import tqdm

from toddlerbot.locomotion.mjx_config import get_env_config
from toddlerbot.reference.balance_pd_ref import BalancePDReference
from toddlerbot.reference.motion_ref import MotionReference
from toddlerbot.reference.walk_simple_ref import WalkSimpleReference
from toddlerbot.reference.walk_zmp_ref import WalkZMPReference
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.math_utils import euler2quat

# This script is used to test the motion reference in the simulation.
# It requires a joystick connected to the computer. Without a joystick,
# you can still run the script by commenting out joystick-related lines
# and setting control_inputs to a fixed dictionary.


def test_ref_motion(
    robot: Robot,
    sim: MuJoCoSim,
    motion_ref: MotionReference,
    command_range: List[List[float]],
    vis_type: str,
):
    joystick = Joystick()

    state_ref = np.concatenate(
        [
            np.zeros(3, dtype=np.float32),  # Position
            euler2quat(
                np.array([0, 0, np.random.uniform(0, np.pi * 2)])
            ),  # Orientation
            np.zeros(3, dtype=np.float32),  # Linear velocity
            np.zeros(3, dtype=np.float32),  # Angular velocity
            motion_ref.default_motor_pos,
            motion_ref.default_joint_pos,
            np.ones(2, dtype=np.float32),  # Stance mask
        ]
    )
    pose_command = np.random.uniform(-1, 1, 5)

    # path_frame_mid = sim.model.body("path_frame").mocapid[0]

    p_bar = tqdm(desc="Running the test")
    step_idx = 0
    while True:
        try:
            control_inputs = joystick.get_controller_input()

            command = np.zeros(len(command_range), dtype=np.float32)
            if "walk" in motion_ref.name:
                command[:3] = pose_command[:3]
                for task, input in control_inputs.items():
                    axis = None
                    if task == "walk_x":
                        axis = 5
                    elif task == "walk_y":
                        axis = 6
                    elif task == "walk_turn":
                        axis = 7

                    if axis is not None:
                        command[axis] = np.interp(
                            input,
                            [-1, 0, 1],
                            [command_range[axis][1], 0.0, command_range[axis][0]],
                        )

            elif "balance" in motion_ref.name:
                for task, input in control_inputs.items():
                    if task == "look_left" and input > 0:
                        command[0] = input * command_range[0][1]
                    elif task == "look_right" and input > 0:
                        command[0] = input * command_range[0][0]
                    elif task == "look_up" and input > 0:
                        command[1] = input * command_range[1][1]
                    elif task == "look_down" and input > 0:
                        command[1] = input * command_range[1][0]
                    elif task == "lean_left" and input > 0:
                        command[3] = input * command_range[3][0]
                    elif task == "lean_right" and input > 0:
                        command[3] = input * command_range[3][1]
                    elif task == "twist_left" and input > 0:
                        command[4] = input * command_range[4][0]
                    elif task == "twist_right" and input > 0:
                        command[4] = input * command_range[4][1]
                    elif task == "squat":
                        command[5] = np.interp(
                            input,
                            [-1, 0, 1],
                            [command_range[5][1], 0.0, command_range[5][0]],
                        )

            elif "squat" in motion_ref.name:
                command[:3] = pose_command[:3]
                for task, input in control_inputs.items():
                    if task == "squat":
                        command[5] = np.interp(
                            control_inputs["squat"],
                            [-1, 0, 1],
                            [command_range[5][1], 0.0, command_range[5][0]],
                        )

            time_curr = step_idx * sim.control_dt
            state_ref = motion_ref.get_state_ref(state_ref, time_curr, command)

            # sim.data.mocap_pos[path_frame_mid] = state_ref[:3]
            # sim.data.mocap_quat[path_frame_mid] = state_ref[3:7]

            if step_idx == 0 or "walk" in motion_ref.name:
                qpos = motion_ref.get_qpos_ref(state_ref, path_frame=False)
                qpos[:3] = motion_ref.torso_pos_init + state_ref[:3]
                # motor_angles = dict(
                #     zip(robot.motor_ordering, state_ref[13 : 13 + robot.nu])
                # )
                # sim.set_motor_angles(motor_angles)
                # sim.step()
                sim.set_qpos(qpos)
                sim.forward()
            else:
                motor_angles = dict(
                    zip(robot.motor_ordering, state_ref[13 : 13 + robot.nu])
                )
                sim.set_motor_target(motor_angles)
                sim.step()

            step_idx += 1

            p_bar_steps = int(1 / sim.control_dt)
            if step_idx % p_bar_steps == 0:
                p_bar.update(p_bar_steps)

            if vis_type == "view":
                time.sleep(sim.control_dt)

        except KeyboardInterrupt:
            print("KeyboardInterrupt: Stopping the simulation...")
            break

    sim.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the walking simulation.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in descriptions.",
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
        default="view",
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
        sim = MuJoCoSim(robot, vis_type=args.vis, fixed_base="fixed" in args.robot)
    else:
        raise ValueError("Unknown simulator")

    motion_ref: MotionReference | None = None

    if "walk" in args.ref:
        walk_cfg = get_env_config("walk")
        command_range = walk_cfg.commands.command_range

        if args.ref == "walk_simple":
            motion_ref = WalkSimpleReference(
                robot,
                walk_cfg.sim.timestep * walk_cfg.action.n_frames,
                walk_cfg.action.cycle_time,
            )
        else:
            motion_ref = WalkZMPReference(
                robot,
                walk_cfg.sim.timestep * walk_cfg.action.n_frames,
                walk_cfg.action.cycle_time,
                walk_cfg.action.waist_roll_max,
            )

    elif "balance" in args.ref:
        balance_cfg = get_env_config("balance")
        command_range = balance_cfg.commands.command_range
        motion_ref = BalancePDReference(
            robot, balance_cfg.sim.timestep * balance_cfg.action.n_frames
        )

    else:
        raise ValueError("Unknown ref motion")

    test_ref_motion(robot, sim, motion_ref, command_range, args.vis)
