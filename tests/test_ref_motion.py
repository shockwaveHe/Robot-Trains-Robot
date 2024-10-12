import argparse
import time
from typing import List

import numpy as np

from toddlerbot.envs.balance_env import BalanceCfg
from toddlerbot.envs.squat_env import SquatCfg
from toddlerbot.envs.turn_env import TurnCfg
from toddlerbot.envs.walk_env import WalkCfg
from toddlerbot.ref_motion import MotionReference
from toddlerbot.ref_motion.balance_pd_ref import BalancePDReference
from toddlerbot.ref_motion.squat_ref import SquatReference
from toddlerbot.ref_motion.walk_simple_ref import WalkSimpleReference
from toddlerbot.ref_motion.walk_zmp_ref import WalkZMPReference
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick


def test_motion_ref(
    robot: Robot,
    sim: MuJoCoSim,
    motion_ref: MotionReference,
    command_range: List[List[float]],
):
    joystick = Joystick()

    default_joint_pos = np.array(
        list(robot.default_joint_angles.values()), dtype=np.float32
    )
    state_ref = np.concatenate(
        [
            np.zeros(3, dtype=np.float32),  # Position
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # Quaternion
            np.zeros(3, dtype=np.float32),  # Linear velocity
            np.zeros(3, dtype=np.float32),  # Angular velocity
            default_joint_pos,  # Joint positions
            np.zeros_like(default_joint_pos),  # Joint velocities
            np.ones(2, dtype=np.float32),  # Stance mask
        ]
    )
    time_curr = 0.0
    while True:
        try:
            control_inputs = joystick.get_controller_input()

            command = np.zeros(len(command_range), dtype=np.float32)
            if "walk" in motion_ref.name:
                for task, input in control_inputs.items():
                    axis = None
                    if task == "walk_vertical":
                        axis = 0
                    elif task == "walk_horizontal":
                        axis = 1
                    elif task == "turn":
                        axis = 2

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

            elif "squat" in motion_ref.name:
                command[:5] = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
                command[5] = np.interp(
                    control_inputs["squat"],
                    [-1, 0, 1],
                    [command_range[5][1], 0.0, command_range[5][0]],
                )

            state_ref = motion_ref.get_state_ref(state_ref, time_curr, command)
            joint_angles = np.asarray(state_ref[13 : 13 + robot.nu])

            if "walk" in motion_ref.name:
                sim.set_torso_quat(np.asarray(state_ref[3:7]))
                sim.set_joint_angles(dict(zip(robot.joint_ordering, joint_angles)))
                sim.forward()
            else:
                motor_angles = robot.joint_to_motor_angles(
                    dict(zip(robot.joint_ordering, joint_angles))
                )
                sim.set_motor_angles(motor_angles)
                sim.step()

            time_curr += sim.control_dt
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
        default="toddlerbot_OP3",
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
        sim.load_keyframe()
    else:
        raise ValueError("Unknown simulator")

    motion_ref: MotionReference | None = None

    if "walk" in args.ref:
        walk_cfg = WalkCfg()
        turn_cfg = TurnCfg()
        command_range = (
            walk_cfg.commands.command_range + turn_cfg.commands.command_range
        )

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
            )

    elif "balance" in args.ref:
        balance_cfg = BalanceCfg()
        command_range = balance_cfg.commands.command_range
        motion_ref = BalancePDReference(
            robot, balance_cfg.sim.timestep * balance_cfg.action.n_frames
        )

    elif "squat" in args.ref:
        squat_cfg = SquatCfg()
        command_range = squat_cfg.commands.command_range
        motion_ref = SquatReference(
            robot, squat_cfg.sim.timestep * squat_cfg.action.n_frames
        )

    else:
        raise ValueError("Unknown ref motion")

    test_motion_ref(robot, sim, motion_ref, command_range)
