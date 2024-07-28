import argparse
import os
import pickle
import threading
import time
from collections import deque
from typing import Deque, Dict, List

import numpy as np
import numpy.typing as npt

from toddlerbot.sim import BaseSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import round_floats
from toddlerbot.utils.misc_utils import (
    dump_profiling_data,
    log,
    precise_sleep,
    snake2camel,
)
from toddlerbot.visualization.vis_plot import (
    plot_angular_velocity_tracking,
    plot_joint_angle_tracking,
    plot_joint_velocity_tracking,
    plot_orientation_tracking,
)


def run_policy(
    task: str,
    state: Dict[str, npt.NDArray[np.float32]],
    last_action: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    if task == "stand":
        return np.zeros_like(last_action)
    else:
        raise ValueError("Unknown task")


def fetch_state(
    sim: BaseSim,
    obs_queue: Deque[Dict[str, npt.NDArray[np.float32]]],
    obs_stop_event: threading.Event,
):
    while not obs_stop_event.is_set():
        step_start = time.time()

        obs_dict = sim.get_observation()
        obs_queue.append(obs_dict)

        time_until_next_step = sim.dt - (time.time() - step_start)
        if time_until_next_step > 0:
            # precise_sleep will block until the time has passed
            time.sleep(time_until_next_step)


def main():
    parser = argparse.ArgumentParser(description="Run the walking simulation.")
    parser.add_argument(
        "--robot-name",
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
    args = parser.parse_args()

    exp_name = f"stand_{args.robot_name}_{args.sim}"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    exp_folder_path = f"results/{time_str}_{exp_name}"

    # A 0.3725 offset moves the robot slightly up from the ground
    robot = Robot(args.robot_name)

    if args.sim == "pybullet":
        from toddlerbot.sim.pybullet_sim import PyBulletSim

        sim = PyBulletSim(robot)
    elif args.sim == "mujoco":
        from toddlerbot.sim.mujoco_sim import MuJoCoSim

        sim = MuJoCoSim(robot)
    elif args.sim == "real":
        from toddlerbot.sim.real_world import RealWorld

        sim = RealWorld(robot)
    else:
        raise ValueError("Unknown simulator")

    initial_joint_angles = robot.initialize_motor_angles()

    duration = float("inf")
    control_dt = 0.01
    joint_ordering = list(initial_joint_angles.keys())
    default_q = np.array(initial_joint_angles.values())
    action = np.zeros((len(joint_ordering),), dtype=np.float32)
    target_q = default_q + action

    header_name = snake2camel(f"sim2{sim.name}")
    debug = False

    time_seq_ref_dict: Dict[str, List[float]] = {}
    time_seq_dict: Dict[str, List[float]] = {}
    joint_angle_ref_dict: Dict[str, List[float]] = {}
    joint_angle_dict: Dict[str, List[float]] = {}
    joint_vel_dict: Dict[str, List[float]] = {}

    time_list: List[float] = []
    euler_obs_list: List[npt.NDArray[np.float32]] = []
    ang_vel_obs_list: List[npt.NDArray[np.float32]] = []

    obs_queue: Deque[Dict[str, npt.NDArray[np.float32]]] = deque()
    obs_stop_event = threading.Event()
    state_thread = threading.Thread(
        target=fetch_state, args=(sim, joint_ordering, obs_queue, obs_stop_event)
    )
    state_thread.start()

    while len(obs_queue) < 10:
        time.sleep(0.1)

    step_idx = 0
    try:
        while step_idx < duration / control_dt:
            step_start = time.time()

            # Get the latest state from the queue
            state = obs_queue.pop()
            q_obs = state["q_obs"]
            dq_obs = state["dq_obs"]
            euler_obs = state["euler_obs"]
            ang_vel_obs = state["ang_vel_obs"]

            q_obs_delta = q_obs - default_q

            time_list.append(step_idx * control_dt)
            euler_obs_list.append(euler_obs)
            ang_vel_obs_list.append(ang_vel_obs)

            for i, joint_name in enumerate(joint_ordering):
                if joint_name not in time_seq_dict:
                    time_seq_ref_dict[joint_name] = []
                    time_seq_dict[joint_name] = []
                    joint_angle_dict[joint_name] = []
                    joint_vel_dict[joint_name] = []

                # Assume the state fetching is instantaneous
                time_seq_ref_dict[joint_name].append(step_idx * control_dt)
                time_seq_dict[joint_name].append(step_idx * control_dt)
                joint_angle_dict[joint_name].append(q_obs[i])
                joint_vel_dict[joint_name].append(dq_obs[i])

            action = run_policy("stand", state, action)
            target_q = default_q

            joint_angles: Dict[str, float] = {}
            for name, target_angle in zip(joint_ordering, target_q):
                joint_angles[name] = target_angle

                if name not in joint_angle_ref_dict:
                    joint_angle_ref_dict[name] = []

                joint_angle_ref_dict[name].append(target_angle)

            if debug:
                log(
                    f"q: {round_floats(q_obs_delta, 3)}",
                    header=header_name,
                    level="debug",
                )
                log(
                    f"dq: {round_floats(dq_obs, 3)}",
                    header=header_name,
                    level="debug",
                )
                log(
                    f"euler: {euler_obs}",
                    header=header_name,
                    level="debug",
                )
                log(f"ang_vel: {ang_vel_obs}", header=header_name, level="debug")

                log(f"Joint angles: {joint_angles}", header=header_name, level="debug")

            sim.set_motor_angles(joint_angles)

            step_idx += 1

            step_time = time.time() - step_start
            log(
                f"Control Frequency: {1 / step_time:.2f} Hz",
                header=header_name,
                level="debug",
            )
            time_until_next_step = control_dt - step_time
            if time_until_next_step > 0:
                precise_sleep(time_until_next_step)

    except KeyboardInterrupt:
        log("KeyboardInterrupt recieved. Closing...", header=header_name)

    finally:
        obs_stop_event.set()
        state_thread.join()

        exp_name = f"sim2{sim.name}_{robot.name}"
        time_str = time.strftime("%Y%m%d_%H%M%S")
        exp_folder_path = f"results/{time_str}_{exp_name}"

        os.makedirs(exp_folder_path, exist_ok=True)

        if hasattr(sim.visualizer, "save_recording"):
            sim.visualizer.save_recording(exp_folder_path)  # type: ignore
        else:
            log(
                "Current visualizer does not support video writing.",
                header=header_name,
                level="warning",
            )

        sim.close()

        prof_path = os.path.join(exp_folder_path, "profile_output.lprof")
        dump_profiling_data(prof_path)

        log_data_path = os.path.join(exp_folder_path, "log_data.pkl")
        log_data_dict = {
            "time_seq_dict": time_seq_dict,
            "time_seq_ref_dict": time_seq_ref_dict,
            "dof_pos_dict": joint_angle_dict,
            "dof_pos_ref_dict": joint_angle_ref_dict,
            "dof_vel_dict": joint_vel_dict,
            "euler_obs_list": euler_obs_list,
            "ang_vel_obs_list": ang_vel_obs_list,
        }
        with open(log_data_path, "wb") as f:
            pickle.dump(log_data_dict, f)

        log("Visualizing...", header="Walking")
        plot_orientation_tracking(
            time_list,
            euler_obs_list,
            save_path=exp_folder_path,
        )
        plot_angular_velocity_tracking(
            time_list,
            ang_vel_obs_list,
            save_path=exp_folder_path,
        )
        plot_joint_angle_tracking(
            time_seq_dict,
            time_seq_ref_dict,
            joint_angle_dict,
            joint_angle_ref_dict,
            save_path=exp_folder_path,
        )
        plot_joint_velocity_tracking(
            time_seq_dict,
            joint_vel_dict,
            save_path=exp_folder_path,
        )


if __name__ == "__main__":
    main()
