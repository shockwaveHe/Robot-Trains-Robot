import argparse
import importlib
import os
import pickle
import pkgutil
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Type

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from toddlerbot.policies import BasePolicy
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


def import_all_policies():
    policies: Dict[str, Type[BasePolicy]] = {}
    package_path = os.path.join("toddlerbot", "policies")
    package_name = "policies"

    # Iterate over all Python files in the package directory
    for _, module_name, _ in pkgutil.iter_modules([package_path]):
        module = importlib.import_module(f"toddlerbot.{package_name}.{module_name}")
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if (
                isinstance(attribute, type)
                and issubclass(attribute, BasePolicy)
                and attribute != BasePolicy
            ):
                policies[module_name] = attribute
                break

    return policies


POLICIES = import_all_policies()


def run_policy(
    policy_name: str,
    state: Dict[str, npt.NDArray[np.float32]],
    last_action: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    policy = POLICIES[policy_name]()
    return policy.run(state, last_action)


def fetch_state(
    sim: BaseSim,
    obs_queue: Deque[Dict[str, npt.NDArray[np.float32]]],
    obs_stop_event: threading.Event,
):
    while not obs_stop_event.is_set():
        step_start = time.time()

        obs_dict = sim.get_observation()
        obs_queue.append(obs_dict)

        step_time = time.time() - step_start
        print(f"Obs latency: {step_time * 1000:.2f} ms")

        time_until_next_step = sim.dt - step_time
        if time_until_next_step > 0:
            # precise_sleep will block until the time has passed
            time.sleep(time_until_next_step)


def main():
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
        "--policy",
        type=str,
        default="stand",
        help="The name of the task.",
    )
    args = parser.parse_args()

    debug: dict[str, bool] = {"log": False, "plot": True, "render": True}

    exp_name = f"stand_{args.robot}_{args.sim}"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    exp_folder_path = f"results/{time_str}_{exp_name}"

    robot = Robot(args.robot)

    default_act = np.array(list(robot.init_motor_angles.values()), dtype=np.float32)
    action = np.zeros_like(default_act)
    target_act = default_act + action

    default_q = np.array(list(robot.init_joint_angles.values()), dtype=np.float32)
    header_name = snake2camel(f"sim2{args.sim}")

    time_seq_ref_dict: Dict[str, List[float]] = {}
    time_seq_dict: Dict[str, List[float]] = {}
    joint_angle_ref_dict: Dict[str, List[float]] = {}
    joint_angle_dict: Dict[str, List[float]] = {}
    joint_vel_dict: Dict[str, List[float]] = {}

    time_obs_list: List[float] = []
    euler_obs_list: List[npt.NDArray[np.float32]] = []
    ang_vel_obs_list: List[npt.NDArray[np.float32]] = []

    if args.sim == "mujoco":
        from toddlerbot.sim.mujoco_sim import MuJoCoSim

        sim = MuJoCoSim(robot)
        sim.simulate(vis_type="render" if debug["render"] else "none")
    elif args.sim == "real":
        from toddlerbot.sim.real_world import RealWorld

        sim = RealWorld(robot)
    else:
        raise ValueError("Unknown simulator")

    obs_queue: Deque[Dict[str, npt.NDArray[np.float32]]] = deque()
    obs_stop_event = threading.Event()
    state_thread = threading.Thread(
        target=fetch_state, args=(sim, obs_queue, obs_stop_event)
    )
    state_thread.start()

    while len(obs_queue) < 10:
        time.sleep(0.1)

    step_idx = 0
    duration = 60.0  # float("inf")
    p_bar = tqdm(total=duration / sim.control_dt, desc="Running the policy")
    try:
        while step_idx < duration / sim.control_dt:
            step_start = time.time()

            step_time = step_idx * sim.control_dt

            # Get the latest state from the queue
            obs_dict = obs_queue.pop()
            print(len(obs_queue))

            time_obs_list.append(obs_dict["imu_time"].item())
            euler_obs_list.append(obs_dict["imu_euler"])
            ang_vel_obs_list.append(obs_dict["imu_ang_vel"])

            for i, joint_name in enumerate(robot.joint_ordering):
                if joint_name not in time_seq_dict:
                    time_seq_ref_dict[joint_name] = []
                    time_seq_dict[joint_name] = []
                    joint_angle_dict[joint_name] = []
                    joint_vel_dict[joint_name] = []

                # Assume the state fetching is instantaneous
                time_seq_ref_dict[joint_name].append(step_time)
                time_seq_dict[joint_name].append(obs_dict["time"][i])
                joint_angle_dict[joint_name].append(obs_dict["q"][i])
                joint_vel_dict[joint_name].append(obs_dict["dq"][i])

            action = run_policy(args.policy, obs_dict, action)
            target_act = default_act + action

            motor_angles: Dict[str, float] = {}
            for name, target_act_pos in zip(robot.motor_ordering, target_act):
                motor_angles[name] = target_act_pos

            sim.set_motor_angles(motor_angles)

            joint_angle_ref = robot.motor_to_joint_angles(motor_angles)
            for joint_name, joint_angle in joint_angle_ref.items():
                if joint_name not in joint_angle_ref_dict:
                    joint_angle_ref_dict[joint_name] = []

                joint_angle_ref_dict[joint_name].append(joint_angle)

            q_obs_delta = obs_dict["q"] - default_q

            step_idx += 1
            step_time = time.time() - step_start

            p_bar_steps = int(1 / sim.control_dt)
            if step_idx % p_bar_steps == 0:
                p_bar.update(p_bar_steps)

            if debug["log"]:
                log(
                    f"q: {round_floats(q_obs_delta, 3)}",
                    header=header_name,
                    level="debug",
                )
                log(
                    f"dq: {round_floats(obs_dict['dq'], 3)}",
                    header=header_name,
                    level="debug",
                )
                log(
                    f"euler: {obs_dict['imu_euler']}", header=header_name, level="debug"
                )
                log(
                    f"ang_vel: {obs_dict['imu_ang_vel']}",
                    header=header_name,
                    level="debug",
                )

                log(f"Joint angles: {motor_angles}", header=header_name, level="debug")
                log(
                    f"Control latency: {step_time * 1000:.2f} ms",
                    header=header_name,
                    level="debug",
                )

            print(f"Control latency: {step_time * 1000:.2f} ms")

            time_until_next_step = sim.control_dt - step_time
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

        if debug["render"] and hasattr(sim, "save_recording"):
            sim.save_recording(exp_folder_path)  # type: ignore

        sim.close()

        prof_path = os.path.join(exp_folder_path, "profile_output.lprof")
        dump_profiling_data(prof_path)

        log_data_path = os.path.join(exp_folder_path, "log_data.pkl")
        log_data_dict = {
            "time_seq_dict": time_seq_dict,
            "time_seq_ref_dict": time_seq_ref_dict,
            "joint_angle_dict": joint_angle_dict,
            "joint_angle_ref_dict": joint_angle_ref_dict,
            "joint_vel_dict": joint_vel_dict,
            "time_obs_list": time_obs_list,
            "euler_obs_list": euler_obs_list,
            "ang_vel_obs_list": ang_vel_obs_list,
        }
        with open(log_data_path, "wb") as f:
            pickle.dump(log_data_dict, f)

        if debug["plot"]:
            log("Visualizing...", header="Walking")
            plot_orientation_tracking(
                time_obs_list,
                euler_obs_list,
                save_path=exp_folder_path,
            )
            plot_angular_velocity_tracking(
                time_obs_list,
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
