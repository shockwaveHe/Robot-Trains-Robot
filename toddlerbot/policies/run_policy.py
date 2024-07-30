import argparse
import importlib
import os
import pickle
import pkgutil
import time
from typing import Any, Dict, List, Type

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
    profile,
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


def plot_results(
    obs_dict_list: List[Dict[str, npt.NDArray[np.float32]]],
    motor_angles_list: List[Dict[str, float]],
    exp_folder_path: str,
):
    time_obs_list: List[float] = []
    euler_obs_list: List[npt.NDArray[np.float32]] = []
    ang_vel_obs_list: List[npt.NDArray[np.float32]] = []
    time_seq_dict: Dict[str, List[float]] = {}
    time_seq_ref_dict: Dict[str, List[float]] = {}
    joint_angle_dict: Dict[str, List[float]] = {}
    joint_vel_dict: Dict[str, List[float]] = {}
    for i, obs_dict in enumerate(obs_dict_list):
        obs_time = obs_dict["time"].item()
        time_obs_list.append(obs_time)
        euler_obs_list.append(obs_dict["imu_euler"])
        ang_vel_obs_list.append(obs_dict["imu_ang_vel"])

        for j, joint_name in enumerate(robot.joint_ordering):
            if joint_name not in time_seq_dict:
                time_seq_ref_dict[joint_name] = []
                time_seq_dict[joint_name] = []
                joint_angle_dict[joint_name] = []
                joint_vel_dict[joint_name] = []

            # Assume the state fetching is instantaneous
            time_seq_dict[joint_name].append(obs_time)
            time_seq_ref_dict[joint_name].append(i * sim.control_dt)
            joint_angle_dict[joint_name].append(obs_dict["q"][j])
            joint_vel_dict[joint_name].append(obs_dict["dq"][j])

    joint_angle_ref_dict: Dict[str, List[float]] = {}
    for motor_angles in motor_angles_list:
        joint_angle_ref = robot.motor_to_joint_angles(motor_angles)
        for joint_name, joint_angle in joint_angle_ref.items():
            if joint_name not in joint_angle_ref_dict:
                joint_angle_ref_dict[joint_name] = []
            joint_angle_ref_dict[joint_name].append(joint_angle)

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


def run_policy(
    policy: BasePolicy,
    state: Dict[str, npt.NDArray[np.float32]],
    last_action: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    return policy.run(state, last_action)


# @profile()
def main(robot: Robot, sim: BaseSim, policy: BasePolicy, debug: Dict[str, Any]):
    exp_name = f"stand_{robot.name}_{sim.name}"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    exp_folder_path = f"results/{time_str}_{exp_name}"

    header_name = snake2camel(f"sim2{sim.name}")

    # default_q = np.array(list(robot.init_joint_angles.values()), dtype=np.float32)

    default_act = np.array(list(robot.init_motor_angles.values()), dtype=np.float32)
    action = np.zeros_like(default_act)
    target_act = default_act + action

    obs_dict_list: List[Dict[str, npt.NDArray[np.float32]]] = []
    motor_angles_list: List[Dict[str, float]] = []

    step_idx = 0
    p_bar = tqdm(total=debug["duration"] / sim.control_dt, desc="Running the policy")
    try:
        while step_idx < debug["duration"] / sim.control_dt:
            step_start = time.time()

            # Get the latest state from the queue
            obs_dict = sim.get_observation()
            # q_obs_delta = obs_dict["q"] - default_q

            action = run_policy(policy, obs_dict, action)
            target_act = default_act + action

            motor_angles: Dict[str, float] = {}
            for motor_name, target_act_pos in zip(robot.motor_ordering, target_act):
                motor_angles[motor_name] = target_act_pos

            sim.set_motor_angles(motor_angles)

            obs_dict_list.append(obs_dict)
            motor_angles_list.append(motor_angles)

            step_time = time.time() - step_start
            step_idx += 1

            p_bar_steps = int(1 / sim.control_dt)
            if step_idx % p_bar_steps == 0:
                p_bar.update(p_bar_steps)

            if debug["log"]:
                log(
                    f"obs_dict: {round_floats(obs_dict, 4)}",
                    header=header_name,
                    level="debug",
                )
                log(
                    f"Joint angles: {round_floats(motor_angles,4)}",
                    header=header_name,
                    level="debug",
                )
                log(
                    f"Control latency: {step_time * 1000:.2f} ms",
                    header=header_name,
                    level="debug",
                )

            time_until_next_step = sim.control_dt - step_time
            if time_until_next_step > 0:
                precise_sleep(time_until_next_step)

    except KeyboardInterrupt:
        log("KeyboardInterrupt recieved. Closing...", header=header_name)

    finally:
        exp_name = f"sim2{sim.name}_{robot.name}"
        time_str = time.strftime("%Y%m%d_%H%M%S")
        exp_folder_path = f"results/{time_str}_{exp_name}"

        os.makedirs(exp_folder_path, exist_ok=True)

        if debug["render"] and hasattr(sim, "save_recording"):
            sim.save_recording(exp_folder_path)  # type: ignore

        sim.close()

        prof_path = os.path.join(exp_folder_path, "profile_output.lprof")
        dump_profiling_data(prof_path)

        log_data_dict = {
            "obs_dict_list": obs_dict_list,
            "motor_angles_list": motor_angles_list,
        }
        log_data_path = os.path.join(exp_folder_path, "log_data.pkl")
        with open(log_data_path, "wb") as f:
            pickle.dump(log_data_dict, f)

        if debug["plot"]:
            log("Visualizing...", header="Walking")
            plot_results(obs_dict_list, motor_angles_list, exp_folder_path)


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
        "--policy",
        type=str,
        default="stand",
        help="The name of the task.",
    )
    args = parser.parse_args()

    debug: Dict[str, Any] = {
        "duration": float("inf"),
        "log": False,
        "plot": True,
        "render": True,
    }

    robot = Robot(args.robot)

    if args.sim == "mujoco":
        from toddlerbot.sim.mujoco_sim import MuJoCoSim

        sim = MuJoCoSim(robot)
        sim.simulate(vis_type="render" if debug["render"] else "none")
    elif args.sim == "real":
        from toddlerbot.sim.real_world import RealWorld

        sim = RealWorld(robot)
    else:
        raise ValueError("Unknown simulator")

    policy = POLICIES[args.policy]()

    main(robot, sim, policy, debug)
