import argparse
import json
import os
import pickle
import time
from typing import Dict, List

import numpy as np

from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import round_floats
from toddlerbot.utils.misc_utils import log, precise_sleep
from toddlerbot.visualization.vis_plot import plot_joint_angle_tracking


def get_random_sine_signal_config(
    duration: float,
    control_dt: float,
    mean: float,
    frequency_range: List[float],
    amplitude_range: List[float],
):
    frequency = np.random.uniform(*frequency_range)  # type: ignore
    amplitude = np.random.uniform(*amplitude_range)  # type: ignore

    sine_signal_config: Dict[str, float] = {
        "frequency": frequency,
        "amplitude": amplitude,
        "duration": duration,
        "control_dt": control_dt,
        "mean": mean,
    }

    return sine_signal_config


def get_sin_signal(sine_signal_config: Dict[str, float]):
    """
    Generates a sinusoidal signal based on the given parameters.
    """
    t = np.linspace(
        0,
        sine_signal_config["duration"],
        int(sine_signal_config["duration"] / sine_signal_config["control_dt"]),
        endpoint=False,
    )
    signal = sine_signal_config["mean"] + sine_signal_config["amplitude"] * np.sin(
        2 * np.pi * sine_signal_config["frequency"] * t
    )
    return t, signal


def actuate_single_motor(sim, robot, joint_name, signal_pos, control_dt, prep_time=1):
    """
    Actuates a single joint with the given signal and collects the response.
    """
    # Convert signal time to sleep time between updates
    joint_data_dict = {"pos": [], "time": []}

    sim.set_joint_angles(initial_joint_angles)
    precise_sleep(prep_time)

    time_start = time.time()
    for idx, angle in enumerate(signal_pos):
        step_start = time.time()

        joint_angles = initial_joint_angles.copy()
        joint_angles[joint_name] = angle

        # log(f"Setting joint {joint_name} to {angle}...", header="SysID", level="debug")
        sim.set_joint_angles(joint_angles)

        joint_state_dict = sim.get_joint_state(
            motor_list=[robot.config.motor_params[joint_name].brand]
        )

        # Assume no control latency
        joint_data_dict["time"].append(idx * control_dt)
        # joint_data_dict["time"].append(
        #     joint_state_dict[joint_name].time - time_start
        # )

        joint_data_dict["pos"].append(joint_state_dict[joint_name].pos)

        time_until_next_step = control_dt - (time.time() - step_start)
        if time_until_next_step > 0:
            # log(
            #     f"Sleeping for {time_until_next_step} s...",
            #     header="SysID",
            #     level="debug",
            # )
            precise_sleep(time_until_next_step)

    # time_end = time.time()
    # log(
    #     f"Actuation duration: {time_end - time_start} s",
    #     header="SysID",
    #     level="debug",
    # )

    if hasattr(sim, "negated_joint_names") and joint_name in sim.negated_joint_names:
        joint_data_dict["pos"] = [-pos for pos in joint_data_dict["pos"]]

    return joint_data_dict


# TODO: Implement when back
def actuate_whole_body(sim, robot, joint_name, signal_pos, control_dt, prep_time=1):
    """
    Actuates a single joint with the given signal and collects the response.
    """
    # Convert signal time to sleep time between updates
    joint_data_dict = {"pos": [], "time": []}

    _, initial_joint_angles = robot.initialize_joint_angles()
    initial_joint_angles[joint_name] = signal_pos[0]

    if joint_name == "left_hip_roll":
        initial_joint_angles["right_hip_roll"] = -np.pi / 4

    if joint_name == "left_hip_pitch" or joint_name == "left_knee":
        initial_joint_angles["left_hip_yaw"] = -np.pi / 4
        initial_joint_angles["right_hip_yaw"] = -np.pi / 4
        initial_joint_angles["right_hip_roll"] = -np.pi / 8

    if joint_name == "left_ank_roll":
        initial_joint_angles["left_ank_pitch"] = np.pi / 6

    if sim.name == "real_world":
        sim.set_joint_angles(initial_joint_angles)
        precise_sleep(prep_time)

        time_start = time.time()
        for idx, angle in enumerate(signal_pos):
            step_start = time.time()

            joint_angles = initial_joint_angles.copy()
            joint_angles[joint_name] = angle

            # log(f"Setting joint {joint_name} to {angle}...", header="SysID", level="debug")
            sim.set_joint_angles(joint_angles)

            joint_state_dict = sim.get_joint_state(
                motor_list=[robot.config.motor_params[joint_name].brand]
            )

            # Assume no control latency
            joint_data_dict["time"].append(idx * control_dt)
            # joint_data_dict["time"].append(
            #     joint_state_dict[joint_name].time - time_start
            # )

            joint_data_dict["pos"].append(joint_state_dict[joint_name].pos)

            time_until_next_step = control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                # log(
                #     f"Sleeping for {time_until_next_step} s...",
                #     header="SysID",
                #     level="debug",
                # )
                precise_sleep(time_until_next_step)

        # time_end = time.time()
        # log(
        #     f"Actuation duration: {time_end - time_start} s",
        #     header="SysID",
        #     level="debug",
        # )

        if (
            hasattr(sim, "negated_joint_names")
            and joint_name in sim.negated_joint_names
        ):
            joint_data_dict["pos"] = [-pos for pos in joint_data_dict["pos"]]

    return joint_data_dict


def collect_data(
    robot: Robot,
    joint_name: str,
    exp_folder_path: str,
    n_trials: int,
    duration: float = 3,
    control_dt: float = 0.01,
    frequency_range: List[float] = [0.5, 2],
    amplitude_min: float = np.pi / 12,
):
    from toddlerbot.sim.real_world import RealWorld

    real_world = RealWorld(robot)
    lower_limit = robot.joints_info[joint_name]["lower_limit"]
    upper_limit = robot.joints_info[joint_name]["upper_limit"]

    if joint_name == "left_ank_pitch":
        lower_limit = 0.0
        upper_limit = np.pi / 6

    mean = (lower_limit + upper_limit) / 2
    amplitude_max = upper_limit - mean

    time_seq_ref_dict = {}
    time_seq_dict = {}
    joint_angle_ref_dict = {}
    joint_angle_dict = {}

    real_world_data_dict = {}
    title_list: List[str] = []
    for trial in range(n_trials):
        signal_config = get_random_sine_signal_config(
            duration, control_dt, mean, frequency_range, [amplitude_min, amplitude_max]
        )
        signal_time, signal_pos = get_sin_signal(signal_config)
        signal_config_rounded = round_floats(signal_config, 3)
        del signal_config_rounded["duration"]
        del signal_config_rounded["control_dt"]
        del signal_config_rounded["mean"]
        title_list.append(json.dumps(signal_config_rounded))

        # Actuate the joint and collect data
        log(
            f"Actuating {joint_name} in real with {signal_config}...",
            header="SysID",
            level="debug",
        )
        joint_data_dict = actuate_single_motor(
            real_world, robot, joint_name, signal_pos, control_dt
        )

        real_world_data_dict[trial] = {
            "signal_config": signal_config,
            "joint_data": joint_data_dict,
        }

        time_seq_ref_dict[f"trial_{trial}"] = list(signal_time)
        time_seq_dict[f"trial_{trial}"] = joint_data_dict["time"]
        joint_angle_ref_dict[f"trial_{trial}"] = list(signal_pos)
        joint_angle_dict[f"trial_{trial}"] = joint_data_dict["pos"]

    real_world.close()

    plot_joint_angle_tracking(
        time_seq_dict,
        time_seq_ref_dict,
        joint_angle_dict,
        joint_angle_ref_dict,
        save_path=exp_folder_path,
        file_name=f"{joint_name}_real_world_tracking",
        title_list=title_list,
    )

    return real_world_data_dict


def main():
    parser = argparse.ArgumentParser(description="Run the SysID data collection.")
    parser.add_argument(
        "--robot-name",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    parser.add_argument(
        "--joint-names",
        type=str,
        nargs="+",  # Indicates that one or more values are expected
        help="The names of the joints to perform SysID on.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=15,
        help="The number of trials to collect data for.",
    )
    parser.add_argument(
        "--n-loads",
        type=int,
        default=0,
        help="The number of loads in the load box.",
    )
    parser.add_argument(
        "--exp-folder-path",
        type=str,
        default="",
        help="The path to the experiment folder.",
    )
    args = parser.parse_args()

    if len(args.exp_folder_path) > 0:
        exp_folder_path = args.exp_folder_path
    else:
        exp_phrases: List[str] = []
        if "sysID" in args.robot_name:
            exp_phrases.append(args.robot_name)
        else:
            exp_phrases.append(f"sysID_{args.robot_name}")

        exp_phrases.append("J=" + "_".join(args.joint_names))
        exp_phrases.append("N=" + str(args.n_trials))
        exp_phrases.append("L=" + str(args.n_loads))

        exp_name = "_".join(exp_phrases)
        time_str = time.strftime("%Y%m%d_%H%M%S")
        exp_folder_path = f"results/{time_str}_{exp_name}"

    os.makedirs(exp_folder_path, exist_ok=True)

    args_dict = vars(args)
    # Save to JSON file
    with open(os.path.join(exp_folder_path, "args.json"), "w") as json_file:
        json.dump(args_dict, json_file, indent=4)

    robot = Robot(args.robot_name)

    ###### Collect data in the real world ######
    real_world_data_file_path = os.path.join(exp_folder_path, "real_world_data.pkl")
    real_world_data_dict = {}

    is_collected = False
    if os.path.exists(real_world_data_file_path):
        is_collected = True
        with open(real_world_data_file_path, "rb") as f:
            real_world_data_dict = pickle.load(f)

        for joint_name in args.joint_names:
            if joint_name not in real_world_data_dict:
                is_collected = False
                break

    if not is_collected:
        for joint_name in args.joint_names:
            if joint_name in real_world_data_dict:
                continue

            real_world_data_dict[joint_name] = collect_data(
                robot, joint_name, exp_folder_path, n_trials=args.n_trials
            )
            # Save the data in intermediate steps
            with open(real_world_data_file_path, "wb") as f:
                pickle.dump(real_world_data_dict, f)


if __name__ == "__main__":
    main()
