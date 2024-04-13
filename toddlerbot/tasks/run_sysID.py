import argparse
import copy
import json
import os
import pickle
import shutil
import time
from xml.etree.ElementTree import ElementTree, tostring

import numpy as np
import optuna

from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.real_world import RealWorld
from toddlerbot.sim.robot import HumanoidRobot
from toddlerbot.utils.file_utils import find_description_path
from toddlerbot.utils.math_utils import round_floats
from toddlerbot.utils.misc_utils import log, precise_sleep
from toddlerbot.utils.vis_plot import plot_joint_tracking


def update_xml(tree, joint_name, params_dict):
    """
    Update the MuJoCo XML file with new actuator parameters and return it as a string.
    """
    # Load the XML file
    root = tree.getroot()

    # Iterate over all joints in the XML
    joint = root.find(f".//joint[@name='{joint_name}']")
    for param_name, param_value in params_dict.items():
        if param_name in ["damping", "armature", "frictionloss"]:
            joint.set(param_name, str(param_value))

    # Ensure <actuator> element exists
    actuator = root.find("./actuator")
    motor_name = f"{joint_name}_act"
    motor = actuator.find(f".//position[@name='{motor_name}']")
    if motor is None:
        motor = ElementTree.SubElement(actuator, "position", name=motor_name)
    motor.set("kp", str(params_dict["p_gain"]))

    # Convert the updated XML tree back to a string
    xml_string = tostring(root, encoding="unicode")

    return xml_string


def optimize_parameters(
    robot,
    joint_name,
    tree,
    assets_dict,
    joint_data_dict,
    damping_range=(1e-3, 2, 1e-3),
    armature_range=(1e-3, 0.1, 1e-3),
    friction_range=(0, 1, 1e-3),
    p_gain_range=(1, 1000, 0.1),
    sampler="TPE",
    n_trials=500,
):
    signal_config_list = []
    observed_response = []
    for data in joint_data_dict.values():
        signal_config_list.append(data["signal_config"])
        observed_response.append(data["joint_traj"]["pos"])

    observed_response = np.concatenate(observed_response)

    def objective(trial: optuna.Trial):
        # Calculate the model response using the current set of parameters
        damping = trial.suggest_float(
            "damping", *damping_range[:2], step=damping_range[2]
        )
        armature = trial.suggest_float(
            "armature", *armature_range[:2], step=armature_range[2]
        )
        frictionloss = trial.suggest_float(
            "frictionloss", *friction_range[:2], step=friction_range[2]
        )
        p_gain = trial.suggest_float("p_gain", *p_gain_range[:2], step=p_gain_range[2])

        params_dict = {
            "damping": damping,
            "armature": armature,
            "frictionloss": frictionloss,
            "p_gain": p_gain,
        }

        xml_str = update_xml(copy.deepcopy(tree), joint_name, params_dict)
        sim = MuJoCoSim(robot, xml_str=xml_str, assets=assets_dict, fixed=True)
        sim.simulate(headless=True, callback=False)

        model_response = []
        for signal_config in signal_config_list:
            signal_time, signal_pos = generate_sinusoidal_signal(signal_config)
            joint_traj = actuate(
                sim, robot, joint_name, signal_pos, 1 / signal_config["sampling_rate"]
            )
            model_response.extend(joint_traj["pos"])

        sim.close()

        error = np.sqrt(np.mean((observed_response - np.array(model_response)) ** 2))
        return error

    if sampler == "TPE":
        sampler = optuna.samplers.TPESampler()
    elif sampler == "CMA":
        sampler = optuna.samplers.CmaEsSampler()
    else:
        raise ValueError("Invalid sampler")

    study = optuna.create_study(storage="sqlite:///db.sqlite3", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=True)

    return study.best_params


def generate_random_sinusoidal_config(
    duration, frequency_range, amplitude_range, rate_range
):
    frequency = np.random.uniform(*frequency_range)
    amplitude = np.random.uniform(*amplitude_range)
    sampling_rate = np.random.uniform(*rate_range)

    return {
        "frequency": frequency,
        "amplitude": amplitude,
        "duration": duration,
        "sampling_rate": sampling_rate,
    }


def generate_sinusoidal_signal(signal_config):
    """
    Generates a sinusoidal signal based on the given parameters.
    """
    t = np.linspace(
        0,
        signal_config["duration"],
        int(signal_config["duration"] * signal_config["sampling_rate"]),
        endpoint=False,
    )
    signal = signal_config["amplitude"] * np.sin(
        2 * np.pi * signal_config["frequency"] * t
    )
    return t, signal


def actuate(sim, robot, joint_name, signal_pos, sleep_time, prep_time=2):
    """
    Actuates a single joint with the given signal and collects the response.
    """
    # Convert signal time to sleep time between updates
    _, initial_joint_angles = robot.initialize_joint_angles()

    joint_traj_dict = {"pos": [], "time": []}

    sim.set_joint_angles(initial_joint_angles)
    precise_sleep(prep_time)

    time_start = time.time()
    for angle in signal_pos:
        step_start = time.time()

        joint_angles = initial_joint_angles.copy()
        joint_angles[joint_name] = angle

        sim.set_joint_angles(joint_angles)

        joint_state_dict = sim.get_joint_state()

        joint_traj_dict["time"].append(joint_state_dict[joint_name].time - time_start)

        pos = joint_state_dict[joint_name].pos
        if (
            hasattr(sim, "negated_joint_names")
            and joint_name in sim.negated_joint_names
        ):
            pos *= -1

        joint_traj_dict["pos"].append(pos)

        time_until_next_step = sleep_time - (time.time() - step_start)
        if time_until_next_step > 0:
            precise_sleep(time_until_next_step)

    return joint_traj_dict


def collect_data(
    robot,
    joint_name,
    exp_folder_path,
    n_trials=10,
    duration=3,
    frequency_range=(0.5, 2),
    rate_range=(10, 100),
    amplitude_min=np.pi / 8,
):
    real_world = RealWorld(robot)
    amplitude_max = min(
        abs(robot.joints_info[joint_name]["lower_limit"]),
        abs(robot.joints_info[joint_name]["upper_limit"]),
    )

    time_seq_ref_dict = {}
    time_seq_dict = {}
    joint_angle_ref_dict = {}
    joint_angle_dict = {}

    joint_data_dict = {}
    title_list = []
    for trial in range(n_trials):
        signal_config = generate_random_sinusoidal_config(
            duration, frequency_range, (amplitude_min, amplitude_max), rate_range
        )
        signal_time, signal_pos = generate_sinusoidal_signal(signal_config)
        signal_config_rounded = round_floats(signal_config, 3)
        del signal_config_rounded["duration"]
        title_list.append(json.dumps(signal_config_rounded))

        # Actuate the joint and collect data
        log(
            f"Actuating {joint_name} in real with {signal_config}...",
            header="SysID",
            level="debug",
        )
        joint_traj = actuate(
            real_world,
            robot,
            joint_name,
            signal_pos,
            1 / signal_config["sampling_rate"],
        )

        joint_data_dict[trial] = {
            "signal_config": signal_config,
            "joint_traj": joint_traj,
        }

        time_seq_ref_dict[f"trial_{trial}"] = list(signal_time)
        time_seq_dict[f"trial_{trial}"] = joint_traj["time"]
        joint_angle_ref_dict[f"trial_{trial}"] = list(signal_pos)
        joint_angle_dict[f"trial_{trial}"] = joint_traj["pos"]

    real_world.close()

    plot_joint_tracking(
        time_seq_dict,
        time_seq_ref_dict,
        joint_angle_dict,
        joint_angle_ref_dict,
        save_path=exp_folder_path,
        file_name=f"{joint_name}_real_world_tracking",
        title_list=title_list,
    )

    return joint_data_dict


def evaluate(
    robot,
    joint_name,
    new_xml_path,
    joint_data_dict,
    exp_folder_path,
    headless=False,
):
    sim = MuJoCoSim(robot, xml_path=new_xml_path, fixed=True)
    sim.simulate(headless=headless)

    signal_config_list = []
    observed_response = []
    for data in joint_data_dict.values():
        signal_config_list.append(data["signal_config"])
        observed_response.append(data["joint_traj"]["pos"])

    observed_response = np.concatenate(observed_response)

    time_seq_ref_dict = {}
    time_seq_dict = {}
    joint_angle_ref_dict = {}
    joint_angle_dict = {}

    joint_data_dict = {}
    title_list = []

    model_response = []
    for trial, signal_config in enumerate(signal_config_list):
        signal_config = signal_config_list[trial]
        signal_time, signal_pos = generate_sinusoidal_signal(signal_config)

        signal_config_rounded = round_floats(signal_config, 3)
        del signal_config_rounded["duration"]
        title_list.append(json.dumps(signal_config_rounded))
        # Actuate the joint and collect data
        log(
            f"Actuating {joint_name} in {sim.name} "
            + f"with {round_floats(signal_config, 3)}...",
            header="SysID",
            level="debug",
        )
        joint_traj = actuate(
            sim, robot, joint_name, signal_pos, 1 / signal_config["sampling_rate"]
        )
        model_response.extend(joint_traj["pos"])

        joint_data_dict[f"trial_{trial}"] = {
            "signal_config": signal_config,
            "joint_traj": joint_traj,
        }

        time_seq_ref_dict[f"trial_{trial}"] = list(signal_time)
        time_seq_dict[f"trial_{trial}"] = joint_traj["time"]
        joint_angle_ref_dict[f"trial_{trial}"] = list(signal_pos)
        joint_angle_dict[f"trial_{trial}"] = joint_traj["pos"]

    sim.close()

    error = np.sqrt(np.mean((observed_response - np.array(model_response)) ** 2))
    log(f"Root mean squared error: {error}", header="SysID", level="info")

    plot_joint_tracking(
        time_seq_dict,
        time_seq_ref_dict,
        joint_angle_dict,
        joint_angle_ref_dict,
        save_path=exp_folder_path,
        file_name=f"{joint_name}_sim_tracking",
        title_list=title_list,
    )

    return joint_data_dict


def main():
    parser = argparse.ArgumentParser(description="Run the SysID data collection.")
    parser.add_argument(
        "--robot-name",
        type=str,
        default="sustaina_op",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    parser.add_argument(
        "--joint-name",
        type=str,
        help="The name of the joint to perform SysID on.",
    )
    parser.add_argument(
        "--exp-folder-path",
        type=str,
        default="",
        help="The path to the experiment folder.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run the simulation in headless mode.",
    )
    args = parser.parse_args()

    if len(args.exp_folder_path) > 0:
        exp_folder_path = args.exp_folder_path
    else:
        exp_name = f"sysID_{args.robot_name}"
        time_str = time.strftime("%Y%m%d_%H%M%S")
        exp_folder_path = f"results/{time_str}_{exp_name}"

    os.makedirs(exp_folder_path, exist_ok=True)

    robot = HumanoidRobot(args.robot_name)

    # Save the collected data
    real_world_data_file_name = f"{args.joint_name}_real_world_data.pkl"
    real_world_data_file_path = os.path.join(exp_folder_path, real_world_data_file_name)
    if os.path.exists(real_world_data_file_path):
        with open(real_world_data_file_path, "rb") as f:
            real_world_data_dict = pickle.load(f)
    else:
        real_world_data_dict = collect_data(
            robot, args.joint_name, exp_folder_path, n_trials=10
        )
        with open(real_world_data_file_path, "wb") as f:
            pickle.dump(real_world_data_dict, f)

    fixed_xml_path = find_description_path(args.robot_name, suffix="_fixed.xml")
    new_xml_path = os.path.join(
        os.path.dirname(fixed_xml_path), f"{args.robot_name}_sysID.xml"
    )
    opt_params_file_name = f"{args.joint_name}_opt_params.json"
    opt_params_file_path = os.path.join(exp_folder_path, opt_params_file_name)
    if os.path.exists(opt_params_file_path):
        with open(opt_params_file_path, "r") as f:
            opt_params = json.load(f)
    else:
        shutil.copy2(fixed_xml_path, new_xml_path)

        tree = ElementTree()
        tree.parse(new_xml_path)
        root = tree.getroot()

        assets_dict = {}
        # Find mesh file references
        for mesh in root.findall(".//mesh"):
            mesh_file = mesh.get("file")
            if mesh_file and mesh_file not in assets_dict:
                mesh_file_path = os.path.join(
                    os.path.dirname(fixed_xml_path), mesh_file
                )
                with open(mesh_file_path, "rb") as f:
                    assets_dict[mesh_file] = f.read()

        opt_params = optimize_parameters(
            robot,
            args.joint_name,
            tree,
            assets_dict,
            real_world_data_dict,
            sampler="CMA",
            n_trials=1000,
        )

        with open(opt_params_file_path, "w") as f:
            json.dump(opt_params, f, indent=4)

        # opt_params = {
        #     "damping": 0.859,
        #     "armature": 0.041,
        #     "frictionloss": 0.402,
        #     "p_gain": 14.9,
        # }

        update_xml(tree, args.joint_name, opt_params)
        tree.write(new_xml_path)

    sim_data_dict = evaluate(
        robot,
        args.joint_name,
        new_xml_path,
        real_world_data_dict,
        exp_folder_path,
        headless=args.headless,
    )
    sim_data_file_name = f"{args.joint_name}_sim_data.pkl"
    sim_data_file_path = os.path.join(exp_folder_path, sim_data_file_name)

    with open(sim_data_file_path, "wb") as f:
        pickle.dump(sim_data_dict, f)


if __name__ == "__main__":
    main()
