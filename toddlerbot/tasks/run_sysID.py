import argparse
import copy
import json
import os
import pickle
import time
from xml.etree.ElementTree import ElementTree, tostring

import numpy as np
import optuna

from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.real_world import RealWorld
from toddlerbot.sim.robot import HumanoidRobot
from toddlerbot.utils.constants import MUJOCO_TIMESTEP
from toddlerbot.utils.file_utils import find_description_path
from toddlerbot.utils.math_utils import round_floats
from toddlerbot.utils.misc_utils import log, precise_sleep
from toddlerbot.utils.vis_plot import plot_joint_tracking


def generate_random_sinusoidal_config(
    duration, control_dt, frequency_range, amplitude_range
):
    frequency = np.random.uniform(*frequency_range)
    amplitude = np.random.uniform(*amplitude_range)

    return {
        "frequency": frequency,
        "amplitude": amplitude,
        "duration": duration,
        "control_dt": control_dt,
    }


def generate_sinusoidal_signal(signal_config):
    """
    Generates a sinusoidal signal based on the given parameters.
    """
    t = np.linspace(
        0,
        signal_config["duration"],
        int(signal_config["duration"] / signal_config["control_dt"]),
        endpoint=False,
    )
    signal = signal_config["amplitude"] * np.sin(
        2 * np.pi * signal_config["frequency"] * t
    )
    return t, signal


# @profile
def actuate(sim, robot, joint_name, signal_pos, control_dt, prep_time=1):
    """
    Actuates a single joint with the given signal and collects the response.
    """
    # Convert signal time to sleep time between updates
    joint_data_dict = {"pos": [], "time": []}
    _, initial_joint_angles = robot.initialize_joint_angles()

    if sim.name == "real_world":
        sim.set_joint_angles(initial_joint_angles)
        precise_sleep(prep_time)

        time_start = time.time()
        for angle in signal_pos:
            step_start = time.time()

            joint_angles = initial_joint_angles.copy()
            joint_angles[joint_name] = angle

            # log(f"Setting joint {joint_name} to {angle}...", header="SysID", level="debug")
            sim.set_joint_angles(joint_angles)

            joint_state_dict = sim.get_joint_state()

            joint_data_dict["time"].append(
                joint_state_dict[joint_name].time - time_start
            )

            pos = joint_state_dict[joint_name].pos
            if (
                hasattr(sim, "negated_joint_names")
                and joint_name in sim.negated_joint_names
            ):
                pos *= -1

            joint_data_dict["pos"].append(pos)

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

    elif sim.name == "mujoco":
        prep_steps = int(prep_time / MUJOCO_TIMESTEP)
        control_steps = int(control_dt / MUJOCO_TIMESTEP)

        joint_control_traj = []
        for _ in range(prep_steps):
            joint_control_traj.append(initial_joint_angles)

        for angle in signal_pos:
            joint_angles = initial_joint_angles.copy()
            joint_angles[joint_name] = angle
            for _ in range(control_steps):
                joint_control_traj.append(joint_angles)

        joint_state_traj = sim.rollout(joint_control_traj)

        joint_data_dict = {"pos": [], "time": []}
        time_start = joint_state_traj[prep_steps][joint_name].time
        for i in range(prep_steps, len(joint_state_traj), control_steps):
            joint_state_dict = joint_state_traj[i]
            joint_data_dict["time"].append(
                joint_state_dict[joint_name].time - time_start
            )
            joint_data_dict["pos"].append(joint_state_dict[joint_name].pos)

    return joint_data_dict


def collect_data(
    robot,
    joint_name,
    exp_folder_path,
    n_trials=10,
    duration=3,
    control_dt=0.04,
    frequency_range=(0.5, 2),
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

    real_world_data_dict = {}
    title_list = []
    for trial in range(n_trials):
        signal_config = generate_random_sinusoidal_config(
            duration, control_dt, frequency_range, (amplitude_min, amplitude_max)
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
        joint_data_dict = actuate(real_world, robot, joint_name, signal_pos, control_dt)

        real_world_data_dict[trial] = {
            "signal_config": signal_config,
            "joint_data": joint_data_dict,
        }

        time_seq_ref_dict[f"trial_{trial}"] = list(signal_time)
        time_seq_dict[f"trial_{trial}"] = joint_data_dict["time"]
        joint_angle_ref_dict[f"trial_{trial}"] = list(signal_pos)
        joint_angle_dict[f"trial_{trial}"] = joint_data_dict["pos"]

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

    return real_world_data_dict


def update_xml(tree, params_dict):
    """
    Update the MuJoCo XML file with new actuator parameters and return it as a string.
    """
    # Load the XML file
    root = tree.getroot()

    for joint_name, params in params_dict.items():
        if "left" in joint_name:
            joint_name_symmetric = joint_name.replace("left", "right")
        elif "right" in joint_name:
            joint_name_symmetric = joint_name.replace("right", "left")

        for name in [joint_name, joint_name_symmetric]:
            # Find the joint by name
            joint = root.find(f".//joint[@name='{name}']")
            if joint is not None:
                # Update the joint with new parameters
                for param_name, param_value in params.items():
                    joint.set(param_name, str(param_value))
            else:
                print(f"Joint '{name}' not found in the XML tree.")

    # Convert the updated XML tree back to a string
    xml_string = tostring(root, encoding="unicode")

    return xml_string


def optimize_parameters(
    robot,
    joint_name,
    tree,
    assets_dict,
    signal_config_list,
    observed_response,
    damping_range=(0, 2, 1e-3),
    armature_range=(0, 0.1, 1e-4),
    friction_range=(0, 1, 1e-3),
    sampler="TPE",
    n_iters=500,
):
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
        params_dict = {
            joint_name: {
                "damping": damping,
                "armature": armature,
                "frictionloss": frictionloss,
            }
        }
        xml_str = update_xml(copy.deepcopy(tree), params_dict)
        sim = MuJoCoSim(robot, xml_str=xml_str, assets=assets_dict, fixed=True)

        model_response = []
        for signal_config in signal_config_list:
            _, signal_pos = generate_sinusoidal_signal(signal_config)
            joint_data_dict = actuate(
                sim, robot, joint_name, signal_pos, signal_config["control_dt"]
            )
            model_response.extend(joint_data_dict["pos"])

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
    study.optimize(objective, n_trials=n_iters, n_jobs=-1, show_progress_bar=True)

    return study.best_params


# @profile
def evaluate(
    robot,
    joint_name,
    new_xml_path,
    signal_config_list,
    observed_response,
    exp_folder_path,
):
    sim = MuJoCoSim(robot, xml_path=new_xml_path, fixed=True)

    time_seq_ref_dict = {}
    time_seq_dict = {}
    joint_angle_ref_dict = {}
    joint_angle_dict = {}

    sim_data_dict = {}
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
        joint_data_dict = actuate(
            sim, robot, joint_name, signal_pos, signal_config["control_dt"]
        )
        model_response.extend(joint_data_dict["pos"])

        sim_data_dict[f"trial_{trial}"] = {
            "signal_config": signal_config,
            "joint_data": joint_data_dict,
        }

        time_seq_ref_dict[f"trial_{trial}"] = list(signal_time)
        time_seq_dict[f"trial_{trial}"] = joint_data_dict["time"]
        joint_angle_ref_dict[f"trial_{trial}"] = list(signal_pos)
        joint_angle_dict[f"trial_{trial}"] = joint_data_dict["pos"]

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

    return sim_data_dict


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
        default=5,
        help="The number of trials to collect data for.",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=1000,
        help="The number of iterations to optimize the parameters.",
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
        exp_name = f"sysID_{args.robot_name}"
        time_str = time.strftime("%Y%m%d_%H%M%S")
        exp_folder_path = f"results/{time_str}_{exp_name}"

    os.makedirs(exp_folder_path, exist_ok=True)

    args_dict = vars(args)

    # Save to JSON file
    with open(os.path.join(exp_folder_path, "args.json"), "w") as json_file:
        json.dump(args_dict, json_file, indent=4)

    robot = HumanoidRobot(args.robot_name)

    ###### Collect data in the real world ######
    real_world_data_file_path = os.path.join(exp_folder_path, "real_world_data.pkl")
    if os.path.exists(real_world_data_file_path):
        with open(real_world_data_file_path, "rb") as f:
            real_world_data_dict = pickle.load(f)
    else:
        real_world_data_dict = {}
        for joint_name in args.joint_names:
            real_world_data_dict[joint_name] = collect_data(
                robot, joint_name, exp_folder_path, n_trials=args.n_trials
            )
        with open(real_world_data_file_path, "wb") as f:
            pickle.dump(real_world_data_dict, f)

    ###### Optimize the hyperparameters ######
    fixed_xml_path = find_description_path(args.robot_name, suffix="_fixed.xml")
    new_xml_path = os.path.join(
        os.path.dirname(fixed_xml_path), f"{args.robot_name}_sysID.xml"
    )
    opt_params_file_path = os.path.join(exp_folder_path, "opt_params.json")
    if os.path.exists(opt_params_file_path):
        with open(opt_params_file_path, "r") as f:
            opt_params_dict = json.load(f)
    else:
        tree = ElementTree()
        tree.parse(fixed_xml_path)
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

        opt_params_dict = {}
        for joint_name in args.joint_names:
            signal_config_list = []
            observed_response = []
            for data in real_world_data_dict[joint_name].values():
                signal_config_list.append(data["signal_config"])
                observed_response.append(data["joint_traj"]["pos"])

            observed_response = np.concatenate(observed_response)

            opt_params_dict[joint_name] = optimize_parameters(
                robot,
                joint_name,
                tree,
                assets_dict,
                signal_config_list,
                observed_response,
                sampler="CMA",
                n_iters=args.n_iters,
            )

        with open(opt_params_file_path, "w") as f:
            json.dump(opt_params_dict, f, indent=4)

        # opt_params_dict = {
        #     "left_hip_yaw": {
        #         "damping": 0.043,
        #         "armature": 0.001,
        #         "frictionloss": 0.213,
        #     },
        #     "left_hip_roll": {
        #         "damping": 0.357,
        #         "armature": 0.006,
        #         "frictionloss": 0.073,
        #     },
        #     "left_hip_pitch": {
        #         "damping": 0.327,
        #         "armature": 0.005,
        #         "frictionloss": 0.037,
        #     },
        # }

        update_xml(tree, opt_params_dict)
        tree.write(new_xml_path)

    ###### Evaluate the optimized parameters in the simulation ######
    sim_data_dict = {}
    for joint_name in args.joint_names:
        signal_config_list = []
        observed_response = []
        for data in real_world_data_dict[joint_name].values():
            signal_config_list.append(data["signal_config"])
            observed_response.append(data["joint_traj"]["pos"])

        observed_response = np.concatenate(observed_response)

        sim_data_dict[joint_name] = evaluate(
            robot,
            joint_name,
            new_xml_path,
            signal_config_list,
            observed_response,
            exp_folder_path,
        )

    sim_data_file_path = os.path.join(exp_folder_path, "sim_data.pkl")
    with open(sim_data_file_path, "wb") as f:
        pickle.dump(sim_data_dict, f)


if __name__ == "__main__":
    main()
