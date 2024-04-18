import argparse
import json
import os
import pickle
import time
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import numpy as np

from toddlerbot.sim.robot import HumanoidRobot
from toddlerbot.tasks.sysID import (
    collect_data,
    evaluate,
    optimize_parameters,
    update_xml,
)
from toddlerbot.utils.file_utils import find_description_path


def multiprocessing_optimization(
    robot, joint_names, tree, assets_dict, real_world_data_dict, n_iters
):
    # Prepare a list of arguments for each joint
    joint_args = [
        (robot, joint_name, tree, assets_dict, real_world_data_dict, n_iters)
        for joint_name in joint_names
    ]

    # Create a pool of processes
    with Pool(processes=len(joint_names)) as pool:
        results = pool.starmap(optimize_parameters, joint_args)

    # Process results
    opt_params_dict = {}
    opt_values_dict = {}
    for joint_name, result in zip(joint_names, results):
        opt_params, opt_values = result
        if opt_params is not None:
            opt_params_dict[joint_name] = opt_params
            opt_values_dict[joint_name] = opt_values

    return opt_params_dict, opt_values_dict


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

    if "all" in args.joint_names:
        args.joint_names = [
            "left_hip_yaw",
            "left_hip_roll",
            "left_hip_pitch",
            "left_knee",
            "left_ank_roll",
            "left_ank_pitch",
        ]

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

    ###### Optimize the hyperparameters ######
    fixed_xml_path = find_description_path(args.robot_name, suffix="_fixed.xml")
    new_fixed_xml_path = os.path.join(
        os.path.dirname(fixed_xml_path), f"{args.robot_name}_fixed_sysID.xml"
    )
    xml_path = find_description_path(args.robot_name, suffix=".xml")
    new_xml_path = os.path.join(
        os.path.dirname(xml_path), f"{args.robot_name}_sysID.xml"
    )
    fixed_tree = ET.parse(fixed_xml_path)
    opt_params_file_path = os.path.join(exp_folder_path, "opt_params.json")
    opt_values_file_path = os.path.join(exp_folder_path, "opt_values.json")

    opt_params_dict = {}
    opt_values_dict = {}
    is_optimized = False
    if os.path.exists(opt_params_file_path) and os.path.exists(opt_values_file_path):
        is_optimized = True
        with open(opt_params_file_path, "r") as f:
            opt_params_dict = json.load(f)

        with open(opt_values_file_path, "r") as f:
            opt_values_dict = json.load(f)

        for joint_name in args.joint_names:
            if joint_name not in opt_params_dict:
                is_optimized = False
                break

    if not is_optimized:
        assets_dict = {}
        # Find mesh file references
        for mesh in fixed_tree.getroot().findall(".//mesh"):
            mesh_file = mesh.get("file")
            if mesh_file and mesh_file not in assets_dict:
                mesh_file_path = os.path.join(
                    os.path.dirname(fixed_xml_path), mesh_file
                )
                with open(mesh_file_path, "rb") as f:
                    assets_dict[mesh_file] = f.read()

        opt_params_dict, opt_values_dict = multiprocessing_optimization(
            robot,
            args.joint_names,
            fixed_tree,
            assets_dict,
            real_world_data_dict,
            n_iters=args.n_iters,
        )

        with open(opt_params_file_path, "w") as f:
            json.dump(opt_params_dict, f, indent=4)

        with open(opt_values_file_path, "w") as f:
            json.dump(opt_values_dict, f, indent=4)

    update_xml(fixed_tree, opt_params_dict)
    fixed_tree.write(new_fixed_xml_path)

    tree = ET.parse(xml_path)
    update_xml(tree, opt_params_dict)
    tree.write(new_xml_path)

    ###### Evaluate the optimized parameters in the simulation ######
    sim_data_dict = {}
    for joint_name in args.joint_names:
        signal_config_list = []
        observed_response = []
        for data in real_world_data_dict[joint_name].values():
            signal_config_list.append(data["signal_config"])
            observed_response.append(data["joint_data"]["pos"])

        observed_response = np.concatenate(observed_response)

        sim_data_dict[joint_name] = evaluate(
            robot,
            joint_name,
            new_fixed_xml_path,
            signal_config_list,
            observed_response,
            exp_folder_path,
        )

    sim_data_file_path = os.path.join(exp_folder_path, "sim_data.pkl")
    with open(sim_data_file_path, "wb") as f:
        pickle.dump(sim_data_dict, f)


if __name__ == "__main__":
    main()
