import argparse
import copy
import json
import os
import pickle
import time
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import optuna

from toddlerbot.sim import BaseSim
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.sysID.collect_data import get_sine_signal
from toddlerbot.utils.constants import SIM_TIMESTEP
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.math_utils import round_floats
from toddlerbot.utils.misc_utils import log
from toddlerbot.visualization.vis_plot import plot_joint_angle_tracking

# TODO: Remove the hardcoded custom_parameters
custom_parameters = [
    {"name": "--robot-name", "type": str, "default": "toddlerbot_legs"},
    {"name": "--sim", "type": str, "default": "mujoco"},
    {"name": "--joint-names", "type": str, "default": "all"},
    {"name": "--n-trials", "type": int, "default": 15},
    {"name": "--n-iters", "type": int, "default": 1000},
    {"name": "--exp-folder-path", "type": str, "default": ""},
]


def actuate_single_motor(
    sim: BaseSim,
    robot: Robot,
    joint_name: str,
    signal_pos: npt.NDArray[np.float32],
    control_dt: float,
    prep_time: float = 1,
):
    """
    Actuates a single joint with the given signal and collects the response.
    """
    # Convert signal time to sleep time between updates
    joint_data_dict: Dict[str, List[float]] = {"pos": [], "time": []}

    joint_angles = robot.init_joint_angles.copy()
    joint_angles[joint_name] = signal_pos[0]

    prep_steps = int(prep_time / SIM_TIMESTEP)
    control_steps = int(control_dt / SIM_TIMESTEP)

    joint_angles_list: List[Dict[str, float]] = []
    for _ in range(prep_steps):
        joint_angles_list.append(joint_angles)

    for joint_angle in signal_pos:
        joint_angles_copy = joint_angles.copy()
        joint_angles_copy[joint_name] = joint_angle
        for _ in range(control_steps):
            joint_angles_list.append(joint_angles_copy)

    joint_state_list = sim.rollout(joint_angles_list)

    joint_data_dict = {"pos": [], "time": []}
    time_start = joint_state_list[prep_steps][joint_name].time
    for i in range(prep_steps, len(joint_state_list), control_steps):
        joint_state_dict = joint_state_list[i]
        joint_data_dict["time"].append(joint_state_dict[joint_name].time - time_start)
        joint_data_dict["pos"].append(joint_state_dict[joint_name].pos)

    return joint_data_dict


def update_xml(
    sim_name: str, tree: ET.ElementTree, params_dict: Dict[str, Dict[str, float]]
):
    """
    Update the MuJoCo XML file with new actuator parameters and return it as a string.
    """
    # Load the XML file
    root = tree.getroot()

    for joint_name, params in params_dict.items():
        joint_name_pair = [joint_name]
        if "left" in joint_name:
            joint_name_pair.append(joint_name.replace("left", "right"))
        elif "right" in joint_name:
            joint_name_pair.append(joint_name.replace("right", "left"))

        for name in joint_name_pair:
            # Find the joint by name
            joint = root.find(f".//joint[@name='{name}']")
            if joint is not None:
                if sim_name == "mujoco":
                    # Update the joint with new parameters
                    if "gain" in params:
                        actuator = root.find(f".//position[@name='{name}_act']")
                        if actuator is not None:
                            actuator.set("kp", str(params["gain"]))
                        else:
                            raise ValueError(
                                f"Actuator '{name}' not found in the XML tree."
                            )

                    for param_name, param_value in params.items():
                        if param_name != "gain":
                            joint.set(param_name, str(param_value))

                elif sim_name == "isaac":
                    dynamics = joint.find("dynamics")
                    if dynamics is None:
                        dynamics = ET.Element("dynamics")
                        joint.append(dynamics)

                    # Update the joint with new parameters
                    for param_name, param_value in params.items():
                        dynamics.set(param_name, str(param_value))

                else:
                    raise ValueError("Invalid simulator")
            else:
                raise ValueError(f"Joint '{name}' not found in the XML tree.")

    # Convert the updated XML tree back to a string
    xml_string = ET.tostring(root, encoding="unicode")

    return xml_string


def extract_data(
    real_world_data_dict: Dict[str, Dict[int, Dict[str, Any]]], joint_name: str
):
    signal_config_list: List[Dict[str, float]] = []
    observed_response_list: List[List[float]] = []
    for data in real_world_data_dict[joint_name].values():
        signal_config_list.append(data["signal_config"])
        observed_response_list.append(data["joint_data"]["pos"])

    observed_response_arr: npt.NDArray[np.float32] = np.concatenate(
        observed_response_list
    )

    return signal_config_list, observed_response_arr


def optimize_parameters(
    robot: Robot,
    sim_name: str,
    joint_name: str,
    tree: ET.ElementTree,
    assets_dict: Dict[str, bytes],
    signal_config_list: List[Dict[str, float]],
    observed_response_arr: npt.NDArray[np.float32],
    n_iters: int = 1000,
    sampler_name: str = "CMA",
    # gain_range: Tuple[float, float, float] = (0, 50, 0.1),
    damping_range: Tuple[float, float, float] = (0, 5, 1e-3),
    armature_range: Tuple[float, float, float] = (0, 0.1, 1e-3),
    friction_range: Tuple[float, float, float] = (0, 1.0, 1e-3),
):
    def objective(trial: optuna.Trial):
        if sim_name == "mujoco":
            from toddlerbot.sim.mujoco_sim import MuJoCoSim

            # gain = trial.suggest_float("gain", *gain_range[:2], step=gain_range[2])
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
                    # "gain": gain,
                    "damping": damping,
                    "armature": armature,
                    "frictionloss": frictionloss,
                }
            }
            xml_str = update_xml(sim_name, copy.deepcopy(tree), params_dict)
            sim = MuJoCoSim(robot, xml_str=xml_str, assets=assets_dict)

        elif sim_name == "isaac":
            from toddlerbot.sim.isaac_sim import IsaacSim

            damping = trial.suggest_float(
                "damping", *damping_range[:2], step=damping_range[2]
            )
            friction = trial.suggest_float(
                "friction", *friction_range[:2], step=friction_range[2]
            )
            params_dict = {joint_name: {"damping": damping, "friction": friction}}
            xml_str = update_xml(sim_name, copy.deepcopy(tree), params_dict)
            time_str = time.strftime("%Y%m%d_%H%M%S")
            urdf_path_temp = os.path.join(
                "toddlerbot",
                "robot_descriptions",
                robot.name,
                f"{robot.name}_{sim_name}_{joint_name}_sysID_{time_str}.urdf",
            )
            with open(urdf_path_temp, "w") as f:
                f.write(xml_str)

            sim = IsaacSim(
                robot,
                urdf_path=urdf_path_temp,
                fixed=True,
                custom_parameters=custom_parameters,
            )

            os.remove(urdf_path_temp)

        else:
            raise ValueError("Invalid simulator")

        model_response: List[float] = []
        for signal_config in signal_config_list:
            _, signal_pos = get_sine_signal(signal_config)
            joint_data_dict = actuate_single_motor(
                sim, robot, joint_name, signal_pos, signal_config["control_dt"]
            )
            model_response.extend(joint_data_dict["pos"])

        sim.close()

        error = np.sqrt(
            np.mean((observed_response_arr - np.array(model_response)) ** 2)
        )

        return error

    if sampler_name == "TPE":
        sampler = optuna.samplers.TPESampler()
    elif sampler_name == "CMA":
        sampler = optuna.samplers.CmaEsSampler()
    else:
        raise ValueError("Invalid sampler")

    time_str = time.strftime("%Y%m%d_%H%M%S")
    storage = "postgresql://optuna_user:password@localhost/optuna_db"
    study = optuna.create_study(
        study_name=f"{robot.name}_{joint_name}_{time_str}",
        storage=storage,
        sampler=sampler,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_iters, n_jobs=1, show_progress_bar=True)

    log(
        f"Best parameters: {study.best_params}; best value: {study.best_value}",
        header="SysID",
        level="info",
    )

    return study.best_params, study.best_value


def multiprocessing_optimization(
    robot: Robot,
    sim_name: str,
    exp_folder_path: str,
    joint_names: List[str],
    n_iters: int,
):
    real_world_data_file_path = os.path.join(exp_folder_path, "real_world_data.pkl")
    if not os.path.exists(real_world_data_file_path):
        raise ValueError("Real world data not found")

    with open(real_world_data_file_path, "rb") as f:
        real_world_data_dict = pickle.load(f)

    if sim_name == "mujoco":
        fixed_xml_path = find_robot_file_path(robot.name, suffix="_fixed.xml")
        sysID_file_path = os.path.join(
            os.path.dirname(fixed_xml_path), f"{robot.name}_{sim_name}_sysID.xml"
        )
        sysID_robot_tree = ET.parse(fixed_xml_path)
        assets_dict = {}
        # Find mesh file references
        for mesh in sysID_robot_tree.getroot().findall(".//mesh"):
            mesh_file = mesh.get("file")
            if mesh_file and mesh_file not in assets_dict:
                mesh_file_path = os.path.join(
                    os.path.dirname(fixed_xml_path), mesh_file
                )
                with open(mesh_file_path, "rb") as f:
                    assets_dict[mesh_file] = f.read()

    elif sim_name == "isaac":
        urdf_path = find_robot_file_path(robot.name, suffix="_isaac.urdf")
        sysID_file_path = os.path.join(
            os.path.dirname(urdf_path), f"{robot.name}_{sim_name}_sysID.urdf"
        )
        sysID_robot_tree = ET.parse(urdf_path)
        assets_dict = {}

    else:
        raise ValueError("Invalid simulator")

    # return sysID_file_path

    optimize_args: List[
        Tuple[
            Robot,
            str,
            str,
            ET.ElementTree,
            Dict[str, bytes],
            List[Dict[str, float]],
            npt.NDArray[np.float32],
            int,
        ]
    ] = [
        (
            robot,
            sim_name,
            joint_name,
            sysID_robot_tree,
            assets_dict,
            *extract_data(real_world_data_dict, joint_name),
            n_iters,
        )
        for joint_name in joint_names
    ]

    # Create a pool of processes
    with Pool(processes=len(joint_names)) as pool:
        results = pool.starmap(optimize_parameters, optimize_args)

    time_str = time.strftime("%Y%m%d_%H%M%S")
    opt_params_file_path = os.path.join(exp_folder_path, f"opt_params_{time_str}.json")
    opt_values_file_path = os.path.join(exp_folder_path, f"opt_values_{time_str}.json")

    # Process results
    opt_params_dict: Dict[str, Dict[str, float]] = {}
    opt_values_dict: Dict[str, float] = {}
    for joint_name, result in zip(joint_names, results):
        opt_params, opt_values = result
        if len(opt_params) > 0:
            opt_params_dict[joint_name] = opt_params
            opt_values_dict[joint_name] = opt_values

        with open(opt_params_file_path, "w") as f:
            json.dump(opt_params_dict, f, indent=4)

        with open(opt_values_file_path, "w") as f:
            json.dump(opt_values_dict, f, indent=4)

    update_xml(sim_name, sysID_robot_tree, opt_params_dict)
    sysID_robot_tree.write(sysID_file_path)

    for joint_name, opt_params in opt_params_dict.items():
        robot.config["joints"][joint_name].update(opt_params)

    robot.write_robot_config()

    if sim_name == "mujoco":
        suffix = "_fixed" if robot.config["general"]["is_fixed"] else ""
        xml_path = find_robot_file_path(robot.name, suffix=suffix + ".xml")
        robot_tree = ET.parse(xml_path)
        update_xml(sim_name, robot_tree, opt_params_dict)
        robot_tree.write(xml_path)

    return sysID_file_path


def evaluate(
    robot: Robot,
    sim_name: str,
    exp_folder_path: str,
    joint_name: str,
    sysID_file_path: str,
):
    real_world_data_file_path = os.path.join(exp_folder_path, "real_world_data.pkl")
    if not os.path.exists(real_world_data_file_path):
        raise ValueError("Real world data not found")

    with open(real_world_data_file_path, "rb") as f:
        real_world_data_dict = pickle.load(f)

    signal_config_list, observed_response_arr = extract_data(
        real_world_data_dict, joint_name
    )

    if sim_name == "mujoco":
        from toddlerbot.sim.mujoco_sim import MuJoCoSim

        sim = MuJoCoSim(robot, xml_path=sysID_file_path)

    elif sim_name == "isaac":
        from toddlerbot.sim.isaac_sim import IsaacSim

        sim = IsaacSim(
            robot,
            urdf_path=sysID_file_path,
            fixed=True,
            custom_parameters=custom_parameters,
        )

    else:
        raise ValueError("Invalid simulator")

    time_seq_ref_dict: Dict[str, List[float]] = {}
    time_seq_dict: Dict[str, List[float]] = {}
    joint_angle_ref_dict: Dict[str, List[float]] = {}
    joint_angle_dict: Dict[str, List[float]] = {}

    sim_data_dict: Dict[int, Dict[str, Any]] = {}
    title_list: List[str] = []
    model_response_list: List[float] = []
    for trial, signal_config in enumerate(signal_config_list):
        signal_config = signal_config_list[trial]
        signal_time, signal_pos = get_sine_signal(signal_config)

        # Actuate the joint and collect data
        log(
            f"Actuating {joint_name} in {sim.name} "
            + f"with {round_floats(signal_config, 3)}...",
            header="SysID",
            level="debug",
        )
        joint_data_dict = actuate_single_motor(
            sim, robot, joint_name, signal_pos, signal_config["control_dt"]
        )
        model_response_list.extend(joint_data_dict["pos"])

        sim_data_dict[trial] = {
            "signal_config": signal_config,
            "joint_data": joint_data_dict,
        }

        title_list.append(
            json.dumps(
                {
                    "freq": round(signal_config["frequency"], 3),
                    "amp": round(signal_config["amplitude"], 3),
                }
            )
        )
        time_seq_ref_dict[f"trial_{trial}"] = list(signal_time)
        time_seq_dict[f"trial_{trial}"] = joint_data_dict["time"]
        joint_angle_ref_dict[f"trial_{trial}"] = list(signal_pos)
        joint_angle_dict[f"trial_{trial}"] = joint_data_dict["pos"]

    sim.close()

    error = np.sqrt(
        np.mean((observed_response_arr - np.array(model_response_list)) ** 2)
    )
    log(f"Root mean squared error: {error}", header="SysID", level="info")

    time_str = time.strftime("%Y%m%d_%H%M%S")
    plot_joint_angle_tracking(
        time_seq_dict,
        time_seq_ref_dict,
        joint_angle_dict,
        joint_angle_ref_dict,
        save_path=exp_folder_path,
        file_name=f"{joint_name}_sim_tracking_{time_str}",
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
        "--sim",
        type=str,
        default="mujoco",
        help="The simulator to use.",
    )
    parser.add_argument(
        "--joint-names",
        type=str,
        nargs="+",  # Indicates that one or more values are expected
        help="The names of the joints to perform SysID on.",
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

    if len(args.exp_folder_path) == 0 or not os.path.exists(args.exp_folder_path):
        raise ValueError("Invalid experiment folder path")

    robot = Robot(args.robot_name)

    ###### Optimize the hyperparameters ######
    sysID_file_path = multiprocessing_optimization(
        robot, args.sim, args.exp_folder_path, args.joint_names, args.n_iters
    )

    ###### Evaluate the optimized parameters in the simulation ######
    sim_data_dict = {}
    for joint_name in args.joint_names:
        sim_data_dict[joint_name] = evaluate(
            robot, args.sim, args.exp_folder_path, joint_name, sysID_file_path
        )

    time_str = time.strftime("%Y%m%d_%H%M%S")
    sim_data_file_path = os.path.join(args.exp_folder_path, f"sim_data_{time_str}.pkl")
    with open(sim_data_file_path, "wb") as f:
        pickle.dump(sim_data_dict, f)


if __name__ == "__main__":
    main()
