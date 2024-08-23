import argparse
import copy
import glob
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
from sklearn.model_selection import train_test_split  # type: ignore

from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import combine_images, find_robot_file_path
from toddlerbot.utils.math_utils import get_sine_signal
from toddlerbot.utils.misc_utils import log
from toddlerbot.visualization.vis_plot import plot_joint_angle_tracking


def extract_data(
    real_world_data_dict: Dict[str, Dict[int, Dict[str, Any]]], joint_name: str
):
    signal_config_list: List[Dict[str, float]] = []
    observed_time_list: List[List[float]] = []
    observed_pos_list: List[List[float]] = []
    for data in real_world_data_dict[joint_name].values():
        signal_config_list.append(data["signal_config"])
        observed_time_list.append(data["joint_data"]["time"])
        observed_pos_list.append(data["joint_data"]["pos"])

    return signal_config_list, observed_time_list, observed_pos_list


def load_datasets(robot: Robot, exp_folder_path: str, test_split_ratio: float = 0.2):
    # Use glob to find all pickle files matching the pattern
    pickle_file_path = os.path.join(exp_folder_path, "log_data.pkl")
    if not os.path.exists(pickle_file_path):
        raise ValueError("No data files found")

    with open(pickle_file_path, "rb") as f:
        data_dict = pickle.load(f)

    obs_list: List[Obs] = data_dict["obs_list"]
    motor_angles_list: List[Dict[str, float]] = data_dict["motor_angles_list"]
    joint_angles_list = [
        robot.motor_to_joint_angles(motor_angles) for motor_angles in motor_angles_list
    ]

    split_obs_dict: Dict[str, npt.NDArray[np.float32]] = {}
    split_action_dict: Dict[str, npt.NDArray[np.float32]] = {}

    def set_obs_action(joint_name: str, idx_range: slice):
        joint_idx = robot.joint_ordering.index(joint_name)
        split_obs_dict[joint_name] = np.array(
            [obs.q[joint_idx] for obs in obs_list[idx_range]]
        )
        split_action_dict[joint_name] = np.array(
            [joint_angles[joint_name] for joint_angles in joint_angles_list[idx_range]]
        )

    if "time_mark_dict" in data_dict:
        time_mark_dict: Dict[str, float] = data_dict["time_mark_dict"]
        joint_names = list(time_mark_dict.keys())
        time_mark_list = list(time_mark_dict.values())
        obs_time = [obs.time for obs in obs_list]
        obs_indices = np.searchsorted(obs_time, time_mark_list)

        last_idx = 0
        for symmetric_name, idx in zip(joint_names, obs_indices):
            if symmetric_name in robot.joint_ordering:
                set_obs_action(symmetric_name, slice(last_idx, idx))
            else:
                set_obs_action(f"left_{symmetric_name}", slice(last_idx, idx))
                set_obs_action(f"right_{symmetric_name}", slice(last_idx, idx))

            last_idx = idx
    else:
        set_obs_action("all", slice(None))

    return split_obs_dict, split_action_dict


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

    joint_angles = robot.init_motor_angles.copy()
    joint_angles[joint_name] = signal_pos[0]

    prep_steps = int(prep_time / sim.dt)
    control_steps = int(control_dt / sim.dt)

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

                    if "n_load" in params:
                        parent_body = None
                        for body in root.findall(".//body"):
                            if joint in body:
                                parent_body = body
                                break

                        if parent_body is not None:
                            parent_body_name = parent_body.get("name", "")
                            if "430" in parent_body_name:
                                inertial_data = inertial_data_430_list[
                                    int(params["n_load"])
                                ]
                            elif "330" in parent_body_name:
                                inertial_data = inertial_data_330_list[
                                    int(params["n_load"])
                                ]
                            else:
                                raise ValueError(
                                    f"Parent body '{parent_body_name}' not found."
                                )

                            # Find the inertial element and update the mass
                            inertial = parent_body.find("inertial")
                            if inertial is not None:
                                current_pos_list = inertial.get("pos", "").split(" ")
                                current_pos_list[0] = inertial_data["pos_x"]
                                inertial.set("pos", " ".join(current_pos_list))
                                inertial.set("mass", inertial_data["mass"])
                                inertial.set(
                                    "diaginertia", inertial_data["diaginertia"]
                                )
                            else:
                                raise ValueError(
                                    f"Inertial element for joint '{name}' not found."
                                )
                        else:
                            raise ValueError(
                                f"Parent body for joint '{name}' not found."
                            )

                    for param_name, param_value in params.items():
                        if param_name in ["damping", "armature", "frictionloss"]:
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


def optimize_parameters(
    robot: Robot,
    sim_name: str,
    joint_name: str,
    tree: ET.ElementTree,
    assets_dict: Dict[str, bytes],
    signal_config_list: List[List[Dict[str, float]]],
    observed_pos_arr: npt.NDArray[np.float32],
    n_load_list: List[int],
    n_iters: int = 1000,
    sampler_name: str = "CMA",
    # gain_range: Tuple[float, float, float] = (0, 50, 0.1),
    damping_range: Tuple[float, float, float] = (0, 5, 1e-3),
    armature_range: Tuple[float, float, float] = (0, 0.1, 1e-3),
    friction_range: Tuple[float, float, float] = (0, 1.0, 1e-3),
):
    def objective(trial: optuna.Trial):
        model_response_all: List[List[List[float]]] = []
        for i, n_load in enumerate(n_load_list):
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
                        "n_load": n_load,
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
                params_dict = {
                    joint_name: {
                        "damping": damping,
                        "friction": friction,
                        "n_load": n_load,
                    }
                }
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

            model_response_list: List[List[float]] = []
            for signal_config in signal_config_list[i]:
                _, signal_pos = get_sine_signal(signal_config)
                joint_data_dict = actuate_single_motor(
                    sim, robot, joint_name, signal_pos, signal_config["control_dt"]
                )
                model_response_list.append(joint_data_dict["pos"])

            sim.close()

            model_response_all.append(model_response_list)

        error = np.sqrt(np.mean((observed_pos_arr - np.array(model_response_all)) ** 2))

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

    # study.enqueue_trial({"damping": 0.099, "armature": 0.002, "frictionloss": 0.239})
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
    sysID_file_path: str,
    sysID_robot_tree: ET.ElementTree,
    assets_dict: Dict[str, bytes],
    signal_config_data: Dict[str, List[List[Dict[str, float]]]],
    observed_pos_data: Dict[str, npt.NDArray[np.float32]],
    n_load_list: List[int],
    n_iters: int,
):
    # return sysID_file_path
    optimize_args: List[
        Tuple[
            Robot,
            str,
            str,
            ET.ElementTree,
            Dict[str, bytes],
            List[List[Dict[str, float]]],
            npt.NDArray[np.float32],
            List[int],
            int,
        ]
    ] = [
        (
            robot,
            sim_name,
            joint_name,
            sysID_robot_tree,
            assets_dict,
            signal_config_data[joint_name],
            observed_pos_data[joint_name],
            n_load_list,
            n_iters,
        )
        for joint_name in signal_config_data
    ]

    # Create a pool of processes
    with Pool(processes=len(signal_config_data)) as pool:
        results = pool.starmap(optimize_parameters, optimize_args)

    time_str = time.strftime("%Y%m%d_%H%M%S")
    opt_params_file_path = os.path.join(exp_folder_path, f"opt_params_{time_str}.json")
    opt_values_file_path = os.path.join(exp_folder_path, f"opt_values_{time_str}.json")

    # Process results
    opt_params_dict: Dict[str, Dict[str, float]] = {}
    opt_values_dict: Dict[str, float] = {}
    for joint_name, result in zip(signal_config_data.keys(), results):
        opt_params, opt_values = result
        if len(opt_params) > 0:
            opt_params_dict[joint_name] = opt_params
            opt_values_dict[joint_name] = opt_values

        with open(opt_params_file_path, "w") as f:
            json.dump(opt_params_dict, f, indent=4)

        with open(opt_values_file_path, "w") as f:
            json.dump(opt_values_dict, f, indent=4)

    for joint_name, opt_params in opt_params_dict.items():
        robot.config["joints"][joint_name].update(opt_params)

    update_xml(sim_name, sysID_robot_tree, opt_params_dict)
    sysID_robot_tree.write(sysID_file_path)

    if sim_name == "mujoco":
        suffix = "_fixed" if robot.config["general"]["is_fixed"] else ""
        xml_path = find_robot_file_path(robot.name, suffix=suffix + ".xml")
        robot_tree = ET.parse(xml_path)
        update_xml(sim_name, robot_tree, opt_params_dict)
        robot_tree.write(xml_path)


def evaluate(
    robot: Robot,
    sim_name: str,
    exp_folder_path: str,
    tree: ET.ElementTree,
    assets_dict: Dict[str, bytes],
    signal_config_data: Dict[str, List[List[Dict[str, float]]]],
    observed_time_data: Dict[str, List[List[List[float]]]],
    observed_pos_data: Dict[str, npt.NDArray[np.float32]],
    n_load_list: List[int],
    dataset_name: str,
):
    for joint_name in signal_config_data:
        model_response_all: List[List[List[float]]] = []

        time_seq_ref_dict: Dict[str, List[float]] = {}
        time_seq_dict: Dict[str, List[float]] = {}
        time_seq_dict_gt: Dict[str, List[float]] = {}
        joint_angle_ref_dict: Dict[str, List[float]] = {}
        joint_angle_dict: Dict[str, List[float]] = {}
        joint_angle_dict_gt: Dict[str, List[float]] = {}
        title_list: List[str] = []

        for i, n_load in enumerate(n_load_list):
            if sim_name == "mujoco":
                from toddlerbot.sim.mujoco_sim import MuJoCoSim

                params_dict = {joint_name: {"n_load": float(n_load)}}
                xml_str = update_xml(sim_name, copy.deepcopy(tree), params_dict)
                sim = MuJoCoSim(robot, xml_str=xml_str, assets=assets_dict)

            elif sim_name == "isaac":
                from toddlerbot.sim.isaac_sim import IsaacSim

                # TODO: update later
                sim = IsaacSim(robot, fixed=True, custom_parameters=custom_parameters)

            else:
                raise ValueError("Invalid simulator")

            model_response_list: List[List[float]] = []
            for trial, signal_config in enumerate(signal_config_data[joint_name][i]):
                signal_time, signal_pos = get_sine_signal(signal_config)

                joint_data_dict = actuate_single_motor(
                    sim, robot, joint_name, signal_pos, signal_config["control_dt"]
                )
                model_response_list.append(joint_data_dict["pos"])

                title_list.append(
                    json.dumps(
                        {
                            "load": n_load,
                            "freq": round(signal_config["frequency"], 3),
                            "amp": round(signal_config["amplitude"], 3),
                        }
                    )
                )
                time_seq_ref_dict[f"l={n_load}_i={trial}"] = list(signal_time)
                time_seq_dict[f"l={n_load}_i={trial}"] = joint_data_dict["time"]
                time_seq_dict_gt[f"l={n_load}_i={trial}"] = observed_time_data[
                    joint_name
                ][i][trial]
                joint_angle_ref_dict[f"l={n_load}_i={trial}"] = list(signal_pos)
                joint_angle_dict[f"l={n_load}_i={trial}"] = joint_data_dict["pos"]
                joint_angle_dict_gt[f"l={n_load}_i={trial}"] = observed_pos_data[
                    joint_name
                ][i][trial]

            sim.close()

            model_response_all.append(model_response_list)

        observed_pos_arr = observed_pos_data[joint_name]
        error = np.sqrt(np.mean((observed_pos_arr - np.array(model_response_all)) ** 2))
        log(
            f"{dataset_name} root mean squared error: {error}",
            header="SysID",
            level="info",
        )

        if dataset_name == "test":
            time_str = time.strftime("%Y%m%d_%H%M%S")
            sim_file_name = f"{joint_name}_sim_{dataset_name}_{time_str}"
            plot_joint_angle_tracking(
                time_seq_dict,
                time_seq_ref_dict,
                joint_angle_dict,
                joint_angle_ref_dict,
                save_path=exp_folder_path,
                file_name=sim_file_name,
                title_list=title_list,
            )

            real_file_name = f"{joint_name}_real_{dataset_name}_{time_str}"
            plot_joint_angle_tracking(
                time_seq_dict_gt,
                time_seq_ref_dict,
                joint_angle_dict_gt,
                joint_angle_ref_dict,
                save_path=exp_folder_path,
                file_name=real_file_name,
                title_list=title_list,
            )

            combined_file_name = f"{joint_name}_combined_{dataset_name}_{time_str}"
            combine_images(
                os.path.join(exp_folder_path, sim_file_name + ".png"),
                os.path.join(exp_folder_path, real_file_name + ".png"),
                os.path.join(exp_folder_path, combined_file_name + ".png"),
            )


def main():
    parser = argparse.ArgumentParser(description="Run the SysID optimization.")
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
        "--n-iters",
        type=int,
        default=1000,
        help="The number of iterations to optimize the parameters.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        required=True,
        help="The name of the run.",
    )
    args = parser.parse_args()

    exp_folder_path = os.path.join("results", args.run_name)
    if not os.path.exists(exp_folder_path):
        raise ValueError("Invalid experiment folder path")

    robot = Robot(args.robot)

    (
        signal_config_train,
        signal_config_test,
        observed_time_train,
        observed_time_test,
        observed_pos_train,
        observed_pos_test,
        n_load_list,
    ) = load_datasets(robot, exp_folder_path)

    if args.sim == "mujoco":
        fixed_xml_path = find_robot_file_path(robot.name, suffix="_fixed.xml")
        sysID_file_path = os.path.join(
            os.path.dirname(fixed_xml_path), f"{robot.name}_{args.sim}_sysID.xml"
        )
        sysID_robot_tree = ET.parse(fixed_xml_path)
        assets_dict: Dict[str, bytes] = {}
        # Find mesh file references
        for mesh in sysID_robot_tree.getroot().findall(".//mesh"):
            mesh_file = mesh.get("file")
            if mesh_file and mesh_file not in assets_dict:
                mesh_file_path = os.path.join(
                    os.path.dirname(fixed_xml_path), mesh_file
                )
                with open(mesh_file_path, "rb") as f:
                    assets_dict[mesh_file] = f.read()

    elif args.sim == "isaac":
        urdf_path = find_robot_file_path(robot.name, suffix="_isaac.urdf")
        sysID_file_path = os.path.join(
            os.path.dirname(urdf_path), f"{robot.name}_{args.sim}_sysID.urdf"
        )
        sysID_robot_tree = ET.parse(urdf_path)
        assets_dict: Dict[str, bytes] = {}

    else:
        raise ValueError("Invalid simulator")

    ###### Optimize the hyperparameters ######
    multiprocessing_optimization(
        robot,
        args.sim,
        args.exp_folder_path,
        sysID_file_path,
        sysID_robot_tree,
        assets_dict,
        signal_config_train,
        observed_pos_train,
        n_load_list,
        args.n_iters,
    )

    ###### Evaluate the optimized parameters in the simulation ######
    sysID_robot_tree = ET.parse(sysID_file_path)
    evaluate(
        robot,
        args.sim,
        args.exp_folder_path,
        sysID_robot_tree,
        assets_dict,
        signal_config_train,
        observed_time_train,
        observed_pos_train,
        n_load_list,
        "train",
    )
    evaluate(
        robot,
        args.sim,
        args.exp_folder_path,
        sysID_robot_tree,
        assets_dict,
        signal_config_test,
        observed_time_test,
        observed_pos_test,
        n_load_list,
        "test",
    )


if __name__ == "__main__":
    main()
