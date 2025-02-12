import argparse
import json
import os
import pickle
import time
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import optuna
from optuna.logging import _get_library_root_logger

from toddlerbot.sim import Obs
from toddlerbot.sim.mujoco_control import MotorController
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.misc_utils import log
from toddlerbot.visualization.vis_plot import (
    plot_joint_tracking,
    plot_joint_tracking_frequency,
)

# This script is used to optimize the parameters of the robot's dynamics model using system identification (SysID) techniques.

logger = _get_library_root_logger()


def load_datasets(robot: Robot, data_path: str):
    """Loads and processes datasets from a specified path for a given robot, extracting observation positions, actions, and motor gains.

    Args:
        robot (Robot): The robot instance containing motor and joint configurations.
        data_path (str): The directory path where the dataset files are located.

    Returns:
        Tuple[Dict[str, List[npt.NDArray[np.float32]]], Dict[str, List[npt.NDArray[np.float32]]], Dict[str, List[float]]]:
        A tuple containing three dictionaries:
            - obs_pos_dict: Maps joint names to lists of observation position arrays.
            - action_dict: Maps joint names to lists of action arrays.
            - kp_dict: Maps joint names to lists of motor gain values.

    Raises:
        ValueError: If no data files are found at the specified path.
    """

    # Use glob to find all pickle files matching the pattern
    pickle_file_path = os.path.join(data_path, "log_data.pkl")
    if not os.path.exists(pickle_file_path):
        raise ValueError("No data files found")

    with open(pickle_file_path, "rb") as f:
        data_dict = pickle.load(f)

    obs_list: List[Obs] = data_dict["obs_list"]
    motor_angles_list: List[Dict[str, float]] = data_dict["motor_angles_list"]

    obs_pos_dict: Dict[str, List[npt.NDArray[np.float32]]] = {}
    action_dict: Dict[str, List[npt.NDArray[np.float32]]] = {}
    kp_dict: Dict[str, List[float]] = {}

    def set_obs_and_action(
        joint_name: str, motor_kps: Dict[str, float], idx_range: slice
    ):
        kp = motor_kps.get(joint_name, 0)

        obs_pos_list: List[List[float]] = []
        for obs in obs_list[idx_range]:
            motor_angles_obs = dict(zip(robot.motor_ordering, obs.motor_pos))
            joint_angles_obs = robot.motor_to_joint_angles(motor_angles_obs)
            obs_pos_list.append(list(joint_angles_obs.values()))

        obs_pos = np.array(obs_pos_list)

        action = np.array(
            [
                list(motor_angles.values())
                for motor_angles in motor_angles_list[idx_range]
            ]
        )

        if joint_name not in obs_pos_dict:
            obs_pos_dict[joint_name] = []
            action_dict[joint_name] = []
            kp_dict[joint_name] = []

        obs_pos_dict[joint_name].append(obs_pos)
        action_dict[joint_name].append(action)
        kp_dict[joint_name].append(kp)

    if "ckpt_dict" in data_dict:
        ckpt_dict: Dict[str, Dict[str, float]] = data_dict["ckpt_dict"]
        ckpt_times = list(ckpt_dict.keys())
        motor_kps_list: List[Dict[str, float]] = []
        joint_names_list: List[List[str]] = []
        for d in list(ckpt_dict.values()):
            motor_kps_list.append(d)
            joint_names_list.append(list(d.keys()))

        obs_time = [obs.time for obs in obs_list]
        obs_indices = np.searchsorted(obs_time, ckpt_times)

        last_idx = 200
        for joint_names, motor_kps, obs_idx in zip(
            joint_names_list, motor_kps_list, obs_indices
        ):
            for joint_name in joint_names:
                # if "ank_roll" in joint_name:
                #     break
                set_obs_and_action(joint_name, motor_kps, slice(last_idx, obs_idx))

            last_idx = obs_idx
    else:
        start_idx = 300
        for joint_name in reversed(robot.joint_ordering):
            joints_config = robot.config["joints"]
            if joints_config[joint_name]["group"] == "leg":
                motor_names = robot.joint_to_motor_name[joint_name]
                motor_kps = {joint_name: joints_config[motor_names[0]]["kp_real"]}
                set_obs_and_action(joint_name, motor_kps, slice(start_idx, None))

    return obs_pos_dict, action_dict, kp_dict


def optimize_parameters(
    robot: Robot,
    sim_name: str,
    joint_name: str,
    obs_list: List[npt.NDArray[np.float32]],
    action_list: List[npt.NDArray[np.float32]],
    kp_list: List[float],
    n_iters: int = 1000,
    early_stop_rounds: int = 200,
    freq_max: float = 10,
    sampler_name: str = "CMA",
    # gain_range: Tuple[float, float, float] = (0, 50, 0.1),
    damping_range: Tuple[float, float, float] = (0.0, 0.5, 1e-3),
    armature_range: Tuple[float, float, float] = (0.0, 0.01, 1e-4),
    frictionloss_range: Tuple[float, float, float] = (0.0, 1.0, 1e-3),
    q_dot_tau_max_range: Tuple[float, float, float] = (0.0, 5.0, 1e-2),
    q_dot_max_range: Tuple[float, float, float] = (5.0, 10.0, 1e-1),
):
    """Optimize the parameters of a robot joint using simulation and Optuna.

    This function performs parameter optimization for a specified joint of a robot using a simulation environment. It utilizes Optuna for hyperparameter tuning to minimize the error between simulated and observed joint positions.

    Args:
        robot (Robot): The robot object containing joint and motor information.
        sim_name (str): The name of the simulation environment, currently supports "mujoco".
        joint_name (str): The name of the joint to optimize.
        obs_list (List[npt.NDArray[np.float32]]): List of observed joint positions.
        action_list (List[npt.NDArray[np.float32]]): List of actions applied to the joint.
        kp_list (List[float]): List of proportional gains for the motor.
        n_iters (int, optional): Number of optimization iterations. Defaults to 1000.
        early_stop_rounds (int, optional): Number of rounds for early stopping. Defaults to 200.
        freq_max (float, optional): Maximum frequency for filtering in Fourier Transform. Defaults to 10.
        sampler_name (str, optional): Name of the Optuna sampler to use. Defaults to "CMA".
        damping_range (Tuple[float, float, float], optional): Range for damping parameter. Defaults to (0.0, 0.5, 1e-3).
        armature_range (Tuple[float, float, float], optional): Range for armature parameter. Defaults to (0.0, 0.01, 1e-4).
        frictionloss_range (Tuple[float, float, float], optional): Range for friction loss parameter. Defaults to (0.0, 1.0, 1e-3).
        q_dot_tau_max_range (Tuple[float, float, float], optional): Range for q_dot_tau_max parameter. Defaults to (0.0, 5.0, 1e-2).
        q_dot_max_range (Tuple[float, float, float], optional): Range for q_dot_max parameter. Defaults to (5.0, 10.0, 1e-1).

    Returns:
        Tuple[Dict[str, float], float]: The best parameters found and the corresponding error value.

    Raises:
        ValueError: If an invalid simulator or sampler is specified.
    """

    if sim_name == "mujoco":
        sim = MuJoCoSim(robot, fixed_base=True)

    else:
        raise ValueError("Invalid simulator")

    if "sysID" in robot.name:
        tau_max_range: Tuple[float, float, float] = (0.0, 2.0, 1e-2)
        if "XC330" in robot.name:
            tau_max_range = (0.0, 1.0, 1e-2)
        elif "XM430" in robot.name:
            tau_max_range = (0.0, 3.0, 1e-2)

    motor_names = robot.joint_to_motor_name[joint_name]
    joint_idx = robot.joint_ordering.index(joint_name)
    joint_pos_real = np.concatenate([obs[:, joint_idx] for obs in obs_list])

    def early_stop_check(
        study: optuna.Study, trial: optuna.Trial, early_stopping_rounds: int
    ):
        """Checks if the current trial should trigger early stopping based on the number of rounds since the best trial.

        Args:
            study (optuna.Study): The study object containing all trials.
            trial (optuna.Trial): The current trial being evaluated.
            early_stopping_rounds (int): The number of rounds to wait before stopping after the best trial.

        Logs a debug message and stops the study if early stopping conditions are met.
        """
        current_trial_number = trial.number
        best_trial_number = study.best_trial.number
        should_stop = (
            current_trial_number - best_trial_number
        ) >= early_stopping_rounds
        if should_stop:
            logger.debug(f"early stopping detected: {should_stop}")
            study.stop()

    def objective(trial: optuna.Trial):
        """Optimize simulation parameters to minimize the error between simulated and real joint positions.

        This function uses Optuna to suggest values for various simulation parameters, including damping, armature, and friction loss, to optimize the joint dynamics of a robot simulation. If the robot's name contains "sysID", additional motor dynamics parameters are also optimized. The function calculates the root mean square error (RMSE) between the simulated and real joint positions and performs a Fourier Transform to compare the frequency domain characteristics, returning a combined error metric.

        Args:
            trial (optuna.Trial): An Optuna trial object used to suggest parameter values.

        Returns:
            float: The combined error metric, consisting of the RMSE and a weighted frequency domain error.
        """
        # gain = trial.suggest_float("gain", *gain_range[:2], step=gain_range[2])
        damping = trial.suggest_float(
            "damping", *damping_range[:2], step=damping_range[2]
        )
        armature = trial.suggest_float(
            "armature", *armature_range[:2], step=armature_range[2]
        )
        frictionloss = trial.suggest_float(
            "frictionloss", *frictionloss_range[:2], step=frictionloss_range[2]
        )
        joint_dyn = {
            joint_name: dict(
                damping=damping, armature=armature, frictionloss=frictionloss
            )
        }
        sim.set_joint_dynamics(joint_dyn)

        if "sysID" in robot.name:
            tau_max = trial.suggest_float(
                "tau_max", *tau_max_range[:2], step=tau_max_range[2]
            )
            q_dot_tau_max = trial.suggest_float(
                "q_dot_tau_max", *q_dot_tau_max_range[:2], step=q_dot_tau_max_range[2]
            )
            q_dot_max = trial.suggest_float(
                "q_dot_max", *q_dot_max_range[:2], step=q_dot_max_range[2]
            )
            sim.set_motor_dynamics(
                dict(tau_max=tau_max, q_dot_tau_max=q_dot_tau_max, q_dot_max=q_dot_max)
            )

        joint_pos_sim_list: List[npt.NDArray[np.float32]] = []
        for action, kp in zip(action_list, kp_list):
            sim.set_motor_kps(dict(zip(motor_names, [kp] * len(motor_names))))

            for a in action:
                obs = sim.get_observation()
                sim.set_motor_target(a)
                sim.step()

                assert obs.joint_pos is not None
                joint_pos_sim_list.append(obs.joint_pos[joint_idx])

        joint_pos_sim = np.array(joint_pos_sim_list)

        # RMSE
        error = np.sqrt(np.mean((joint_pos_real - joint_pos_sim) ** 2))

        # FFT (Fourier Transform) of the joint position data and reference data
        joint_pos_sim_fft = np.fft.fft(joint_pos_sim)
        joint_pos_real_fft = np.fft.fft(joint_pos_real)

        joint_pos_sim_fft_freq = np.fft.fftfreq(len(joint_pos_sim_fft), d=sim.dt)
        joint_pos_real_fft_freq = np.fft.fftfreq(len(joint_pos_real_fft), d=sim.dt)

        magnitude_sim = np.abs(joint_pos_sim_fft[: len(joint_pos_sim_fft) // 2])
        magnitude_real = np.abs(joint_pos_real_fft[: len(joint_pos_real_fft) // 2])

        magnitude_sim_filtered = magnitude_sim[
            joint_pos_sim_fft_freq[: len(joint_pos_sim_fft) // 2] < freq_max
        ]
        magnitude_real_filtered = magnitude_real[
            joint_pos_real_fft_freq[: len(joint_pos_real_fft) // 2] < freq_max
        ]
        error_fft = np.sqrt(
            np.mean((magnitude_real_filtered - magnitude_sim_filtered) ** 2)
        )

        return error + error_fft * 0.01

    sampler: optuna.samplers.BaseSampler | None = None
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

    initial_trial = dict(
        damping=float(sim.model.joint(joint_name).damping),
        armature=float(sim.model.joint(joint_name).armature),
        frictionloss=float(sim.model.joint(joint_name).frictionloss),
    )
    if "sysID" in robot.name:
        assert isinstance(sim.controller, MotorController)
        initial_trial.update(
            dict(
                tau_max=float(sim.controller.tau_max),
                q_dot_tau_max=float(sim.controller.q_dot_tau_max),
                q_dot_max=float(sim.controller.q_dot_max),
            )
        )

    study.enqueue_trial(initial_trial)

    study.optimize(
        objective,
        n_trials=n_iters,
        n_jobs=1,
        show_progress_bar=True,
        callbacks=[partial(early_stop_check, early_stopping_rounds=early_stop_rounds)],
    )

    log(
        f"Best parameters: {study.best_params}; best value: {study.best_value}",
        header="SysID",
        level="info",
    )

    sim.close()

    return study.best_params, study.best_value


def optimize_all(
    robot: Robot,
    sim_name: str,
    obs_pos_dict: Dict[str, List[npt.NDArray[np.float32]]],
    action_dict: Dict[str, List[npt.NDArray[np.float32]]],
    kp_dict: Dict[str, List[float]],
    n_iters: int,
    early_stop_rounds: int,
):
    """Optimizes parameters for each joint of a robot using provided observation and action data.

    Args:
        robot (Robot): The robot instance for which parameters are being optimized.
        sim_name (str): The name of the simulation.
        obs_pos_dict (Dict[str, List[npt.NDArray[np.float32]]]): A dictionary mapping joint names to lists of observed positions.
        action_dict (Dict[str, List[npt.NDArray[np.float32]]]): A dictionary mapping joint names to lists of actions taken.
        kp_dict (Dict[str, List[float]]): A dictionary mapping joint names to lists of proportional gain values.
        n_iters (int): The number of iterations for the optimization process.
        early_stop_rounds (int): The number of rounds for early stopping criteria.

    Returns:
        Tuple[Dict[str, Dict[str, float]], Dict[str, float]]: A tuple containing two dictionaries. The first dictionary maps joint names to their optimized parameters, and the second dictionary maps joint names to their optimized values.
    """

    # return sysID_file_path
    optimize_args: List[
        Tuple[
            Robot,
            str,
            str,
            List[npt.NDArray[np.float32]],
            List[npt.NDArray[np.float32]],
            List[float],
            int,
            int,
        ]
    ] = [
        (
            robot,
            sim_name,
            joint_name,
            obs_pos_dict[joint_name],
            action_dict[joint_name],
            kp_dict[joint_name],
            n_iters,
            early_stop_rounds,
        )
        for joint_name in obs_pos_dict
    ]

    # # Create a pool of processes
    # with Pool(processes=len(obs_pos_dict)) as pool:
    #     results = pool.starmap(optimize_parameters, optimize_args)

    # # Process results
    # for joint_name, result in zip(obs_pos_dict.keys(), results):
    #     opt_params, opt_values = result
    #     if len(opt_params) > 0:
    #         opt_params_dict[joint_name] = opt_params
    #         opt_values_dict[joint_name] = opt_values

    opt_params_dict: Dict[str, Dict[str, float]] = {}
    opt_values_dict: Dict[str, float] = {}
    for args in optimize_args:
        opt_params, opt_values = optimize_parameters(*args)
        opt_params_dict[args[2]] = opt_params
        opt_values_dict[args[2]] = opt_values

    return opt_params_dict, opt_values_dict


def evaluate(
    robot: Robot,
    sim_name: str,
    obs_pos_dict: Dict[str, List[npt.NDArray[np.float32]]],
    action_dict: Dict[str, List[npt.NDArray[np.float32]]],
    kp_dict: Dict[str, List[float]],
    opt_params_dict: Dict[str, Dict[str, float]],
    opt_values_dict: Dict[str, float],
    exp_folder_path: str,
):
    """Evaluates the performance of a robot simulation by comparing simulated and real joint positions, and logs the results.

    Args:
        robot (Robot): The robot object containing joint and motor configurations.
        sim_name (str): The name of the simulator to use, e.g., "mujoco".
        obs_pos_dict (Dict[str, List[npt.NDArray[np.float32]]]): Dictionary mapping joint names to lists of observed position arrays.
        action_dict (Dict[str, List[npt.NDArray[np.float32]]]): Dictionary mapping joint names to lists of action arrays.
        kp_dict (Dict[str, List[float]]): Dictionary mapping joint names to lists of proportional gain values.
        opt_params_dict (Dict[str, Dict[str, float]]): Dictionary of optimized parameters for each joint.
        opt_values_dict (Dict[str, float]): Dictionary of optimized values for each joint.
        exp_folder_path (str): Path to the folder where experiment results will be saved.

    Raises:
        ValueError: If an invalid simulator name is provided.
    """

    opt_params_file_path = os.path.join(exp_folder_path, "opt_params.json")
    opt_values_file_path = os.path.join(exp_folder_path, "opt_values.json")

    with open(opt_params_file_path, "w") as f:
        json.dump(opt_params_dict, f, indent=4)

    with open(opt_values_file_path, "w") as f:
        json.dump(opt_values_dict, f, indent=4)

    dyn_config_path = os.path.join(
        "toddlerbot", "descriptions", robot.name, "config_dynamics.json"
    )
    if os.path.exists(dyn_config_path):
        dyn_config = json.load(open(dyn_config_path, "r"))
        for joint_name in opt_params_dict:
            for param_name in opt_params_dict[joint_name]:
                dyn_config[joint_name][param_name] = opt_params_dict[joint_name][
                    param_name
                ]
    else:
        dyn_config = opt_params_dict

    with open(dyn_config_path, "w") as f:
        json.dump(dyn_config, f, indent=4)

    time_seq_ref_dict: Dict[str, List[float]] = {}
    time_seq_sim_dict: Dict[str, List[float]] = {}
    time_seq_real_dict: Dict[str, List[float]] = {}
    joint_pos_sim_dict: Dict[str, List[float]] = {}
    joint_pos_real_dict: Dict[str, List[float]] = {}
    action_sim_dict: Dict[str, List[float]] = {}
    action_real_dict: Dict[str, List[float]] = {}

    for joint_name in obs_pos_dict:
        obs_list = obs_pos_dict[joint_name]
        action_list = action_dict[joint_name]
        kp_list = kp_dict[joint_name]

        motor_names = robot.joint_to_motor_name[joint_name]
        joint_idx = robot.joint_ordering.index(joint_name)
        joint_pos_real = np.concatenate([obs[:, joint_idx] for obs in obs_list])

        if sim_name == "mujoco":
            sim = MuJoCoSim(robot, fixed_base=True)
        else:
            raise ValueError("Invalid simulator")

        joint_dyn = {
            joint_name: {
                "damping": opt_params_dict[joint_name]["damping"],
                "armature": opt_params_dict[joint_name]["armature"],
                "frictionloss": opt_params_dict[joint_name]["frictionloss"],
            }
        }
        sim.set_joint_dynamics(joint_dyn)

        if "sysID" in robot.name:
            sim.set_motor_dynamics(
                dict(
                    tau_max=opt_params_dict[joint_name]["tau_max"],
                    q_dot_tau_max=opt_params_dict[joint_name]["q_dot_tau_max"],
                    q_dot_max=opt_params_dict[joint_name]["q_dot_max"],
                )
            )

        joint_pos_sim_list: List[npt.NDArray[np.float32]] = []
        for action, kp in zip(action_list, kp_list):
            sim.set_motor_kps(dict(zip(motor_names, [kp] * len(motor_names))))
            for a in action:
                obs = sim.get_observation()
                sim.set_motor_target(a)
                sim.step()

                assert obs.joint_pos is not None
                joint_pos_sim_list.append(obs.joint_pos[joint_idx])

        joint_pos_sim = np.array(joint_pos_sim_list)

        error = np.sqrt(np.mean((joint_pos_real - joint_pos_sim) ** 2))

        log(
            f"{joint_name} root mean squared error: {error}",
            header="SysID",
            level="info",
        )

        time_seq_ref_dict[joint_name] = list(
            np.arange(sum([len(action) for action in action_list]))
            * (sim.n_frames * sim.dt)
        )
        time_seq_sim_dict[joint_name] = time_seq_ref_dict[joint_name]
        time_seq_real_dict[joint_name] = time_seq_ref_dict[joint_name]

        joint_pos_sim_dict[joint_name] = joint_pos_sim.tolist()
        joint_pos_real_dict[joint_name] = joint_pos_real.tolist()

        action_all = np.concatenate(
            [action[:, joint_idx] for action in action_list]
        ).tolist()
        action_sim_dict[joint_name] = action_all
        action_real_dict[joint_name] = action_all

        sim.close()

    plot_joint_tracking(
        time_seq_sim_dict,
        time_seq_real_dict,
        joint_pos_sim_dict,
        joint_pos_real_dict,
        robot.joint_limits,
        save_path=exp_folder_path,
        file_name="sim2real_joint_pos",
        line_suffix=["_sim", "_real"],
    )
    plot_joint_tracking_frequency(
        time_seq_sim_dict,
        time_seq_real_dict,
        joint_pos_sim_dict,
        joint_pos_real_dict,
        save_path=exp_folder_path,
        file_name="sim2real_joint_freq",
        line_suffix=["_sim", "_real"],
    )
    plot_joint_tracking(
        time_seq_sim_dict,
        time_seq_ref_dict,
        joint_pos_sim_dict,
        action_sim_dict,
        robot.joint_limits,
        save_path=exp_folder_path,
        file_name="sim_tracking",
    )
    plot_joint_tracking_frequency(
        time_seq_sim_dict,
        time_seq_ref_dict,
        joint_pos_sim_dict,
        action_sim_dict,
        save_path=exp_folder_path,
        file_name="sim_tracking_freq",
    )
    plot_joint_tracking(
        time_seq_real_dict,
        time_seq_ref_dict,
        joint_pos_real_dict,
        action_real_dict,
        robot.joint_limits,
        save_path=exp_folder_path,
        file_name="real_tracking",
    )
    plot_joint_tracking_frequency(
        time_seq_real_dict,
        time_seq_ref_dict,
        joint_pos_real_dict,
        action_real_dict,
        save_path=exp_folder_path,
        file_name="real_tracking_freq",
    )


def main():
    """Executes the SysID optimization process for a specified robot and simulator.

    This function parses command-line arguments to configure the optimization process,
    validates the experiment folder path, and initializes the robot and experiment settings.
    It then loads datasets, optimizes hyperparameters, and evaluates the optimized parameters
    in the simulation.

    Raises:
        ValueError: If the specified experiment folder path does not exist.
    """

    parser = argparse.ArgumentParser(description="Run the SysID optimization.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in descriptions.",
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
        default="sysID_fixed",
        help="The name of the task.",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=500,
        help="The number of iterations to optimize the parameters.",
    )
    parser.add_argument(
        "--early-stop",
        type=int,
        default=200,
        help="The number of iterations to early stop the optimization.",
    )
    parser.add_argument(
        "--time-str",
        type=str,
        default="",
        required=True,
        help="The name of the run.",
    )
    args = parser.parse_args()

    data_path = os.path.join(
        "results", f"{args.robot}_{args.policy}_real_world_{args.time_str}"
    )
    if not os.path.exists(data_path):
        raise ValueError("Invalid experiment folder path")

    robot = Robot(args.robot)

    exp_name = f"{robot.name}_sysID_{args.sim}_optim"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    exp_folder_path = f"results/{exp_name}_{time_str}"

    os.makedirs(exp_folder_path, exist_ok=True)

    with open(os.path.join(exp_folder_path, "opt_config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    obs_pos_dict, action_dict, kp_dict = load_datasets(robot, data_path)

    ###### Optimize the hyperparameters ######
    # optimize_parameters(
    #     robot,
    #     args.sim,
    #     "waist_yaw",
    #     obs_pos_dict["waist_yaw"],
    #     action_dict["waist_yaw"],
    #     args.n_iters,
    # )

    opt_params_dict, opt_values_dict = optimize_all(
        robot,
        args.sim,
        obs_pos_dict,
        action_dict,
        kp_dict,
        args.n_iters,
        args.early_stop,
    )

    ##### Evaluate the optimized parameters in the simulation ######
    evaluate(
        robot,
        args.sim,
        obs_pos_dict,
        action_dict,
        kp_dict,
        opt_params_dict,
        opt_values_dict,
        exp_folder_path,
    )


if __name__ == "__main__":
    main()
