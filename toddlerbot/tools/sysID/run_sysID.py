import argparse
import json
import os
import pickle
import time
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import optuna

from toddlerbot.actuation import JointState
from toddlerbot.sim import Obs
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.misc_utils import log
from toddlerbot.visualization.vis_plot import plot_joint_tracking


def load_datasets(robot: Robot, data_path: str):
    # Use glob to find all pickle files matching the pattern
    pickle_file_path = os.path.join(data_path, "log_data.pkl")
    if not os.path.exists(pickle_file_path):
        raise ValueError("No data files found")

    with open(pickle_file_path, "rb") as f:
        data_dict = pickle.load(f)

    obs_list: List[Obs] = data_dict["obs_list"]
    motor_angles_list: List[Dict[str, float]] = data_dict["motor_angles_list"]

    obs_time_dict: Dict[str, List[npt.NDArray[np.float32]]] = {}
    obs_pos_dict: Dict[str, List[npt.NDArray[np.float32]]] = {}
    action_dict: Dict[str, List[npt.NDArray[np.float32]]] = {}
    kp_dict: Dict[str, List[float]] = {}

    def set_obs_and_action(
        joint_name: str, motor_kps: Dict[str, float], idx_range: slice
    ):
        kp = motor_kps.get(joint_name, 0)

        obs_time = np.array([obs.time for obs in obs_list[idx_range]])
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

        if joint_name not in obs_time_dict:
            obs_time_dict[joint_name] = []
            obs_pos_dict[joint_name] = []
            action_dict[joint_name] = []
            kp_dict[joint_name] = []

        obs_time_dict[joint_name].append(obs_time)
        obs_pos_dict[joint_name].append(obs_pos)
        action_dict[joint_name].append(action)
        kp_dict[joint_name].append(kp)

    if "ckpt_dict" in data_dict:
        ckpt_dict: Dict[str, Dict[str, float]] = data_dict["ckpt_dict"]
        ckpt_times = list(ckpt_dict.keys())
        motor_kps_list: List[Dict[str, float]] = []
        symm_joint_names: List[str] = []
        for d in list(ckpt_dict.values()):
            motor_kps_list.append(d)
            symm_joint_names.append(list(d.keys())[0])

        obs_time = [obs.time for obs in obs_list]
        obs_indices = np.searchsorted(obs_time, ckpt_times)  # type: ignore

        last_idx = 0
        for symm_joint_name, motor_kps, obs_idx in zip(
            symm_joint_names, motor_kps_list, obs_indices
        ):
            if symm_joint_name not in ["waist_yaw"]:
                last_idx = obs_idx
                continue

            if symm_joint_name in robot.joint_ordering:
                set_obs_and_action(symm_joint_name, motor_kps, slice(last_idx, obs_idx))
            else:
                set_obs_and_action(
                    f"left_{symm_joint_name}", motor_kps, slice(last_idx, obs_idx)
                )
                set_obs_and_action(
                    f"right_{symm_joint_name}", motor_kps, slice(last_idx, obs_idx)
                )

            last_idx = obs_idx
    else:
        set_obs_and_action("all", {}, slice(None))

    return obs_time_dict, obs_pos_dict, action_dict, kp_dict


def optimize_parameters(
    robot: Robot,
    sim_name: str,
    joint_name: str,
    obs_list: List[npt.NDArray[np.float32]],
    action_list: List[npt.NDArray[np.float32]],
    kp_list: List[float],
    n_iters: int = 1000,
    sampler_name: str = "CMA",
    # gain_range: Tuple[float, float, float] = (0, 50, 0.1),
    damping_range: Tuple[float, float, float] = (0, 10, 1e-3),
    armature_range: Tuple[float, float, float] = (0, 0.1, 1e-3),
    # friction_range: Tuple[float, float, float] = (0, 1.0, 1e-3),
):
    if sim_name == "mujoco":
        sim = MuJoCoSim(robot, fixed_base=True)

    else:
        raise ValueError("Invalid simulator")

    initial_trial = {
        "damping": float(sim.model.joint(joint_name).damping),  # type: ignore
        "armature": float(sim.model.joint(joint_name).armature),  # type: ignore
    }
    joint_idx = robot.joint_ordering.index(joint_name)
    motor_name = robot.joint_to_motor_name[joint_name]

    joint_pos_real = np.concatenate([obs[:, joint_idx] for obs in obs_list])  # type: ignore

    def objective(trial: optuna.Trial):
        # gain = trial.suggest_float("gain", *gain_range[:2], step=gain_range[2])
        damping = trial.suggest_float(
            "damping", *damping_range[:2], step=damping_range[2]
        )
        armature = trial.suggest_float(
            "armature", *armature_range[:2], step=armature_range[2]
        )
        # frictionloss = trial.suggest_float(
        #     "frictionloss", *friction_range[:2], step=friction_range[2]
        # )
        joint_dyn = {
            joint_name: {
                # "gain": gain,
                "damping": damping,
                "armature": armature,
                # "frictionloss": frictionloss,
            }
        }

        sim.set_joint_dynamics(joint_dyn)

        joint_pos_sim_list: List[npt.NDArray[np.float32]] = []
        for action, kp in zip(action_list, kp_list):
            if isinstance(motor_name, list):
                sim.set_motor_kps(dict(zip(motor_name, [kp] * len(motor_name))))
            else:
                sim.set_motor_kps({motor_name: kp})

            joint_state_list = sim.rollout(action)
            joint_pos_sim_list.append(
                np.array(
                    [joint_state[joint_name].pos for joint_state in joint_state_list]
                )
            )

        joint_pos_sim = np.concatenate(joint_pos_sim_list)  # type: ignore

        error = np.sqrt(np.mean((joint_pos_real - joint_pos_sim) ** 2))

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

    study.enqueue_trial(initial_trial)
    study.optimize(objective, n_trials=n_iters, n_jobs=1, show_progress_bar=True)

    log(
        f"Best parameters: {study.best_params}; best value: {study.best_value}",
        header="SysID",
        level="info",
    )

    sim.close()

    return study.best_params, study.best_value


def multiprocessing_optimization(
    robot: Robot,
    sim_name: str,
    obs_pos_dict: Dict[str, List[npt.NDArray[np.float32]]],
    action_dict: Dict[str, List[npt.NDArray[np.float32]]],
    kp_dict: Dict[str, List[float]],
    n_iters: int,
):
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
    obs_time_dict: Dict[str, List[npt.NDArray[np.float32]]],
    obs_pos_dict: Dict[str, List[npt.NDArray[np.float32]]],
    action_dict: Dict[str, List[npt.NDArray[np.float32]]],
    kp_dict: Dict[str, List[float]],
    opt_params_dict: Dict[str, Dict[str, float]],
    opt_values_dict: Dict[str, float],
    exp_folder_path: str,
):
    opt_params_file_path = os.path.join(exp_folder_path, "opt_params.json")
    opt_values_file_path = os.path.join(exp_folder_path, "opt_values.json")

    with open(opt_params_file_path, "w") as f:
        json.dump(opt_params_dict, f, indent=4)

    with open(opt_values_file_path, "w") as f:
        json.dump(opt_values_dict, f, indent=4)

    dyn_config_path = os.path.join(
        "toddlerbot", "robot_descriptions", robot.name, "config_dynamics.json"
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

        joint_idx = robot.joint_ordering.index(joint_name)
        motor_name = robot.joint_to_motor_name[joint_name]

        joint_pos_real = np.concatenate([obs[:, joint_idx] for obs in obs_list])  # type: ignore

        if sim_name == "mujoco":
            sim = MuJoCoSim(robot, fixed_base=True)
        else:
            raise ValueError("Invalid simulator")

        joint_dyn = {
            joint_name: {
                "damping": opt_params_dict[joint_name]["damping"],
                "armature": opt_params_dict[joint_name]["armature"],
            }
        }
        sim.set_joint_dynamics(joint_dyn)

        obs_time_sim_list: List[float] = []
        joint_pos_sim_list: List[npt.NDArray[np.float32]] = []
        for action, kp in zip(action_list, kp_list):
            if isinstance(motor_name, list):
                sim.set_motor_kps(dict(zip(motor_name, [kp] * len(motor_name))))
            else:
                sim.set_motor_kps({motor_name: kp})

            joint_state_list = sim.rollout(action)
            obs_time_sim_list.extend(
                [joint_state[joint_name].time for joint_state in joint_state_list]
            )
            joint_pos_sim_list.append(
                np.array(
                    [joint_state[joint_name].pos for joint_state in joint_state_list]
                )
            )

        joint_pos_sim = np.concatenate(joint_pos_sim_list)  # type: ignore

        error = np.sqrt(np.mean((joint_pos_real - joint_pos_sim) ** 2))

        log(
            f"{joint_name} root mean squared error: {error}",
            header="SysID",
            level="info",
        )

        time_seq_ref_dict[joint_name] = np.arange(  # type: ignore
            sum([len(action) for action in action_list])
        ) * (sim.n_frames * sim.dt)
        time_seq_sim_dict[joint_name] = obs_time_sim_list
        obs_time_real = np.concatenate(obs_time_dict[joint_name])  # type: ignore
        obs_time_real -= obs_time_real[0]
        time_seq_real_dict[joint_name] = obs_time_real.tolist()

        joint_pos_sim_dict[joint_name] = joint_pos_sim.tolist()
        joint_pos_real_dict[joint_name] = joint_pos_real.tolist()

        action_all = np.concatenate(  # type: ignore
            [action[:, joint_idx] for action in action_list]
        ).tolist()
        action_sim_dict[joint_name] = action_all
        action_real_dict[joint_name] = action_all

        sim.close()

    plot_joint_tracking(
        time_seq_sim_dict,
        time_seq_ref_dict,
        joint_pos_sim_dict,
        action_sim_dict,
        robot.joint_limits,
        save_path=exp_folder_path,
        file_name="sim_tracking",
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
        default=500,
        help="The number of iterations to optimize the parameters.",
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
        "results", f"{args.robot}_sysID_fixed_real_world_{args.time_str}"
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

    obs_time_dict, obs_pos_dict, action_dict, kp_dict = load_datasets(robot, data_path)

    ###### Optimize the hyperparameters ######
    # optimize_parameters(
    #     robot,
    #     args.sim,
    #     "waist_yaw",
    #     obs_pos_dict["waist_yaw"],
    #     action_dict["waist_yaw"],
    #     args.n_iters,
    # )

    opt_params_dict, opt_values_dict = multiprocessing_optimization(
        robot, args.sim, obs_pos_dict, action_dict, kp_dict, args.n_iters
    )

    ##### Evaluate the optimized parameters in the simulation ######
    evaluate(
        robot,
        args.sim,
        obs_time_dict,
        obs_pos_dict,
        action_dict,
        kp_dict,
        opt_params_dict,
        opt_values_dict,
        exp_folder_path,
    )


if __name__ == "__main__":
    main()
