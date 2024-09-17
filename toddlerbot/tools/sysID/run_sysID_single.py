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
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.misc_utils import log
from toddlerbot.visualization.vis_plot import (
    plot_joint_tracking,
    plot_joint_tracking_frequency,
)

logger = _get_library_root_logger()


class Actuator:
    def __init__(
        self,
        kP: float,
        kD: float,
        mu_s: float,
        mu_d: float,
        tau_max: float,
        q_dot_tau_max: float,
        q_dot_max: float,
        I_m: float,
        b_min: float = 0.02,
        b_max: float = 0.05,
        sigma_q_0: float = 0.0,
        sigma_q_1: float = 0.0,
        q_dot_s: float = 0.01,
        tau_b: float = 1.0,
        epsilon_q_max: float = 0.02,
        fd_vel: bool = False,
    ) -> None:
        self.kP = kP
        self.kD = kD
        self.mu_s = mu_s
        self.mu_d = mu_d
        self.tau_max = tau_max
        self.q_dot_tau_max = q_dot_tau_max
        self.q_dot_max = q_dot_max
        self.b_min = b_min
        self.b_max = b_max
        self.sigma_q_0 = sigma_q_0
        self.sigma_q_1 = sigma_q_1
        self.I_m = I_m

        self.tau_b = tau_b
        self.q_dot_s = q_dot_s
        self.epsilon_q_max = epsilon_q_max
        self.fd_vel = fd_vel
        # Randomized parameters per episode
        self.reset_episode()

    def reset_episode(self) -> None:
        """
        Reset parameters that are randomized at the start of each episode.
        """
        self.epsilon_q = np.random.uniform(-self.epsilon_q_max, self.epsilon_q_max)
        self.b = np.random.uniform(self.b_min, self.b_max)
        # Randomize I_m up to ±20% offset
        self.I_m_randomized = self.I_m * np.random.uniform(0.8, 1.2)
        # Initialize previous error for head actuators
        self.error_prev = 0.0

    def compute_tau_m(self, a: float, q: float, q_dot: float, dt: float) -> float:
        """
        Compute the motor torque τ_m.

        Parameters:
        - a: Joint setpoint.
        - q: Joint position.
        - q_dot: Joint velocity.
        - dt: Time step for numerical differentiation.

        Returns:
        - tau_m: Motor torque.
        """
        q_tilde = q + self.epsilon_q  # Equation (15)
        error = a - q_tilde
        tau_m = self.kP * error - self.kD * q_dot

        return tau_m

    def compute_tau_f(self, q_dot: float) -> float:
        """
        Compute the friction torque τ_f.

        Parameters:
        - q_dot: Joint velocity.

        Returns:
        - tau_f: Friction torque.
        """
        # Equation (17)
        tau_f = self.mu_s * np.tanh(q_dot / self.q_dot_s) + self.mu_d * q_dot
        return tau_f

    def compute_tau_limits(self, q_dot: float) -> Tuple[float, float]:
        """
        Compute the velocity-dependent torque limits τ_min and τ_max.

        Parameters:
        - q_dot: Joint velocity.

        Returns:
        - tau_min: Minimum torque limit.
        - tau_max: Maximum torque limit.
        """
        abs_q_dot = abs(q_dot)

        if abs_q_dot <= self.q_dot_tau_max:
            tau_limit = self.tau_max
        elif abs_q_dot <= self.q_dot_max:
            # Linear decrease of torque limit
            slope = self.tau_max / (self.q_dot_tau_max - self.q_dot_max)
            tau_limit = slope * (abs_q_dot - self.q_dot_tau_max) + self.tau_max
        else:
            tau_limit = 0.0

        tau_max = tau_limit
        tau_min = -tau_limit
        return tau_min, tau_max

    def compute_total_tau(self, tau_m: float, q_dot: float) -> float:
        """
        Compute the total torque τ applied at the joint.

        Parameters:
        - tau_m: Motor torque.
        - q_dot: Joint velocity.

        Returns:
        - tau: Total joint torque.
        """
        tau_f = self.compute_tau_f(q_dot)
        tau_min, tau_max = self.compute_tau_limits(q_dot)
        # Equation (18)
        tau_m_clamped = np.clip(tau_m, tau_min, tau_max)
        tau = tau_m_clamped - tau_f
        return tau

    def compute_q_hat(self, q: float, tau_m: float, q_dot: float) -> float:
        """
        Compute the measured joint position q̂.

        Parameters:
        - q: Joint position.
        - tau_m: Motor torque.
        - q_dot: Joint velocity.

        Returns:
        - q_hat: Measured joint position.
        """
        q_tilde = q + self.epsilon_q  # Equation (15)
        # Backlash term from Equation (19)
        backlash_term = 0.5 * self.b * np.tanh(tau_m / self.tau_b)
        # Noise model from Equation (20)
        sigma_q = self.sigma_q_0 + self.sigma_q_1 * abs(q_dot)
        noise = np.random.normal(0, sigma_q)
        # Equation (19)
        q_hat = q_tilde + backlash_term + noise
        return q_hat

    def step(self, a: float, q: float, q_dot: float, dt: float) -> Tuple[float, float]:
        """
        Simulate one time step of the actuator.

        Parameters:
        - a: Joint setpoint.
        - q: Current joint position.
        - q_dot: Current joint velocity.
        - dt: Time step duration.

        Returns:
        - tau: Total joint torque applied.
        - q_hat: Measured joint position.
        """
        tau_m = self.compute_tau_m(a, q, q_dot, dt)
        tau = self.compute_total_tau(tau_m, q_dot)
        q_hat = self.compute_q_hat(q, tau_m, q_dot)
        return tau, q_hat


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
    obs_vel_dict: Dict[str, List[npt.NDArray[np.float32]]] = {}
    obs_tor_dict: Dict[str, List[npt.NDArray[np.float32]]] = {}
    action_dict: Dict[str, List[npt.NDArray[np.float32]]] = {}
    kp_dict: Dict[str, List[float]] = {}

    def set_obs_and_action(
        joint_name: str, motor_kps: Dict[str, float], idx_range: slice
    ):
        kp = motor_kps.get(joint_name, 0)

        obs_time = np.array([obs.time for obs in obs_list[idx_range]])
        obs_pos = np.array([obs.motor_pos for obs in obs_list[idx_range]])
        obs_vel = np.array([obs.motor_vel for obs in obs_list[idx_range]])
        obs_tor = np.array([obs.motor_tor for obs in obs_list[idx_range]])

        action = np.array(
            [
                list(motor_angles.values())
                for motor_angles in motor_angles_list[idx_range]
            ]
        )

        if joint_name not in obs_time_dict:
            obs_time_dict[joint_name] = []
            obs_pos_dict[joint_name] = []
            obs_vel_dict[joint_name] = []
            obs_tor_dict[joint_name] = []
            action_dict[joint_name] = []
            kp_dict[joint_name] = []

        obs_time_dict[joint_name].append(obs_time)
        obs_pos_dict[joint_name].append(obs_pos)
        obs_vel_dict[joint_name].append(obs_vel)
        obs_tor_dict[joint_name].append(obs_tor)
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

        last_idx = 0
        for joint_names, motor_kps, obs_idx in zip(
            joint_names_list, motor_kps_list, obs_indices
        ):
            for joint_name in joint_names:
                # if "ank_roll" in joint_name:
                #     break
                set_obs_and_action(joint_name, motor_kps, slice(last_idx, obs_idx))

            last_idx = obs_idx
    else:
        start_idx = 500
        for joint_name in reversed(robot.joint_ordering):
            joints_config = robot.config["joints"]
            if joints_config[joint_name]["group"] == "leg":
                motor_names = robot.joint_to_motor_name[joint_name]
                motor_kps = {joint_name: joints_config[motor_names[0]]["kp_real"]}
                set_obs_and_action(joint_name, motor_kps, slice(start_idx, None))

    return obs_time_dict, obs_pos_dict, obs_vel_dict, obs_tor_dict, action_dict, kp_dict


def optimize_parameters(
    robot: Robot,
    sim_name: str,
    motor_name: str,
    obs_pos_list: List[npt.NDArray[np.float32]],
    obs_vel_list: List[npt.NDArray[np.float32]],
    obs_tor_list: List[npt.NDArray[np.float32]],
    action_list: List[npt.NDArray[np.float32]],
    kp_list: List[float],
    n_iters: int = 1000,
    early_stopping_rounds: int = 200,
    freq_max: float = 10,
    sampler_name: str = "CMA",
    # gain_range: Tuple[float, float, float] = (0, 50, 0.1),
    damping_range: Tuple[float, float, float] = (0.001, 10, 1e-3),
    armature_range: Tuple[float, float, float] = (0.0, 0.1, 1e-3),
    frictionloss_range: Tuple[float, float, float] = (0.001, 1.0, 1e-3),
):
    # if sim_name == "mujoco":
    #     sim = MuJoCoSim(robot, fixed_base=True)

    # else:
    #     raise ValueError("Invalid simulator")

    # initial_trial = {
    #     "damping": float(sim.model.joint(joint_name).damping),
    #     "armature": float(sim.model.joint(joint_name).armature),
    #     "frictionloss": float(sim.model.joint(joint_name).frictionloss),
    # }

    motor_idx = robot.motor_ordering.index(motor_name)
    motor_tor = np.concatenate(obs_tor_list)[:, motor_idx]

    def early_stopping_check(
        study: optuna.Study, trial: optuna.Trial, early_stopping_rounds: int
    ):
        current_trial_number = trial.number
        best_trial_number = study.best_trial.number
        should_stop = (
            current_trial_number - best_trial_number
        ) >= early_stopping_rounds
        if should_stop:
            logger.debug(f"early stopping detected: {should_stop}")
            study.stop()

    def objective(trial: optuna.Trial):
        joint_pos_sim_list: List[npt.NDArray[np.float32]] = []
        mu_s = trial.suggest_float("mu_s", 0.0, 1.0, step=0.01)
        mu_d = trial.suggest_float("mu_d", 0.0, 1.0, step=0.01)
        # tau_max = trial.suggest_float("tau_max", 0.0, 10.0, step=0.01)
        # q_dot_tau_max = trial.suggest_float("q_dot_tau_max", 0.0, 1.0, step=0.01)
        # q_dot_max = trial.suggest_float("q_dot_max", 0.0, 10.0, step=0.01)
        tau_max = 0.64
        q_dot_tau_max = 0.45
        q_dot_max = 7.0
        I_m = trial.suggest_float("I_m", 0.0, 1.0, step=1e-3)
        actuator = Actuator(
            robot.config["joints"][motor_name]["kp_sim"],
            robot.config["joints"][motor_name]["kd_sim"],
            mu_s,
            mu_d,
            tau_max,
            q_dot_tau_max,
            q_dot_max,
            I_m,
            fd_vel=False,
        )
        for kp, action, obs_pos, obs_vel in zip(
            kp_list, action_list, obs_pos_list, obs_vel_list
        ):
            actuator.kP = kp

            motor_tor_list: List[float] = []
            for a, q, q_dot in zip(action, obs_pos, obs_vel):
                tau, _ = actuator.step(a, q, q_dot, 0.001)
                motor_tor_list.append(tau)
            joint_pos_sim_list.append(
                np.array(
                    [joint_state[motor_name].pos for joint_state in joint_state_list]
                )
            )

        joint_pos_sim = np.concatenate(joint_pos_sim_list)

        # RMSE
        error = np.sqrt(np.mean((motor_pos - joint_pos_sim) ** 2))

        # FFT (Fourier Transform) of the joint position data and reference data
        joint_pos_sim_fft = np.fft.fft(joint_pos_sim)
        joint_pos_real_fft = np.fft.fft(motor_pos)

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
        study_name=f"{robot.name}_{motor_name}_{time_str}",
        storage=storage,
        sampler=sampler,
        load_if_exists=True,
    )

    study.enqueue_trial(initial_trial)
    study.optimize(
        objective,
        n_trials=n_iters,
        n_jobs=1,
        show_progress_bar=True,
        callbacks=[
            partial(early_stopping_check, early_stopping_rounds=early_stopping_rounds)
        ],
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
    obs_vel_dict: Dict[str, List[npt.NDArray[np.float32]]],
    obs_tor_dict: Dict[str, List[npt.NDArray[np.float32]]],
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
            obs_vel_dict[joint_name],
            obs_tor_dict[joint_name],
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
        motor_names = robot.joint_to_motor_name[joint_name]

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

        obs_time_sim_list: List[float] = []
        joint_pos_sim_list: List[npt.NDArray[np.float32]] = []
        for action, kp in zip(action_list, kp_list):
            sim.set_motor_kps(dict(zip(motor_names, [kp] * len(motor_names))))

            joint_state_list = sim.rollout(action)
            obs_time_sim_list.extend(
                [joint_state[joint_name].time for joint_state in joint_state_list]
            )
            joint_pos_sim_list.append(
                np.array(
                    [joint_state[joint_name].pos for joint_state in joint_state_list]
                )
            )

        joint_pos_sim = np.concatenate(joint_pos_sim_list)

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
        time_seq_sim_dict[joint_name] = obs_time_sim_list
        obs_time_real = np.concatenate(obs_time_dict[joint_name])
        obs_time_real -= obs_time_real[0]
        time_seq_real_dict[joint_name] = obs_time_real.tolist()

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

    obs_time_dict, obs_pos_dict, obs_vel_dict, obs_tor_dict, action_dict, kp_dict = (
        load_datasets(robot, data_path)
    )

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
        obs_vel_dict,
        obs_tor_dict,
        action_dict,
        kp_dict,
        args.n_iters,
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
