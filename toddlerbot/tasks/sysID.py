import copy
import json
import time
from xml.etree.ElementTree import tostring

import numpy as np
import optuna

from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.real_world import RealWorld
from toddlerbot.utils.constants import MUJOCO_TIMESTEP
from toddlerbot.utils.math_utils import round_floats
from toddlerbot.utils.misc_utils import log, precise_sleep
from toddlerbot.utils.vis_plot import plot_joint_tracking


def generate_random_sinusoidal_config(
    duration, control_dt, mean, frequency_range, amplitude_range
):
    frequency = np.random.uniform(*frequency_range)
    amplitude = np.random.uniform(*amplitude_range)

    return {
        "frequency": frequency,
        "amplitude": amplitude,
        "duration": duration,
        "control_dt": control_dt,
        "mean": mean,
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
    signal = signal_config["mean"] + signal_config["amplitude"] * np.sin(
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
    initial_joint_angles[joint_name] = signal_pos[0]

    if joint_name == "left_ank_roll":
        initial_joint_angles["left_ank_pitch"] = np.pi / 6

    if joint_name == "left_hip_pitch" or joint_name == "left_knee":
        initial_joint_angles["left_hip_yaw"] = -np.pi / 4

    is_ankle = joint_name in robot.mighty_zap_joint2id

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

            if is_ankle:
                ankle_id = robot.mighty_zap_joint2id[joint_name]
                for side, ids in robot.ankle2mighty_zap.items():
                    if ankle_id not in ids:
                        continue

                    ankle_side = side
                    ankle_pos_list = [
                        joint_state_dict[robot.mighty_zap_id2joint[id]].pos
                        for id in ids
                    ]
                    joint_data_dict["pos"].append(ankle_pos_list)

            else:
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

        if is_ankle:
            mighty_zap_pos_dict = {ankle_side: np.array(joint_data_dict["pos"])}
            ankle_pos_dict = sim.postprocess_ankle_pos(mighty_zap_pos_dict)
            joint_data_dict["pos"] = ankle_pos_dict[joint_name]

        if (
            hasattr(sim, "negated_joint_names")
            and joint_name in sim.negated_joint_names
        ):
            joint_data_dict["pos"] = [-pos for pos in joint_data_dict["pos"]]

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
    n_trials,
    duration=3,
    control_dt=0.05,
    frequency_range=(0.5, 2),
    amplitude_min=np.pi / 12,
):
    real_world = RealWorld(robot)
    lower_limit = robot.joints_info[joint_name]["lower_limit"]
    upper_limit = robot.joints_info[joint_name]["upper_limit"]

    # TODO: rotate yaw a little bit when moving hip pitch and knee
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
    title_list = []
    for trial in range(n_trials):
        signal_config = generate_random_sinusoidal_config(
            duration, control_dt, mean, frequency_range, (amplitude_min, amplitude_max)
        )
        signal_time, signal_pos = generate_sinusoidal_signal(signal_config)
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
    real_world_data_dict,
    n_iters=500,
    sampler="TPE",
    damping_range=(0, 2, 1e-3),
    armature_range=(0, 0.1, 1e-4),
    friction_range=(0, 0.2, 1e-4),
):
    signal_config_list = []
    observed_response = []
    for data in real_world_data_dict[joint_name].values():
        signal_config_list.append(data["signal_config"])
        observed_response.append(data["joint_data"]["pos"])

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
