import argparse
import json
import os
import pickle
import shutil
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import optuna

from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.real_world import RealWorld
from toddlerbot.sim.robot import HumanoidRobot
from toddlerbot.utils.file_utils import find_description_path
from toddlerbot.utils.misc_utils import *
from toddlerbot.utils.vis_plot import *


def params_array_to_dict(params):
    """
    Convert an array of parameters to a dictionary.
    """
    return {
        "damping": params[0],
        "armature": params[1],
        "friction": params[2],
        "p_gain": params[3],
    }


def generate_sinusoidal_signal(signal_config):
    """
    Generates a sinusoidal signal based on the given parameters.
    """
    t = np.arange(0, signal_config["duration"], 1 / signal_config["sampling_rate"])
    signal = signal_config["amplitude"] * np.sin(
        2 * np.pi * signal_config["frequency"] * t
    )
    return t, signal


def actuate_and_collect_data(sim, robot, joint_name, signal_pos, sampling_rate):
    """
    Actuates a single joint with the given signal and collects the response.
    """
    # Convert signal time to sleep time between updates
    sleep_time = 1 / sampling_rate
    initial_joint_angles = robot.initialize_joint_angles()

    joint_data_dict = {"pos": [], "time": []}
    time_start = time.time()
    for angle in signal_pos:
        # Update the joint angle
        time_curr = time.time() - time_start
        joint_angles = initial_joint_angles.copy()
        joint_angles[joint_name] = angle
        sim.set_joint_angles(robot, joint_angles, interp=False)
        if sim.name == "mujoco":
            mujoco.mj_step(sim.model, sim.data)

        joint_state_dict = sim.get_joint_state(robot)

        joint_data_dict["time"].append(joint_state_dict[joint_name].time - time_start)

        pos = joint_state_dict[joint_name].pos
        if sim.name == "real_world" and joint_name in sim.negated_joint_names:
            pos *= -1
        joint_data_dict["pos"].append(pos)

        time_elapsed = time.time() - time_start - time_curr
        time_unil_next_step = sleep_time - time_elapsed
        if time_unil_next_step > 0:
            sleep(time_unil_next_step)

    # Reset the joint to its initial position
    sim.set_joint_angles(robot, initial_joint_angles, interp=False)

    return joint_data_dict


def update_actuator_xml(robot, xml_path, new_params):
    """
    Update the MuJoCo XML file with new actuator parameters.
    """
    # Load the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    motor_params = robot.config.motor_params

    params_dict = params_array_to_dict(new_params)

    # Iterate over all joints in the XML
    for joint in root.findall(".//joint"):
        joint_name = joint.get("name")
        # Check if the joint name is in the provided armature dictionary
        if joint_name in motor_params:
            for param_name, param_value in params_dict.items():
                if param_name in ["damping", "armature"]:
                    joint.set(param_name, str(param_value))

    # Create <actuator> element if it doesn't exist
    actuator = root.find("./actuator")

    for joint in root.findall(".//joint"):
        joint_name = joint.get("name")
        if joint_name in motor_params:
            motor_name = f"{joint_name}_act"
            motor = actuator.find(f".//position[@name='{motor_name}']")
            # Actuator exists, update its attributes
            motor.set("kp", str(params_dict["p_gain"]))
            motor.set("kv", str(0.0))

    default = root.find("default")
    default_joint = default.find(".//joint")
    default_joint.set("frictionloss", str(params_dict["friction"]))

    # Save the modified XML
    tree.write(xml_path)


def optimize_parameters(
    robot,
    joint_name,
    xml_path,
    signal_pos,
    sampling_rate,
    observed_response,
):
    def objective(trial: optuna.Trial):
        # Calculate the model response using the current set of parameters
        damping = trial.suggest_float("damping", 0, 2)
        armature = trial.suggest_float("armature", 0, 0.1)
        friction = trial.suggest_float("friction", 0, 1)
        p_gain = trial.suggest_float("p_gain", 0, 1000)

        update_actuator_xml(robot, xml_path, [damping, armature, friction, p_gain])
        mujoco_sim = MuJoCoSim(robot, xml_path, fixed=True)

        model_response = actuate_and_collect_data(
            mujoco_sim, robot, joint_name, signal_pos, sampling_rate
        )
        error = np.linalg.norm(
            np.array(observed_response) - np.array(model_response["pos"])
        )
        return error

    sampler = optuna.samplers.CmaEsSampler()  # CMA-ES sampler
    n_trials = 2000
    # sampler = optuna.samplers.TPESampler() # TPE sampler
    # n_trials = 500
    study = optuna.create_study(storage="sqlite:///db.sqlite3", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=True)


def run_actuation(
    sim, robot, name, signal_pos, signal_time, sampling_rate, exp_folder_path
):
    log(f"Actuating {name} in {sim.name}...", header="SysID", level="debug")
    joint_data_dict = actuate_and_collect_data(
        sim, robot, name, signal_pos, sampling_rate
    )
    log(
        f"Finished actuating {name} in the real world...", header="SysID", level="debug"
    )

    plot_line_graph(
        [joint_data_dict["pos"], list(signal_pos)],
        x=[joint_data_dict["time"], list(signal_time)],
        title=f"{name} {sim.name} sysID data",
        x_label="Time",
        y_label="Position",
        save_config=True,
        save_path=exp_folder_path,
        file_suffix="",
        legend_labels=["Real", "Reference"],
    )()

    joint_data_file_name = f"{name}_{sim.name}_data.pkl"
    with open(os.path.join(exp_folder_path, joint_data_file_name), "wb") as f:
        pickle.dump(joint_data_dict, f)


def main():
    parser = argparse.ArgumentParser(description="Run the SysID data collection.")
    parser.add_argument(
        "--robot-name",
        type=str,
        default="sustaina_op",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    args = parser.parse_args()

    exp_name = f"sysID_{args.robot_name}"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    exp_folder_path = f"results/{time_str}_{exp_name}"
    exp_folder_path = "results/20240410_192823_sysID_toddlerbot_legs"

    os.makedirs(exp_folder_path, exist_ok=True)

    fixed_xml_path = find_description_path(args.robot_name, suffix="_fixed.xml")
    new_xml_path = os.path.join(
        os.path.dirname(fixed_xml_path), f"{args.robot_name}_sysID.xml"
    )
    shutil.copy2(fixed_xml_path, new_xml_path)

    robot = HumanoidRobot(args.robot_name)
    # real_world = RealWorld(robot)

    signal_config = {
        "frequency": 0.5,
        "amplitude": np.pi / 4,
        "duration": 10,
        "sampling_rate": 30,
    }
    with open(os.path.join(exp_folder_path, "signal_config.json"), "w") as f:
        json.dump(signal_config, f, indent=4)

    # Generate the control signal
    signal_time, signal_pos = generate_sinusoidal_signal(signal_config)

    # Actuate each joint one by one
    for name, info in robot.joints_info.items():
        if not info["active"]:
            continue

        if name != "left_hip_yaw":
            continue

        # run_actuation(
        #     real_world, robot, name, signal_pos, signal_time, sampling_rate, exp_folder_path
        # )

        with open(os.path.join(exp_folder_path, f"{name}_data.pkl"), "rb") as f:
            joint_data_dict = pickle.load(f)

        optimize_parameters(
            robot,
            name,
            new_xml_path,
            signal_pos,
            signal_config["sampling_rate"],
            joint_data_dict["pos"],
        )

        mujoco_sim = MuJoCoSim(robot, new_xml_path, fixed=True)
        run_actuation(
            mujoco_sim,
            robot,
            name,
            signal_pos,
            signal_time,
            signal_config["sampling_rate"],
            exp_folder_path,
        )

    # Close simulation and controllers properly
    # sim.close()


if __name__ == "__main__":
    main()
