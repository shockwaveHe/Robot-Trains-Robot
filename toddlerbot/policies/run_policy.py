import argparse
import bisect
import importlib
import json
import os
import pickle
import pkgutil
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

from toddlerbot.policies import BasePolicy, get_policy_class, get_policy_names
from toddlerbot.policies.balance_pd import BalancePDPolicy
from toddlerbot.policies.calibrate import CalibratePolicy
from toddlerbot.policies.dp_policy import DPPolicy
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.policies.push_cart import PushCartPolicy
from toddlerbot.policies.record import RecordPolicy
from toddlerbot.policies.replay import ReplayPolicy
from toddlerbot.policies.sysID import SysIDFixedPolicy
from toddlerbot.policies.teleop_follower_pd import TeleopFollowerPDPolicy
from toddlerbot.policies.teleop_joystick import TeleopJoystickPolicy
from toddlerbot.policies.teleop_leader import TeleopLeaderPolicy
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.real_world import RealWorld
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.comm_utils import sync_time
from toddlerbot.utils.misc_utils import dump_profiling_data, log, snake2camel
from toddlerbot.visualization.vis_plot import (
    plot_joint_tracking,
    plot_joint_tracking_frequency,
    plot_joint_tracking_single,
    plot_line_graph,
    plot_loop_time,
    plot_motor_vel_tor_mapping,
    # plot_path_tracking,
)

# from toddlerbot.utils.misc_utils import profile


def dynamic_import_policies(policy_package: str):
    """Dynamically imports all modules within a specified package.

    This function attempts to import each module found in the given package directory. If a module cannot be imported, a log message is generated.

    Args:
        policy_package (str): The name of the package containing the modules to be imported.
    """
    package = importlib.import_module(policy_package)
    package_path = package.__path__

    # Iterate over all modules in the given package directory
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        full_module_name = f"{policy_package}.{module_name}"
        try:
            importlib.import_module(full_module_name)
        except Exception:
            log(f"Could not import {full_module_name}", header="Dynamic Import")


# Call this to import all policies dynamically
dynamic_import_policies("toddlerbot.policies")


def plot_results(
    robot: Robot,
    loop_time_list: List[List[float]],
    obs_list: List[Obs],
    control_inputs_list: List[Dict[str, float]],
    motor_angles_list: List[Dict[str, float]],
    exp_folder_path: str,
):
    """Generates and saves various plots to visualize the performance and behavior of a robot during an experiment.

    Args:
        robot (Robot): The robot object containing information about the robot's configuration and state.
        loop_time_list (List[List[float]]): A list of lists containing timing information for each loop iteration.
        obs_list (List[Obs]): A list of observations recorded during the experiment.
        control_inputs_list (List[Dict[str, float]]): A list of dictionaries containing control inputs applied to the robot.
        motor_angles_list (List[Dict[str, float]]): A list of dictionaries containing motor angles recorded during the experiment.
        exp_folder_path (str): The path to the folder where the plots will be saved.
    """
    loop_time_dict: Dict[str, List[float]] = {
        "obs_time": [],
        "inference_time": [],
        "set_action_time": [],
        "sim_step_time": [],
        "log_time": [],
        # "total_time": [],
    }
    for i, loop_time in enumerate(loop_time_list):
        (
            step_start,
            obs_time,
            inference_time,
            set_action_time,
            sim_step_time,
            step_end,
        ) = loop_time
        loop_time_dict["obs_time"].append((obs_time - step_start) * 1000)
        loop_time_dict["inference_time"].append((inference_time - obs_time) * 1000)
        loop_time_dict["set_action_time"].append(
            (set_action_time - inference_time) * 1000
        )
        loop_time_dict["sim_step_time"].append((sim_step_time - set_action_time) * 1000)
        loop_time_dict["log_time"].append((step_end - sim_step_time) * 1000)
        # loop_time_dict["total_time"].append((step_end - step_start) * 1000)

    time_obs_list: List[float] = []
    # lin_vel_obs_list: List[npt.NDArray[np.float32]] = []
    ang_vel_obs_list: List[npt.NDArray[np.float32]] = []
    pos_obs_list: List[npt.NDArray[np.float32]] = []
    euler_obs_list: List[npt.NDArray[np.float32]] = []
    tor_obs_total_list: List[float] = []
    time_seq_dict: Dict[str, List[float]] = {}
    time_seq_ref_dict: Dict[str, List[float]] = {}
    motor_pos_dict: Dict[str, List[float]] = {}
    motor_vel_dict: Dict[str, List[float]] = {}
    motor_tor_dict: Dict[str, List[float]] = {}
    for i, obs in enumerate(obs_list):
        time_obs_list.append(obs.time)
        # lin_vel_obs_list.append(obs.lin_vel)
        ang_vel_obs_list.append(obs.ang_vel)
        pos_obs_list.append(obs.pos)
        euler_obs_list.append(obs.euler)
        tor_obs_total_list.append(sum(obs.motor_tor))

        for j, motor_name in enumerate(robot.motor_ordering):
            if motor_name not in time_seq_dict:
                time_seq_ref_dict[motor_name] = []
                time_seq_dict[motor_name] = []
                motor_pos_dict[motor_name] = []
                motor_vel_dict[motor_name] = []
                motor_tor_dict[motor_name] = []

            # Assume the state fetching is instantaneous
            time_seq_dict[motor_name].append(float(obs.time))
            time_seq_ref_dict[motor_name].append(float(obs.time))
            # time_seq_ref_dict[motor_name].append(i * policy.control_dt)
            motor_pos_dict[motor_name].append(obs.motor_pos[j])
            motor_vel_dict[motor_name].append(obs.motor_vel[j])
            motor_tor_dict[motor_name].append(obs.motor_tor[j])

    action_dict: Dict[str, List[float]] = {}
    joint_pos_ref_dict: Dict[str, List[float]] = {}
    for motor_angles in motor_angles_list:
        for motor_name, motor_angle in motor_angles.items():
            if motor_name not in action_dict:
                action_dict[motor_name] = []
            action_dict[motor_name].append(motor_angle)

        joint_angle_ref = robot.motor_to_joint_angles(motor_angles)
        for joint_name, joint_angle in joint_angle_ref.items():
            if joint_name not in joint_pos_ref_dict:
                joint_pos_ref_dict[joint_name] = []
            joint_pos_ref_dict[joint_name].append(joint_angle)

    control_inputs_dict: Dict[str, List[float]] = {}
    for control_inputs in control_inputs_list:
        for control_name, control_input in control_inputs.items():
            if control_name not in control_inputs_dict:
                control_inputs_dict[control_name] = []
            control_inputs_dict[control_name].append(control_input)

    plt.switch_backend("Agg")

    plot_loop_time(loop_time_dict, exp_folder_path)

    if "sysID" in robot.name:
        plot_motor_vel_tor_mapping(
            motor_vel_dict["joint_0"],
            motor_tor_dict["joint_0"],
            save_path=exp_folder_path,
        )

    # if hasattr(policy, "com_pos_list"):
    #     plot_len = min(len(policy.com_pos_list), len(time_obs_list))
    #     plot_line_graph(
    #         np.array(policy.com_pos_list).T[:2, :plot_len],
    #         time_obs_list[:plot_len],
    #         legend_labels=["COM X", "COM Y"],
    #         title="Center of Mass Over Time",
    #         x_label="Time (s)",
    #         y_label="COM Position (m)",
    #         save_config=True,
    #         save_path=exp_folder_path,
    #         file_name="com_tracking",
    #     )()

    plot_line_graph(
        tor_obs_total_list,
        time_obs_list,
        legend_labels=["Torque (Nm) or Current (mA)"],
        title="Total Torque or Current  Over Time",
        x_label="Time (s)",
        y_label="Torque (Nm) or Current (mA)",
        save_config=True,
        save_path=exp_folder_path,
        file_name="total_tor_tracking",
    )()
    plot_line_graph(
        np.array(ang_vel_obs_list).T,
        time_obs_list,
        legend_labels=["Roll (X)", "Pitch (Y)", "Yaw (Z)"],
        title="Angular Velocities Over Time",
        x_label="Time (s)",
        y_label="Angular Velocity (rad/s)",
        save_config=True,
        save_path=exp_folder_path,
        file_name="ang_vel_tracking",
    )()
    plot_line_graph(
        np.array(euler_obs_list).T,
        time_obs_list,
        legend_labels=["Roll (X)", "Pitch (Y)", "Yaw (Z)"],
        title="Euler Angles Over Time",
        x_label="Time (s)",
        y_label="Euler Angles (rad)",
        save_config=True,
        save_path=exp_folder_path,
        file_name="euler_tracking",
    )()
    # if len(control_inputs_dict) > 0:
    #     plot_path_tracking(
    #         time_obs_list,
    #         pos_obs_list,
    #         euler_obs_list,
    #         control_inputs_dict,
    #         save_path=exp_folder_path,
    #     )
    plot_joint_tracking(
        time_seq_dict,
        time_seq_ref_dict,
        motor_pos_dict,
        action_dict,
        robot.joint_limits,
        save_path=exp_folder_path,
    )
    plot_joint_tracking_single(
        time_seq_dict,
        motor_tor_dict,
        save_path=exp_folder_path,
        y_label="Torque (Nm) or Current (mA)",
        file_name="motor_tor_tracking",
    )
    plot_joint_tracking_single(
        time_seq_dict,
        motor_vel_dict,
        save_path=exp_folder_path,
    )
    plot_joint_tracking_frequency(
        time_seq_dict,
        time_seq_ref_dict,
        motor_pos_dict,
        action_dict,
        save_path=exp_folder_path,
    )


# @profile()
def run_policy(
    robot: Robot, sim: BaseSim, policy: BasePolicy, vis_type: str, plot: bool
):
    """Executes a control policy on a robot within a simulation environment, logging data and optionally visualizing results.

    Args:
        robot (Robot): The robot instance to control.
        sim (BaseSim): The simulation environment in which the robot operates.
        policy (BasePolicy): The control policy to execute.
        vis_type (str): The type of visualization to use ('view', 'render', etc.).
        plot (bool): Whether to plot the results after execution.
    """
    header_name = snake2camel(sim.name)

    loop_time_list: List[List[float]] = []
    obs_list: List[Obs] = []
    control_inputs_list: List[Dict[str, float]] = []
    motor_angles_list: List[Dict[str, float]] = []

    n_steps_total = (
        float("inf")
        if "real" in sim.name and "fixed" not in policy.name
        else policy.n_steps_total
    )
    p_bar = tqdm(total=n_steps_total, desc="Running the policy")
    start_time = time.time()
    step_idx = 0
    time_until_next_step = 0.0
    last_ckpt_idx = -1
    try:
        while step_idx < n_steps_total:
            step_start = time.time()

            # Get the latest state from the queue
            obs = sim.get_observation()
            obs.time -= start_time

            if "real" not in sim.name and vis_type != "view":
                obs.time += time_until_next_step

            obs_time = time.time()

            if isinstance(policy, SysIDFixedPolicy):
                ckpt_times = list(policy.ckpt_dict.keys())
                ckpt_idx = bisect.bisect_left(ckpt_times, obs.time)
                ckpt_idx = min(ckpt_idx, len(ckpt_times) - 1)
                if ckpt_idx != last_ckpt_idx:
                    motor_kps = policy.ckpt_dict[ckpt_times[ckpt_idx]]
                    motor_kps_updated = {}
                    for joint_name in motor_kps:
                        for motor_name in robot.joint_to_motor_name[joint_name]:
                            motor_kps_updated[motor_name] = motor_kps[joint_name]

                    if np.any(list(motor_kps_updated.values())):
                        sim.set_motor_kps(motor_kps_updated)
                        last_ckpt_idx = ckpt_idx

            # need to enable and disable motors according to logging state
            if isinstance(policy, TeleopLeaderPolicy) and policy.toggle_motor:
                assert isinstance(sim, RealWorld)
                if policy.is_running:
                    # disable all motors when logging
                    sim.dynamixel_controller.disable_motors()
                else:
                    # enable all motors when not logging
                    sim.dynamixel_controller.enable_motors()

                policy.toggle_motor = False

            elif isinstance(policy, RecordPolicy) and policy.toggle_motor:
                assert isinstance(sim, RealWorld)
                sim.dynamixel_controller.disable_motors(policy.disable_motor_indices)
                policy.toggle_motor = False

            control_inputs, motor_target = policy.step(obs, "real" in sim.name)

            inference_time = time.time()

            motor_angles: Dict[str, float] = {}
            for motor_name, motor_angle in zip(robot.motor_ordering, motor_target):
                motor_angles[motor_name] = motor_angle

            sim.set_motor_target(motor_angles)
            set_action_time = time.time()

            sim.step()
            sim_step_time = time.time()

            obs_list.append(obs)
            control_inputs_list.append(control_inputs)
            motor_angles_list.append(motor_angles)

            step_idx += 1

            p_bar_steps = int(1 / policy.control_dt)
            if step_idx % p_bar_steps == 0:
                p_bar.update(p_bar_steps)

            step_end = time.time()

            loop_time_list.append(
                [
                    step_start,
                    obs_time,
                    inference_time,
                    set_action_time,
                    sim_step_time,
                    step_end,
                ]
            )

            time_until_next_step = start_time + policy.control_dt * step_idx - step_end
            # print(f"time_until_next_step: {time_until_next_step * 1000:.2f} ms")
            if ("real" in sim.name or vis_type == "view") and time_until_next_step > 0:
                time.sleep(time_until_next_step)

    except KeyboardInterrupt:
        log("KeyboardInterrupt recieved. Closing...", header=header_name)

    finally:
        p_bar.close()

        exp_name = f"{robot.name}_{policy.name}_{sim.name}"
        time_str = time.strftime("%Y%m%d_%H%M%S")
        exp_folder_path = f"results/{exp_name}_{time_str}"

        os.makedirs(exp_folder_path, exist_ok=True)

        if vis_type == "render" and hasattr(sim, "save_recording"):
            assert isinstance(sim, MuJoCoSim)
            sim.save_recording(exp_folder_path, policy.control_dt, 2)

        sim.close()

    log_data_dict: Dict[str, Any] = {
        "obs_list": obs_list,
        "control_inputs_list": control_inputs_list,
        "motor_angles_list": motor_angles_list,
    }

    if isinstance(policy, SysIDFixedPolicy):
        log_data_dict["ckpt_dict"] = policy.ckpt_dict

    log_data_path = os.path.join(exp_folder_path, "log_data.pkl")
    with open(log_data_path, "wb") as f:
        pickle.dump(log_data_dict, f)

    prof_path = os.path.join(exp_folder_path, "profile_output.lprof")
    dump_profiling_data(prof_path)

    if isinstance(policy, TeleopFollowerPDPolicy):
        policy.dataset_logger.move_files_to_exp_folder(exp_folder_path)

    if isinstance(policy, DPPolicy) and len(policy.camera_frame_list) > 0:
        fps = int(1 / np.diff(policy.camera_time_list).mean())
        log(f"visual_obs fps: {fps}", header=header_name)
        video_path = os.path.join(exp_folder_path, "visual_obs.mp4")
        video_clip = ImageSequenceClip(policy.camera_frame_list, fps=fps)
        video_clip.write_videofile(video_path, codec="libx264", fps=fps)

    if isinstance(policy, ReplayPolicy):
        with open(os.path.join(exp_folder_path, "keyframes.pkl"), "wb") as f:
            pickle.dump(policy.keyframes, f)

    if isinstance(policy, CalibratePolicy):
        motor_config_path = os.path.join(robot.root_path, "config_motors.json")
        if os.path.exists(motor_config_path):
            motor_names = robot.get_joint_attrs("is_passive", False)
            motor_pos_init = np.array(
                robot.get_joint_attrs("is_passive", False, "init_pos")
            )
            motor_pos_delta = (
                np.array(list(motor_angles_list[-1].values()), dtype=np.float32)
                - policy.default_motor_pos
            )
            motor_pos_delta[
                np.logical_and(motor_pos_delta > -0.005, motor_pos_delta < 0.005)
            ] = 0.0

            with open(motor_config_path, "r") as f:
                motor_config = json.load(f)

            for motor_name, init_pos in zip(
                motor_names, motor_pos_init + motor_pos_delta
            ):
                motor_config[motor_name]["init_pos"] = float(init_pos)

            with open(motor_config_path, "w") as f:
                json.dump(motor_config, f, indent=4)
        else:
            raise FileNotFoundError(f"Could not find {motor_config_path}")

    if isinstance(policy, PushCartPolicy):
        video_path = os.path.join(exp_folder_path, "visual_obs.mp4")
        fps = int(1 / np.diff(policy.grasp_policy.camera_time_list).mean())
        log(f"visual_obs fps: {fps}", header=header_name)
        video_clip = ImageSequenceClip(policy.grasp_policy.camera_frame_list, fps=fps)
        video_clip.write_videofile(video_path, codec="libx264", fps=fps)

    if isinstance(policy, TeleopJoystickPolicy):
        policy_dict = {
            "hug": policy.hug_policy,
            "pick": policy.pick_policy,
            "grasp": policy.push_cart_policy.grasp_policy
            if hasattr(policy.push_cart_policy, "grasp_policy")
            else policy.teleop_policy,
        }
        for task_name, task_policy in policy_dict.items():
            if (
                not isinstance(task_policy, DPPolicy)
                or len(task_policy.camera_frame_list) == 0
            ):
                continue

            video_path = os.path.join(exp_folder_path, f"{task_name}_visual_obs.mp4")
            fps = int(1 / np.diff(task_policy.camera_time_list).mean())
            log(f"{task_name} visual_obs fps: {fps}", header=header_name)
            video_clip = ImageSequenceClip(task_policy.camera_frame_list, fps=fps)
            video_clip.write_videofile(video_path, codec="libx264", fps=fps)

    if plot:
        log("Visualizing...", header=header_name)
        plot_results(
            robot,
            loop_time_list,
            obs_list,
            control_inputs_list,
            motor_angles_list,
            exp_folder_path,
        )


def main(args=None):
    """Executes a policy for a specified robot and simulator configuration.

    This function parses command-line arguments to configure and run a policy for a robot. It supports different robots, simulators, visualization types, and tasks. The function initializes the appropriate simulation environment and policy based on the provided arguments and executes the policy.

    Args:
        args (list, optional): List of command-line arguments. If None, defaults to sys.argv.

    Raises:
        ValueError: If an unknown simulator is specified.
        AssertionError: If the teleop leader policy is used with an unsupported robot or simulator.
    """
    parser = argparse.ArgumentParser(description="Run a policy.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in descriptions.",
        choices=[
            "toddlerbot",
            "toddlerbot_gripper",
            "toddlerbot_arms",
            "toddlerbot_active",
        ],
    )
    parser.add_argument(
        "--sim",
        type=str,
        default="mujoco",
        help="The name of the simulator to use.",
        choices=["mujoco", "real"],
    )
    parser.add_argument(
        "--vis",
        type=str,
        default="render",
        help="The visualization type.",
        choices=["render", "view", "none"],
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="stand",
        help="The name of the task.",
        choices=get_policy_names(),
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="The policy checkpoint to load for RL policies.",
    )
    parser.add_argument(
        "--command",
        type=str,
        default="",
        help="The policy checkpoint to load for RL policies.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="The policy run to replay.",
    )
    parser.add_argument(
        "--ip",
        type=str,
        default="",
        help="The ip address of the follower.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        # choices=["hug", "pick", "grasp"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot",
        default=True,
        help="Skip the plot functions.",
    )

    parser.add_argument(
        "--teleop_mujoco_path",
        type=str,
        default="toddlerbot/descriptions/toddlerbot_active/toddlerbot_active_scene_teleop.xml",
        help="The xml configuration file for the teleop mujoco simulation.",
    )
    parser.add_argument(
        "--teleop_ik_config",
        type=str,
        default="toddlerbot/manipulation/teleoperation/ik_configs/quest_toddy.yaml",
        help="The ik configuration file for the teleop mujoco simulation.",
    )
    args = parser.parse_args(args)

    robot = Robot(args.robot)

    # t1 = time.time()

    sim: BaseSim | None = None
    if args.sim == "mujoco":
        sim = MuJoCoSim(robot, vis_type=args.vis, fixed_base="fixed" in args.policy)
        init_motor_pos = sim.get_observation().motor_pos

    elif args.sim == "real":
        sim = RealWorld(robot)
        init_motor_pos = sim.get_observation(retries=-1).motor_pos

    else:
        raise ValueError("Unknown simulator")

    # t2 = time.time()

    PolicyClass = get_policy_class(args.policy.replace("_fixed", ""))

    if "replay" in args.policy:
        policy = PolicyClass(args.policy, robot, init_motor_pos, args.run_name)

    elif "teleop_leader" in args.policy:
        assert args.robot == "toddlerbot_arms", (
            "The teleop leader policy is only for the arms"
        )
        assert args.sim == "real", (
            "The sim needs to be the real world for the teleop leader policy"
        )
        for motor_name in robot.motor_ordering:
            for gain_name in ["kp_real", "kd_real", "kff1_real", "kff2_real"]:
                robot.config["joints"][motor_name][gain_name] = 0.0

        policy = PolicyClass(
            args.policy, robot, init_motor_pos, ip=args.ip, task=args.task
        )  # type: ignore

    elif "teleop_follower" in args.policy:
        # Run the command
        if len(args.ip) > 0:
            sync_time(args.ip)

        policy = PolicyClass(
            args.policy, robot, init_motor_pos, ip=args.ip, task=args.task
        )  # type: ignore

    elif "teleop_joystick" in args.policy:
        if len(args.ip) > 0:
            sync_time(args.ip)

        policy = PolicyClass(  # type: ignore
            args.policy, robot, init_motor_pos, ip=args.ip, run_name=args.run_name
        )

    elif "push_cart" in args.policy:
        policy = PolicyClass(args.policy, robot, init_motor_pos, args.ckpt)

    elif "teleop_vr_leader" in args.policy:
        policy = PolicyClass(
            args.policy,
            robot,
            init_motor_pos,
            args.teleop_mujoco_path,
            args.teleop_ik_config,
            ip=args.ip,
        )
    elif issubclass(PolicyClass, MJXPolicy):
        fixed_command = None
        if len(args.command) > 0:
            fixed_command = np.array(args.command.split(" "), dtype=np.float32)

        policy = PolicyClass(
            args.policy, robot, init_motor_pos, args.ckpt, fixed_command=fixed_command
        )

    elif issubclass(PolicyClass, DPPolicy):
        policy = PolicyClass(
            args.policy, robot, init_motor_pos, args.ckpt, task=args.task
        )

    elif issubclass(PolicyClass, BalancePDPolicy):
        # Run the command
        if len(args.ip) > 0:
            sync_time(args.ip)

        fixed_command = None
        if len(args.command) > 0:
            fixed_command = np.array(args.command.split(" "), dtype=np.float32)

        policy = PolicyClass(
            args.policy, robot, init_motor_pos, ip=args.ip, fixed_command=fixed_command
        )
    elif "talk" in args.policy:
        policy = PolicyClass(args.policy, robot, init_motor_pos, ip=args.ip)  # type:ignore
    else:
        policy = PolicyClass(args.policy, robot, init_motor_pos)

    # t3 = time.time()

    # print(f"Time taken to initialize sim: {t2 - t1:.2f} s")
    # print(f"Time taken to initialize policy: {t3 - t2:.2f} s")

    run_policy(robot, sim, policy, args.vis, args.plot)


if __name__ == "__main__":
    main()
