import argparse
import os
import pickle
import time
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from toddlerbot.policies import BasePolicy
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import round_floats
from toddlerbot.utils.misc_utils import (
    dump_profiling_data,
    log,
    precise_sleep,
    # profile,
    snake2camel,
)
from toddlerbot.visualization.vis_plot import (
    plot_joint_tracking,
    plot_joint_tracking_single,
    plot_line_graph,
    plot_loop_time,
)


def plot_results(
    robot: Robot,
    loop_time_list: List[List[float]],
    obs_list: List[Obs],
    motor_angles_list: List[Dict[str, float]],
    control_dt: float,
    exp_folder_path: str,
):
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
    euler_obs_list: List[npt.NDArray[np.float32]] = []
    time_seq_dict: Dict[str, List[float]] = {}
    time_seq_ref_dict: Dict[str, List[float]] = {}
    motor_pos_dict: Dict[str, List[float]] = {}
    motor_vel_dict: Dict[str, List[float]] = {}
    joint_pos_dict: Dict[str, List[float]] = {}
    joint_vel_dict: Dict[str, List[float]] = {}
    for i, obs in enumerate(obs_list):
        time_obs_list.append(obs.time)
        # lin_vel_obs_list.append(obs.lin_vel)
        ang_vel_obs_list.append(obs.ang_vel)
        euler_obs_list.append(obs.euler)

        for j, motor_name in enumerate(robot.motor_ordering):
            if motor_name not in time_seq_dict:
                time_seq_ref_dict[motor_name] = []
                time_seq_dict[motor_name] = []
                motor_pos_dict[motor_name] = []
                motor_vel_dict[motor_name] = []

            # Assume the state fetching is instantaneous
            time_seq_dict[motor_name].append(float(obs.time))
            time_seq_ref_dict[motor_name].append(i * control_dt)
            motor_pos_dict[motor_name].append(obs.motor_pos[j])
            motor_vel_dict[motor_name].append(obs.motor_vel[j])

            joint_name = robot.motor_to_joint_name[motor_name]

            if obs.joint_pos is not None:
                if joint_name not in joint_pos_dict:
                    joint_pos_dict[joint_name] = []

                joint_pos_dict[joint_name].append(obs.joint_pos[j])

            if obs.joint_vel is not None:
                if joint_name not in joint_vel_dict:
                    joint_vel_dict[joint_name] = []

                joint_vel_dict[joint_name].append(obs.joint_vel[j])

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

    plot_loop_time(loop_time_dict, exp_folder_path)

    # plot_line_graph(
    #     np.array(lin_vel_obs_list).T,
    #     time_obs_list,
    #     legend_labels=["X", "Y", "Z"],
    #     title="Linear Velocities Over Time",
    #     x_label="Time (s)",
    #     y_label="Linear Velocity (m/s)",
    #     save_config=True,
    #     save_path=exp_folder_path,
    #     file_name="lin_vel_tracking",
    # )()
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
        motor_vel_dict,
        save_path=exp_folder_path,
    )
    if len(joint_pos_dict) > 0:
        time_seq_dict_joint: Dict[str, List[float]] = {}
        time_seq_ref_dict_joint: Dict[str, List[float]] = {}
        for motor_name, time_seq_ref in time_seq_ref_dict.items():
            joint_name = robot.joint_ordering[robot.motor_ordering.index(motor_name)]
            time_seq_dict_joint[joint_name] = time_seq_dict[motor_name]
            time_seq_ref_dict_joint[joint_name] = time_seq_ref

        plot_joint_tracking(
            time_seq_dict_joint,
            time_seq_ref_dict_joint,
            joint_pos_dict,
            joint_pos_ref_dict,
            robot.joint_limits,
            save_path=exp_folder_path,
            file_name="joint_pos_tracking",
        )


def run_policy(policy: BasePolicy, obs: Obs) -> npt.NDArray[np.float32]:
    return policy.step(obs)


# @profile()
def main(robot: Robot, sim: BaseSim, policy: BasePolicy, debug: Dict[str, Any]):
    header_name = snake2camel(sim.name)

    loop_time_list: List[List[float]] = []
    obs_list: List[Obs] = []
    motor_angles_list: List[Dict[str, float]] = []

    is_prepared = False
    num_total_steps = (
        float("inf")
        if "real" in sim.name and "fixed" not in policy.name
        else policy.num_total_steps
    )
    p_bar = tqdm(total=num_total_steps, desc="Running the policy")
    start_time = time.time()
    step_idx = 0
    time_until_next_step = 0
    try:
        while step_idx < num_total_steps:
            step_start = time.time()

            # Get the latest state from the queue
            obs = sim.get_observation()
            obs.time -= start_time

            if "real" in sim.name:
                assert isinstance(sim, RealWorld)
                if not is_prepared and obs.time > policy.prep_duration:
                    is_prepared = True
                    sim.imu.set_zero_pose()
            else:
                obs.time += time_until_next_step

            obs_time = time.time()

            action = run_policy(policy, obs)
            inference_time = time.time()

            motor_angles: Dict[str, float] = {}
            for motor_name, act in zip(robot.motor_ordering, action):
                motor_angles[motor_name] = act

            sim.set_motor_angles(motor_angles)
            set_action_time = time.time()

            sim.step()
            sim_step_time = time.time()

            obs_list.append(obs)
            motor_angles_list.append(motor_angles)

            step_idx += 1

            p_bar_steps = int(1 / policy.control_dt)
            if step_idx % p_bar_steps == 0:
                p_bar.update(p_bar_steps)

            if debug["log"]:
                log(
                    f"obs: {round_floats(obs.__dict__, 4)}",
                    header=header_name,
                    level="debug",
                )
                log(
                    f"Joint angles: {round_floats(motor_angles,4)}",
                    header=header_name,
                    level="debug",
                )

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
            if "real" in sim.name and time_until_next_step > 0:
                precise_sleep(time_until_next_step)

    except KeyboardInterrupt:
        log("KeyboardInterrupt recieved. Closing...", header=header_name)

    finally:
        p_bar.close()

        exp_name = f"{robot.name}_{policy.name}_{sim.name}"
        time_str = time.strftime("%Y%m%d_%H%M%S")
        exp_folder_path = f"results/{exp_name}_{time_str}"

        os.makedirs(exp_folder_path, exist_ok=True)

        log_data_dict: Dict[str, Any] = {
            "obs_list": obs_list,
            "motor_angles_list": motor_angles_list,
        }
        if "sysID" in policy.name:
            assert isinstance(policy, SysIDFixedPolicy)
            log_data_dict["time_mark_dict"] = policy.time_mark_dict

        log_data_path = os.path.join(exp_folder_path, "log_data.pkl")
        with open(log_data_path, "wb") as f:
            pickle.dump(log_data_dict, f)

        if debug["render"] and hasattr(sim, "save_recording"):
            assert isinstance(sim, MuJoCoSim)
            sim.save_recording(exp_folder_path, policy.control_dt, 2)

        sim.close()

        prof_path = os.path.join(exp_folder_path, "profile_output.lprof")
        dump_profiling_data(prof_path)

        # if debug["plot"]:
        #     log("Visualizing...", header="Walking")
        #     plot_results(
        #         robot,
        #         loop_time_list,
        #         obs_list,
        #         motor_angles_list,
        #         policy.control_dt,
        #         exp_folder_path,
        #     )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the walking simulation.")
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
        "--vis",
        type=str,
        default="render",
        help="The visualization type.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="stand",
        help="The name of the task.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="The policy checkpoint to load.",
    )
    args = parser.parse_args()

    robot = Robot(args.robot)

    if args.sim == "mujoco":
        from toddlerbot.sim.mujoco_sim import MuJoCoSim

        sim = MuJoCoSim(robot, vis_type=args.vis, fixed_base="fixed" in args.policy)
        init_motor_pos = sim.get_observation().motor_pos

    elif args.sim == "real":
        from toddlerbot.sim.real_world import RealWorld

        sim = RealWorld(robot)
        init_motor_pos = sim.get_observation(retries=-1).motor_pos

    else:
        raise ValueError("Unknown simulator")

    if "stand_open" in args.policy:
        from toddlerbot.policies.stand_open import StandPolicy

        policy = StandPolicy(robot, init_motor_pos)

    elif "rotate_torso_open" in args.policy:
        from toddlerbot.policies.rotate_torso_open import RotateTorsoPolicy

        policy = RotateTorsoPolicy(robot, init_motor_pos)

    elif "squat_open" in args.policy:
        from toddlerbot.policies.squat_open import SquatPolicy

        policy = SquatPolicy(robot)

    elif "sysID_fixed" in args.policy:
        from toddlerbot.policies.sysID_fixed import SysIDFixedPolicy

        policy = SysIDFixedPolicy(robot, init_motor_pos)

    elif "walk_fixed" in args.policy:
        from toddlerbot.policies.walk_fixed import WalkFixedPolicy

        policy = WalkFixedPolicy(robot, init_motor_pos, args.ckpt)

    elif "walk" in args.policy:
        from toddlerbot.policies.walk import WalkPolicy

        policy = WalkPolicy(robot, init_motor_pos, args.ckpt)

    elif "rotate_torso" in args.policy:
        from toddlerbot.policies.rotate_torso import RotateTorsoPolicy

        policy = RotateTorsoPolicy(args.policy, robot, init_motor_pos, args.ckpt)

    elif "squat" in args.policy:
        from toddlerbot.policies.squat import SquatPolicy

        policy = SquatPolicy(args.policy, robot, init_motor_pos, args.ckpt)

    else:
        raise ValueError("Unknown policy")

    debug_config: Dict[str, Any] = {"log": False, "plot": True, "render": True}

    main(robot, sim, policy, debug_config)
