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
    plot_angular_velocity_tracking,
    plot_joint_angle_tracking,
    plot_joint_velocity_tracking,
    plot_loop_time,
    plot_orientation_tracking,
)


def plot_results(
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
    euler_obs_list: List[npt.NDArray[np.float32]] = []
    ang_vel_obs_list: List[npt.NDArray[np.float32]] = []
    time_seq_dict: Dict[str, List[float]] = {}
    time_seq_ref_dict: Dict[str, List[float]] = {}
    motor_angle_dict: Dict[str, List[float]] = {}
    joint_angle_dict: Dict[str, List[float]] = {}
    joint_vel_dict: Dict[str, List[float]] = {}
    for i, obs in enumerate(obs_list):
        time_obs_list.append(obs.time)
        euler_obs_list.append(obs.imu_euler)
        ang_vel_obs_list.append(obs.imu_ang_vel)

        for j, motor_name in enumerate(robot.motor_ordering):
            if motor_name not in motor_angle_dict:
                motor_angle_dict[motor_name] = []

            motor_angle_dict[motor_name].append(obs.a[j])

        for j, joint_name in enumerate(robot.joint_ordering):
            if joint_name not in time_seq_dict:
                time_seq_ref_dict[joint_name] = []
                time_seq_dict[joint_name] = []
                joint_angle_dict[joint_name] = []
                joint_vel_dict[joint_name] = []

            # Assume the state fetching is instantaneous
            time_seq_dict[joint_name].append(float(obs.time))
            time_seq_ref_dict[joint_name].append(i * control_dt)
            joint_angle_dict[joint_name].append(obs.q[j])
            joint_vel_dict[joint_name].append(obs.dq[j])

    time_seq_dict_copy: Dict[str, List[float]] = {}
    time_seq_ref_dict_copy: Dict[str, List[float]] = {}
    for joint_name, time_seq_ref in time_seq_ref_dict.items():
        motor_name = robot.motor_ordering[robot.joint_ordering.index(joint_name)]
        time_seq_dict_copy[motor_name] = time_seq_dict[joint_name]
        time_seq_ref_dict_copy[motor_name] = time_seq_ref

    motor_angle_ref_dict: Dict[str, List[float]] = {}
    joint_angle_ref_dict: Dict[str, List[float]] = {}
    for motor_angles in motor_angles_list:
        for motor_name, motor_angle in motor_angles.items():
            if motor_name not in motor_angle_ref_dict:
                motor_angle_ref_dict[motor_name] = []
            motor_angle_ref_dict[motor_name].append(motor_angle)

        joint_angle_ref = robot.motor_to_joint_angles(motor_angles)
        for joint_name, joint_angle in joint_angle_ref.items():
            if joint_name not in joint_angle_ref_dict:
                joint_angle_ref_dict[joint_name] = []
            joint_angle_ref_dict[joint_name].append(joint_angle)

    plot_loop_time(loop_time_dict, exp_folder_path)

    plot_orientation_tracking(
        time_obs_list,
        euler_obs_list,
        save_path=exp_folder_path,
    )
    plot_angular_velocity_tracking(
        time_obs_list,
        ang_vel_obs_list,
        save_path=exp_folder_path,
    )
    plot_joint_angle_tracking(
        time_seq_dict,
        time_seq_ref_dict,
        joint_angle_dict,
        joint_angle_ref_dict,
        robot.joint_limits,
        save_path=exp_folder_path,
    )
    plot_joint_angle_tracking(
        time_seq_dict_copy,
        time_seq_ref_dict_copy,
        motor_angle_dict,
        motor_angle_ref_dict,
        robot.joint_limits,
        save_path=exp_folder_path,
        file_name="motor_angle_tracking",
    )
    plot_joint_velocity_tracking(
        time_seq_dict,
        joint_vel_dict,
        save_path=exp_folder_path,
    )


def run_policy(policy: BasePolicy, obs: Obs) -> npt.NDArray[np.float32]:
    return policy.step(obs)


# @profile()
def main(robot: Robot, sim: BaseSim, policy: BasePolicy, debug: Dict[str, Any]):
    header_name = snake2camel(sim.name)

    loop_time_list: List[List[float]] = []
    obs_list: List[Obs] = []
    motor_angles_list: List[Dict[str, float]] = []

    start_time = time.time()
    step_idx = 0
    p_bar = tqdm(total=n_steps, desc="Running the policy")
    time_until_next_step = 0
    try:
        while step_idx < n_steps:
            step_start = time.time()

            # Get the latest state from the queue
            obs = sim.get_observation()
            obs.time -= start_time

            if "real" not in sim.name:
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

        if debug["plot"]:
            log("Visualizing...", header="Walking")
            plot_results(
                loop_time_list,
                obs_list,
                motor_angles_list,
                policy.control_dt,
                exp_folder_path,
            )


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

        sim = MuJoCoSim(robot, vis_type="render", fixed_base="fixed" in args.policy)

    elif args.sim == "real":
        from toddlerbot.sim.real_world import RealWorld

        sim = RealWorld(robot)
        # TODO: Debug IMU
        sim.has_imu = False

    else:
        raise ValueError("Unknown simulator")

    if args.policy == "stand":
        from toddlerbot.policies.stand import StandPolicy

        policy = StandPolicy(robot, sim.get_observation().q)

    elif args.policy == "rotate_torso":
        from toddlerbot.policies.rotate_torso import RotateTorsoPolicy

        policy = RotateTorsoPolicy(robot)

    elif args.policy == "squat":
        from toddlerbot.policies.squat import SquatPolicy

        policy = SquatPolicy(robot)

    elif args.policy == "walk_fixed":
        from toddlerbot.policies.walk_fixed import WalkFixedPolicy

        run_name = f"{args.robot}_{args.policy}_ppo_{args.ckpt}"
        policy = WalkFixedPolicy(robot, run_name, sim.get_observation().q)

    elif args.policy == "walk":
        from toddlerbot.policies.walk import WalkPolicy

        run_name = f"{args.robot}_{args.policy}_ppo_{args.ckpt}"
        policy = WalkPolicy(robot, run_name)

    elif args.policy == "sysID_fixed":
        from toddlerbot.policies.sysID_fixed import SysIDFixedPolicy

        policy = SysIDFixedPolicy(robot)

    else:
        raise ValueError("Unknown policy")

    if "real" not in args.sim and hasattr(policy, "time_arr"):
        n_steps: float = round(policy.time_arr[-1] / policy.control_dt) + 1  # type: ignore
    else:
        n_steps = float("inf")

    debug: Dict[str, Any] = {
        "n_steps": n_steps,
        "log": False,
        "plot": True,
        "render": True,
    }

    main(robot, sim, policy, debug)
