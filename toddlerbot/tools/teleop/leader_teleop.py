import argparse
import os
import pickle
import time
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from toddlerbot.actuation.dynamixel.dynamixel_control import (
    DynamixelConfig,
    DynamixelController,
)
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


def state_dict_to_action(state_dict):
    action = np.zeros(14)
    for i, key in enumerate(state_dict.keys()):
        action[i] = state_dict[key].pos
    return action


def state_dict_to_np(state_dict):
    np_array = np.zeros((len(state_dict), 4))
    for i, key in enumerate(state_dict.keys()):
        np_array[i, :] = np.array(
            [key, state_dict[key].time, state_dict[key].pos, state_dict[key].vel]
        )
    return np_array


class TeleopPolicy(BasePolicy):
    def __init__(self, robot: Robot):
        super().__init__(robot)
        self.name = "teleop_fixed"

        self.default_action = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )

    # note: calibrate zero at: toddlerbot/tools/calibration/calibrate_zero.py --robot toddlerbot_arms
    # note: zero points can be accessed in config_motors.json

    def step(self, obs: Obs, obs_real: Obs) -> npt.NDArray[np.float32]:
        sim_action = obs_real.q
        # state_dict = self.controller.get_motor_state()
        # print(np.array(list(state_dict.values())))
        # action = state_dict_to_action(state_dict)
        # print(self.default_action)
        # return self.default_action
        return sim_action


# @profile()
def main(
    robot: Robot,
    sim: BaseSim,
    sim_real: BaseSim,
    policy: BasePolicy,
    config: Dict[str, Any],
):
    header_name = snake2camel(sim.name)

    loop_time_list: List[List[float]] = []
    obs_list: List[Obs] = []
    motor_angles_list: List[Dict[str, float]] = []

    start_time = time.time()
    step_idx = 0
    p_bar = tqdm(total=n_steps, desc="Running the policy")
    try:
        while step_idx < n_steps:
            step_start = time.time()

            # Get the latest state from the queue
            obs = sim.get_observation()
            obs_real = sim_real.get_observation()
            obs_time = time.time()

            obs.time -= start_time
            action = policy.step(obs, obs_real)
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

            if config["log"]:
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
            if time_until_next_step > 0:
                precise_sleep(time_until_next_step)

    except KeyboardInterrupt:
        log("KeyboardInterrupt recieved. Closing...", header=header_name)

    finally:
        exp_name = f"{robot.name}_{policy.name}_{sim.name}"
        time_str = time.strftime("%Y%m%d_%H%M%S")
        exp_folder_path = f"results/{exp_name}_{time_str}"

        os.makedirs(exp_folder_path, exist_ok=True)

        if config["render"] and hasattr(sim, "save_recording"):
            assert isinstance(sim, MuJoCoSim)
            sim.save_recording(exp_folder_path, policy.control_dt, 2)

        sim.close()

        p_bar.close()

        prof_path = os.path.join(exp_folder_path, "profile_output.lprof")
        dump_profiling_data(prof_path)

        log_data_dict = {
            "obs_list": obs_list,
            "motor_angles_list": motor_angles_list,
        }
        log_data_path = os.path.join(exp_folder_path, "log_data.pkl")
        with open(log_data_path, "wb") as f:
            pickle.dump(log_data_dict, f)

        if config["plot"]:
            log("Visualizing...", header="Walking")
            # plot_results(
            #     loop_time_list,
            #     obs_list,
            #     motor_angles_list,
            #     policy.control_dt,
            #     exp_folder_path,
            # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Using a second leader upper body for teleoperation"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot_arms",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    parser.add_argument(
        "--sim",
        type=str,
        default="mujoco",
        help="The name of the simulator to use.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="teleop_fixed",
        help="The name of the task.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="The policy checkpoint to load.",
    )
    args = parser.parse_args()

    # real and sim robot need slightly different configurations
    robot_sim = Robot(args.robot)
    robot_real = Robot(args.robot)
    for joint in list(robot_real.config["joints"].keys()):
        robot_real.config["joints"][joint]["kp_real"] = 0.0
        robot_real.config["joints"][joint]["kd_real"] = 0.0
        robot_real.config["joints"][joint]["kff1_real"] = 0.0
        robot_real.config["joints"][joint]["kff2_real"] = 0.0

    if args.policy == "teleop_fixed":
        # from toddlerbot.policies.stand import StandPolicy
        policy: BasePolicy = TeleopPolicy(robot_real)

    else:
        raise ValueError("Unknown policy")

    if "real" not in args.sim and hasattr(policy, "time_arr"):
        n_steps: float = round(policy.time_arr[-1] / policy.control_dt) + 1  # type: ignore
    else:
        n_steps = float("inf")

    config: Dict[str, Any] = {
        "n_steps": n_steps,
        "log": False,
        "plot": True,
        "render": True,
    }

    if args.sim == "mujoco":
        from toddlerbot.sim.mujoco_sim import MuJoCoSim
        from toddlerbot.sim.real_world import RealWorld

        sim = MuJoCoSim(robot_sim, vis_type="view", fixed_base="fixed" in args.policy)
        sim_real = RealWorld(robot_real)
        sim_real.has_imu = False
        sim_real.dynamixel_controller.disable_motors([18, 20, 21, 22, 25, 27, 28, 29])

    else:
        raise ValueError("Unknown simulator")

    main(robot_sim, sim, sim_real, policy, config)
