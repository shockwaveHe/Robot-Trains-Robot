import argparse
import os
import pickle
import time
from typing import Any, Dict, List

import joblib

# import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import zmq
from pynput import keyboard
from tqdm import tqdm

from toddlerbot.policies import BasePolicy
from toddlerbot.sensing.Camera import Camera
from toddlerbot.sensing.FSR import FSR
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.dataset_utils import DatasetLogger
from toddlerbot.utils.math_utils import round_floats
from toddlerbot.utils.misc_utils import (
    dump_profiling_data,
    log,
    precise_sleep,
    # profile,
    snake2camel,
)

default_pose = np.array(
    [
        -0.8590293,
        -0.5307572,
        -0.00153399,
        0.8789711,
        1.1581554,
        -0.8406215,
        -1.0937283,
        0.8805051,
        -0.52615523,
        -0.05062127,
        0.8759031,
        -1.2394564,
        0.79613614,
        -1.1366798,
    ]
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


"""
Replay Policy loads the dataset logged and follows the same trajectory
"""


class ReplayFollowerPolicy(BasePolicy):
    def __init__(self, robot: Robot, log_path: str, replay_dest: str):
        super().__init__(robot)
        self.name = "replay_fixed"

        self.default_action = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )

        self.log_path = log_path
        self.toggle_motor = False
        self.data_dict = joblib.load(log_path)
        self.data_dict["state_array"][:, 0] = (
            self.data_dict["state_array"][:, 0] - self.data_dict["state_array"][0, 0]
        )
        print(self.data_dict.keys())
        self.replay_start = 0

        self.log = False
        self.replay_done = False
        self.blend_percentage = 0.0
        self.default_pose = default_pose
        # Start a listener for the spacebar
        self._start_spacebar_listener()

    def _start_spacebar_listener(self):
        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    # if space is pressed, and is done playing, play next trajectory
                    # print(
                    #     f"\n\nLogging is now {'enabled' if self.log else 'disabled'}.\n\n"
                    # )
                    pass
            except AttributeError:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    def reset_slowly(self, obs_real):
        leader_action = self.default_pose * self.blend_percentage + obs_real.q * (
            1 - self.blend_percentage
        )
        self.blend_percentage += 0.002
        self.blend_percentage = min(1, self.blend_percentage)
        return leader_action

    def step(self, obs: Obs, obs_real: Obs) -> npt.NDArray[np.float32]:
        if self.replay_start < 1:
            self.replay_start = time.time()

        curr_idx = np.argmin(
            np.abs(
                self.data_dict["state_array"][:, 0] - (time.time() - self.replay_start)
            )
        )
        sim_action = self.data_dict["state_array"][curr_idx, 1:15]

        if (time.time() - self.replay_start) > self.data_dict["state_array"][-1, 0]:
            print("Replay done")
            self.replay_done = True

        if self.replay_done:
            leader_action = self.reset_slowly(obs_real)
            if self.blend_percentage >= 0.99:
                self.log = True
                self.toggle_motor = True
        else:
            leader_action = sim_action
        return sim_action, leader_action


class TeleopFollowerPolicy(BasePolicy):
    def __init__(self, robot: Robot):
        super().__init__(robot)
        self.name = "teleop_follower_fixed"

        self.default_action = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )
        self.log = False
        self.toggle_motor = True
        self.blend_percentage = 0.0
        self.nlogs = 1
        print(
            '\n\nBy default, logging is disabled. Press "space" to toggle logging.\n\n'
        )
        self.dataset_logger = DatasetLogger()

        self.follower_camera = Camera(camera_id=0)

        self.default_pose = default_pose

        # start a zmq listener
        self.start_zmq()

        # optional: blend to current pose of leader

    def start_zmq(self):
        # Set up ZeroMQ context and socket for receiving data
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.PULL)
        self.socket.bind("tcp://0.0.0.0:5555")  # Listen on all interfaces
        self.get_zmq_data()

    def get_zmq_data(self):
        try:
            # Non-blocking receive
            serialized_array = self.socket.recv(zmq.NOBLOCK)
            send_dict = pickle.loads(serialized_array)
            return send_dict
        except zmq.Again:
            # No data is available
            print("No message available right now")
            return None

    # note: calibrate zero at: toddlerbot/tools/calibration/calibrate_zero.py --robot toddlerbot_arms
    # note: zero points can be accessed in config_motors.json

    def step(self, obs_real: Obs) -> npt.NDArray[np.float32]:
        remote_state = self.get_zmq_data()
        if remote_state is not None:
            curr_t = time.time()
            if curr_t - remote_state["time"] > 0.1:
                print(
                    "Stale data received, skipping. Time diff: ",
                    (curr_t - remote_state["time"]) * 1000,
                    "ms",
                )
            else:
                print(remote_state)

        # sim_action = remote_state["action"]
        sim_action = obs_real.q

        # Log the data
        if self.log:
            t1 = time.time()
            camera_frame = self.follower_camera.get_state()
            # fsrL, fsrR = self.fsr.get_state()
            fsrL, fsrR = [0, 0]
            t2 = time.time()
            print(f"camera_frame: {t2 - t1:.2f} s, current_time: {obs_real.time}")
            self.dataset_logger.log_entry(
                obs_real.time, obs_real.q, [fsrL, fsrR], camera_frame
            )
        else:
            # clean up the log when not logging.
            self.dataset_logger.maintain_log()

        return sim_action


# @profile()
"""
robot: follower robot
sim: follower world
sim_real: leader world
policy: teleoperation policy to get leader action and apply to follower
"""


def main(
    robot: Robot,
    sim: BaseSim,
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
            obs_time = time.time()

            obs.time -= start_time
            action = policy.step(obs)
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

    except Exception as e:
        log(f"Error: {e}", header=header_name)

    finally:
        if policy.name == "teleop_follower_fixed":
            exp_name = f"{robot.name}_{policy.name}_{sim.name}"
            time_str = time.strftime("%Y%m%d_%H%M%S")
            exp_folder_path = f"results/{exp_name}_{time_str}"

            os.makedirs(exp_folder_path, exist_ok=True)

            if config["render"] and hasattr(sim, "save_recording"):
                assert isinstance(sim, MuJoCoSim)
                sim.save_recording(exp_folder_path, policy.control_dt, 2)

            prof_path = os.path.join(exp_folder_path, "profile_output.lprof")
            dump_profiling_data(prof_path)

            log_data_dict = {
                "obs_list": obs_list,
                "motor_angles_list": motor_angles_list,
            }
            log_data_path = os.path.join(exp_folder_path, "log_data.pkl")
            with open(log_data_path, "wb") as f:
                pickle.dump(log_data_dict, f)

            # save the dataset
            policy.dataset_logger.save(os.path.join(exp_folder_path, "dataset.lz4"))

            # # disable the motors
            # sim.dynamixel_controller.disable_motors()

        sim.close()

        p_bar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Using a second leader upper body for teleoperation"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in robot_descriptions. [toddlerbot_arms, toddlerbot]",
    )
    parser.add_argument(
        "--sim",
        type=str,
        default="real",
        help="The name of the simulator to use.",
    )
    parser.add_argument(
        "--replay_env",
        type=str,
        default="real",
        help="The name of the replay destination to use. [mujoco, real]",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="teleop_follower_fixed",
        help="The name of the task. [replay_follower_fixed, teleop_follower_fixed]",
    )

    args = parser.parse_args()

    # real and sim robot need slightly different configurations
    robot = Robot(args.robot)

    if args.policy == "teleop_follower_fixed":
        policy: BasePolicy = TeleopFollowerPolicy(robot)

    elif args.policy == "replay_follower_fixed":
        # path = "/Users/weizhuo2/Documents/gits/toddleroid/results/toddlerbot_arms_teleop_fixed_mujoco_20240904_202637/dataset.lz4"
        path = "/Users/weizhuo2/Documents/gits/toddleroid/results/toddlerbot_arms_teleop_fixed_mujoco_20240906_175139/dataset.lz4"
        policy: BasePolicy = ReplayFollowerPolicy(
            robot, log_path=path, replay_dest=args.replay_env
        )
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
        "replay_dest": args.replay_env,
    }

    if args.sim == "real":
        from toddlerbot.sim.real_world import RealWorld

        sim = RealWorld(robot)
        sim.has_imu = False

    else:
        raise ValueError("Unknown simulator")

    main(robot, sim, policy, config)
