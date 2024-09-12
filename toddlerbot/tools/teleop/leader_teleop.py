import argparse
import os
import pickle
import threading
import time
from typing import Any, Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import redis
import zmq
from pynput import keyboard
from tqdm import tqdm

from toddlerbot.actuation.dynamixel.dynamixel_control import (
    DynamixelConfig,
    DynamixelController,
)
from toddlerbot.policies import BasePolicy
from toddlerbot.sensing.Camera import Camera
from toddlerbot.sensing.FSR import FSR
from toddlerbot.sim import BaseSim, Obs
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


class ReplayPolicy(BasePolicy):
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
        self.fsr = FSR()

        self.default_pose = default_pose

        # start a zmq listener
        self.start_zmq()

    def start_zmq(self):
        # Set up ZeroMQ context and socket for receiving data
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.PULL)
        self.socket.bind("tcp://0.0.0.0:5555")  # Listen on all interfaces

    def get_zmq_data(self):
        # Receive the serialized numpy array
        serialized_array = self.socket.recv()
        send_dict = pickle.loads(serialized_array)
        return send_dict

    # note: calibrate zero at: toddlerbot/tools/calibration/calibrate_zero.py --robot toddlerbot_arms
    # note: zero points can be accessed in config_motors.json

    def step(self, obs: Obs, obs_real: Obs) -> npt.NDArray[np.float32]:
        sim_action = obs_real.q
        # state_dict = self.controller.get_motor_state()
        # print(np.array(list(state_dict.values())))
        # action = state_dict_to_action(state_dict)
        # print(self.default_action)
        # return self.default_action

        # Log the data
        if self.log:
            t1 = time.time()
            camera_frame = self.follower_camera.get_state()
            fsrL, fsrR = self.fsr.get_state()
            t2 = time.time()
            print(f"camera_frame: {t2 - t1:.2f} s, current_time: {obs_real.time}")
            self.dataset_logger.log_entry(
                obs_real.time, obs_real.q, [fsrL, fsrR], camera_frame
            )
        else:
            # clean up the log when not logging.
            self.dataset_logger.maintain_log()

        leader_action = self.default_pose * self.blend_percentage + obs_real.q * (
            1 - self.blend_percentage
        )
        self.blend_percentage += 0.002
        self.blend_percentage = min(1, self.blend_percentage)
        return sim_action, leader_action


class TeleopPolicy(BasePolicy):
    def __init__(self, robot: Robot):
        super().__init__(robot)
        self.name = "teleop_fixed"

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
        self.fsr = FSR()

        self.default_pose = np.array(
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

        self.start_zmq()

        # Start a listener for the spacebar
        self._start_spacebar_listener()

    def start_zmq(self):
        # Set up ZeroMQ context and socket for receiving data
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.PUSH)
        # Set high water mark and enable non-blocking send
        self.socket.setsockopt(zmq.SNDHWM, 10)  # Limit queue to 10 messages
        self.socket.setsockopt(
            zmq.IMMEDIATE, 1
        )  # Prevent blocking if receiver is not available
        self.socket.connect("tcp://10.5.6.212:5555")

    def send_msg(self, send_dict):
        # Serialize the numpy array using pickle
        serialized_array = pickle.dumps(send_dict)
        # Send the serialized data
        try:
            # Send the serialized data with non-blocking to avoid hanging if the queue is full
            self.socket.send(serialized_array, zmq.NOBLOCK)
            # print("Message sent!")
        except zmq.Again:
            pass

    def _start_spacebar_listener(self):
        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    self.log = not self.log
                    self.toggle_motor = True
                    self.blend_percentage = 0.0
                    # if logging is toggled to off(done), log the episode end
                    if not self.log:
                        self.dataset_logger.log_episode_end()
                        print(f"Logged {self.nlogs} entries.")
                        self.nlogs += 1
                    print(
                        f"\n\nLogging is now {'enabled' if self.log else 'disabled'}.\n\n"
                    )
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

    # note: calibrate zero at: toddlerbot/tools/calibration/calibrate_zero.py --robot toddlerbot_arms
    # note: zero points can be accessed in config_motors.json

    def step(self, obs: Obs, obs_real: Obs) -> npt.NDArray[np.float32]:
        sim_action = obs_real.q
        # state_dict = self.controller.get_motor_state()
        # print(np.array(list(state_dict.values())))
        # action = state_dict_to_action(state_dict)
        # print(self.default_action)
        # return self.default_action

        # compile data to send to follower
        send_dict = {"time": time.time(), "log": self.log, "sim_action": sim_action}
        self.send_msg(send_dict)

        # Log the data
        if self.log:
            t1 = time.time()
            camera_frame = self.follower_camera.get_state()
            fsrL, fsrR = self.fsr.get_state()
            t2 = time.time()
            print(f"camera_frame: {t2 - t1:.2f} s, current_time: {obs_real.time}")
            self.dataset_logger.log_entry(
                obs_real.time, obs_real.q, [fsrL, fsrR], camera_frame
            )
        else:
            # clean up the log when not logging.
            self.dataset_logger.maintain_log()

        leader_action = self.reset_slowly(obs_real)
        return sim_action, leader_action


# @profile()
"""
robot: follower robot
sim: follower world
sim_real: leader world
policy: teleoperation policy to get leader action and apply to follower
"""


def main(
    robot: Robot,
    robot_leader: Robot,
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
            action, action_leader = policy.step(obs, obs_real)
            inference_time = time.time()

            motor_angles: Dict[str, float] = {}
            for motor_name, act in zip(robot.motor_ordering, action):
                motor_angles[motor_name] = act

            motor_angles_leader: Dict[str, float] = {}
            for motor_name, act in zip(robot_leader.motor_ordering, action_leader):
                motor_angles_leader[motor_name] = act

            # need to enable and disable motors according to logging state
            if policy.toggle_motor:
                sim_real.dynamixel_controller.enable_motors()
                if policy.log:
                    # set motor kp kd
                    sim_real.dynamixel_controller.set_kp_kd(0, 0)
                    # when logging, only enable damping for part of the motors
                    sim_real.dynamixel_controller.disable_motors(
                        [18, 20, 21, 22, 25, 27, 28, 29]
                    )
                    print("Disabling motors")
                else:
                    # when not logging, enable all motors, with positive kp kd
                    sim_real.dynamixel_controller.set_kp_kd(2000, 8000)
                    print("Enabling motors")
                policy.toggle_motor = False

            if not policy.log:
                sim_real.set_motor_angles(motor_angles_leader)

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
        if policy.name == "teleop_fixed":
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

            # disable the motors
            sim_real.dynamixel_controller.disable_motors()

        sim.close()

        p_bar.close()


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
        "--sim_follower",
        type=str,
        default="mujoco",
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
        default="teleop_fixed",
        help="The name of the task. [replay_fixed, teleop_fixed]",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="The policy checkpoint to load.",
    )
    args = parser.parse_args()

    # real and sim robot need slightly different configurations
    robot_follower = Robot(args.robot)
    robot_leader = Robot(args.robot)

    if args.policy == "teleop_fixed":
        # for teleop policy, initial kp kd are both zero. should be passive.
        for joint in list(robot_leader.config["joints"].keys()):
            robot_leader.config["joints"][joint]["kp_real"] = 0.0
            robot_leader.config["joints"][joint]["kd_real"] = 0.0
            robot_leader.config["joints"][joint]["kff1_real"] = 0.0
            robot_leader.config["joints"][joint]["kff2_real"] = 0.0
        # from toddlerbot.policies.stand import StandPolicy
        policy: BasePolicy = TeleopPolicy(robot_leader)
    elif args.policy == "replay_fixed":
        # path = "/Users/weizhuo2/Documents/gits/toddleroid/results/toddlerbot_arms_teleop_fixed_mujoco_20240904_202637/dataset.lz4"
        path = "/Users/weizhuo2/Documents/gits/toddleroid/results/toddlerbot_arms_teleop_fixed_mujoco_20240906_175139/dataset.lz4"
        policy: BasePolicy = ReplayPolicy(
            robot_leader, log_path=path, replay_dest=args.replay_env
        )
    else:
        raise ValueError("Unknown policy")

    if "real" not in args.sim_follower and hasattr(policy, "time_arr"):
        n_steps: float = round(policy.time_arr[-1] / policy.control_dt) + 1
    else:
        n_steps = float("inf")

    config: Dict[str, Any] = {
        "n_steps": n_steps,
        "log": False,
        "plot": True,
        "render": True,
        "replay_dest": args.replay_env,
    }

    if args.sim_follower == "mujoco":
        from toddlerbot.sim.mujoco_sim import MuJoCoSim
        from toddlerbot.sim.real_world import RealWorld

        sim_follower = MuJoCoSim(
            robot_follower, vis_type="view", fixed_base="fixed" in args.policy
        )
        sim_leader = RealWorld(robot_leader)
        sim_leader.has_imu = False
        if policy.name == "teleop_fixed":
            sim_leader.dynamixel_controller.disable_motors(
                [18, 20, 21, 22, 25, 27, 28, 29]
            )

    else:
        raise ValueError("Unknown simulator")

    main(robot_follower, robot_leader, sim_follower, sim_leader, policy, config)
