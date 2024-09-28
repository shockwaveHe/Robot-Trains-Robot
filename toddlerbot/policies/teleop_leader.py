# type: ignore
import time

import numpy as np
import numpy.typing as npt
from pynput import keyboard

from toddlerbot.policies import BasePolicy
from toddlerbot.sensing.Camera import Camera
from toddlerbot.sensing.FSR import FSR
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.comm_utils import ZMQNode
from toddlerbot.utils.dataset_utils import DatasetLogger


class TeleopLeaderPolicy(BasePolicy, policy_name="teleop_leader"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
    ):
        super().__init__(name, robot, init_motor_pos)

        # self.default_action = np.array(
        #     list(robot.default_motor_angles.values()), dtype=np.float32
        # )
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
        self.last_t = time.time()

        self.zmq = ZMQNode(type="Sender", ip="10.5.6.171")
        # self.zmq = ZMQNode(type="Sender", ip="127.0.0.1") # test locally
        self.test_idx = 0

        # Start a listener for the spacebar
        self._start_keyboard_listener()

    def _start_keyboard_listener(self):
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
        leader_action = (
            self.default_pose * self.blend_percentage
            + obs_real.motor_pos * (1 - self.blend_percentage)
        )
        self.blend_percentage += 0.002
        self.blend_percentage = min(1, self.blend_percentage)
        return leader_action

    # note: calibrate zero at: toddlerbot/tools/calibration/calibrate_zero.py --robot toddlerbot_arms
    # note: zero points can be accessed in config_motors.json

    def step(self, obs: Obs, obs_real: Obs) -> npt.NDArray[np.float32]:
        tstart = time.time()
        # print("f")
        sim_action = obs_real.motor_pos
        # state_dict = self.controller.get_motor_state()
        # print(np.array(list(state_dict.values())))
        # action = state_dict_to_action(state_dict)
        # print(self.default_action)
        # return self.default_action

        fsrL, fsrR = self.fsr.get_state()

        # compile data to send to follower
        send_dict = {
            "time": time.time(),
            "log": self.log,
            "sim_action": sim_action,
            "fsr": np.array([fsrL, fsrR]),
            "test": self.test_idx,
        }
        # print(f"Sending: {send_dict['time']}")
        self.zmq.send_msg(send_dict)
        self.test_idx += 1
        print(time.time(), self.test_idx)

        # Log the data
        if self.log:
            t1 = time.time()
            # camera_frame = self.follower_camera.get_state()
            # camera_frame = None
            t2 = time.time()
            print(
                f"Logging traj {self.nlogs}: camera_frame: {t2 - t1:.2f} s, current_time: {obs_real.time}"
            )
            self.dataset_logger.log_entry(
                obs_real.time, obs_real.motor_pos, [fsrL, fsrR], None
            )
        else:
            # clean up the log when not logging.
            self.dataset_logger.maintain_log()
            # time.sleep(0.1)

        leader_action = self.reset_slowly(obs_real)

        tend = time.time()
        # print(f"Loop time: {1000*(tend - self.last_t):.2f} ms")
        self.last_t = tend

        # print(sim_action[0], leader_action[0], obs_real.motor_pos[0])
        # print(f"Total time: {1000*(tend - tstart):.2f} ms")
        return sim_action, leader_action
