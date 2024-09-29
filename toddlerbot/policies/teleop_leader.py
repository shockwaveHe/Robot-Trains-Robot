import time

import numpy as np
import numpy.typing as npt
from pynput import keyboard

from toddlerbot.policies import BasePolicy
from toddlerbot.sensing.FSR import FSR
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.comm_utils import ZMQMessage, ZMQNode
from toddlerbot.utils.dataset_utils import DatasetLogger


class TeleopLeaderPolicy(BasePolicy, policy_name="teleop_leader"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ip: str = "127.0.0.1",
    ):
        super().__init__(name, robot, init_motor_pos)

        self.default_motor_pos = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )

        self.dataset_logger = DatasetLogger()
        self.zmq = ZMQNode(type="sender", ip=ip)  # test locally

        self.fsr = None
        try:
            self.fsr = FSR()
        except Exception:
            pass

        self.is_logging = False
        self.toggle_motor = True
        self.n_logs = 1
        self.trial_idx = 0
        self.last_time = time.time()

        # Start a listener for the spacebar
        self._start_keyboard_listener()

        print(
            '\n\nBy default, logging is disabled. Press "space" to toggle logging.\n\n'
        )

    def _start_keyboard_listener(self):
        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    self.is_logging = not self.is_logging
                    self.toggle_motor = True
                    # if logging is toggled to off(done), log the episode end
                    if not self.is_logging:
                        self.dataset_logger.log_episode_end()
                        print(f"Logged {self.n_logs} entries.")
                        self.n_logs += 1
                    print(
                        f"\n\nLogging is now {'enabled' if self.is_logging else 'disabled'}.\n\n"
                    )
            except AttributeError:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    # note: calibrate zero at: toddlerbot/tools/calibration/calibrate_zero.py --robot toddlerbot_arms
    # note: zero points can be accessed in config_motors.json

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        action = obs.motor_pos
        if self.fsr is not None:
            fsrL, fsrR = self.fsr.get_state()
        else:
            fsrL, fsrR = 0.0, 0.0

        # compile data to send to follower
        msg = ZMQMessage(
            time=time.time(),
            is_logging=self.is_logging,
            action=action,
            fsr=np.array([fsrL, fsrR]),
            trial=self.trial_idx,
        )
        # print(f"Sending: {send_dict['time']}")
        self.zmq.send_msg(msg)
        self.trial_idx += 1
        print(time.time(), self.trial_idx)

        # Log the data
        if self.is_logging:
            self.dataset_logger.log_entry(obs.time, action, [fsrL, fsrR], None)
        else:
            # clean up the log when not logging.
            self.dataset_logger.maintain_log()
            # time.sleep(0.1)

        # time_curr = time.time()
        # print(f"Loop time: {1000 * (time_curr - self.last_time):.2f} ms")
        # self.last_time = time.time()

        return action
