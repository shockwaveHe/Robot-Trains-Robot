# type: ignore

import time

# import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sensing.Camera import Camera
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.comm_utils import ZMQNode
from toddlerbot.utils.dataset_utils import DatasetLogger

default_pose = np.array(
    [
        -0.60745645,
        -0.9265244,
        0.02147579,
        1.2348546,
        0.52922344,
        0.49394178,
        -1.125942,
        0.5123496,
        -0.96180606,
        -0.25003886,
        1.2195148,
        -0.35128164,
        -0.6504078,
        -1.1535536,
    ]
)


class TeleopFollowerPolicy(BasePolicy, policy_name="teleop_follower"):
    def __init__(self, robot: Robot):
        super().__init__(
            name="teleop_follower", robot=robot, init_motor_pos=default_pose
        )
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

        # remote variables
        self.remote_log = self.log
        self.remote_action = None
        self.remote_fsr = None

        # start a zmq listener
        self.zmq = ZMQNode(type="Receiver")

        # optional: blend to current pose of leader

    # note: calibrate zero at: toddlerbot/tools/calibration/calibrate_zero.py --robot toddlerbot_arms
    # note: zero points can be accessed in config_motors.json

    def step(self, obs_real: Obs) -> npt.NDArray[np.float32]:
        tstart = time.time()

        sim_action = self.default_action
        remote_state = self.zmq.get_all_msg()
        if remote_state is not None:
            curr_t = time.time()
            if curr_t - remote_state["time"] > 0.1:
                print(
                    "Stale data received, skipping. Time diff: ",
                    (curr_t - remote_state["time"]) * 1000,
                    "ms",
                )
            else:
                # print(
                #     "Time diff: ",
                #     (curr_t - remote_state["time"]) * 1000,
                #     "ms",
                # )
                self.remote_log, self.remote_action, self.remote_fsr = (
                    remote_state["log"],
                    remote_state["sim_action"],
                    remote_state["fsr"],
                )
                # print(self.remote_action)
                sim_action[16:30] = self.remote_action
                # print(self.robot.joint_limits["left_gripper_rack"][1])
                sim_action[30] = (
                    self.remote_fsr[0]
                    / 100.0
                    * self.robot.joint_limits["left_gripper_rack"][1]
                )
                sim_action[31] = (
                    self.remote_fsr[1]
                    / 100.0
                    * self.robot.joint_limits["right_gripper_rack"][1]
                )
                # print(sim_action[30], sim_action[31])
                self.log = self.remote_log

        # Log the data
        if self.log:
            t1 = time.time()
            camera_frame = self.follower_camera.get_state()
            # camera_frame = None
            # fsrL, fsrR = self.fsr.get_state()
            fsrL, fsrR = self.remote_fsr[0], self.remote_fsr[1]
            t2 = time.time()
            nlogs = len(self.dataset_logger.data_dict["episode_ends"])
            print(
                f"Logging traj {nlogs} (starts at 0): Camera_frame: {t2 - t1:.2f} s, current_time: {obs_real.time}"
            )
            self.dataset_logger.log_entry(
                time.time(), obs_real.motor_pos, [fsrL, fsrR], camera_frame
            )
        else:
            # clean up the log when not logging.
            self.dataset_logger.maintain_log()

        tend = time.time()
        # print(f"Time taken: {1000*(tend - tstart):.2f} ms")

        return sim_action
