import time

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sensing.camera import Camera
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.comm_utils import ZMQNode
from toddlerbot.utils.dataset_utils import DatasetLogger
from toddlerbot.utils.math_utils import interpolate_action


class TeleopFollowerPolicy(BasePolicy, policy_name="teleop_follower"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
    ):
        super().__init__(name, robot, init_motor_pos)

        self.default_motor_pos = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )
        self.arm_motor_slice = slice(
            robot.motor_ordering.index("left_sho_pitch"),
            robot.motor_ordering.index("right_wrist_roll") + 1,
        )
        self.left_gripper_index = (
            robot.motor_ordering.index("left_gripper_rack")
            if "left_gripper_rack" in robot.motor_ordering
            else None
        )
        self.right_gripper_index = (
            robot.motor_ordering.index("right_gripper_rack")
            if "right_gripper_rack" in robot.motor_ordering
            else None
        )

        self.dataset_logger = DatasetLogger()
        self.zmq = ZMQNode(type="receiver")

        self.camera = None
        try:
            self.camera = Camera(camera_id=0)
        except Exception:
            pass

        self.is_logging = False
        self.toggle_motor = True
        self.n_logs = 1

        # remote variables
        self.remote_is_logging = self.is_logging
        self.remote_action = None
        self.remote_fsr = None

        self.prep_duration = 2.0
        self.prep_time, self.prep_action = self.move(
            -self.control_dt,
            init_motor_pos,
            self.default_motor_pos,
            self.prep_duration,
            end_time=0.0,
        )

        print('\nBy default, logging is disabled. Press "menu" to toggle logging.\n')

    # note: calibrate zero at: toddlerbot/tools/calibrate_zero.py --robot toddlerbot_arms
    # note: zero points can be accessed in config_motors.json
    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        if obs.time < self.prep_time[-1]:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return action

        action = self.default_motor_pos.copy()

        msg = self.zmq.get_msg()
        if msg is not None:
            time_curr = time.time()
            if time_curr - msg.time > 0.1:
                print(
                    "Stale data received, skipping. Time diff: ",
                    (time_curr - msg.time) * 1000,
                    "ms",
                )
            else:
                # print(
                #     "Time diff: ",
                #     (time_curr - remote_state["time"]) * 1000,
                #     "ms",
                # )
                self.remote_is_logging = msg.is_logging
                self.remote_action = msg.action
                self.remote_fsr = msg.fsr

                action[self.arm_motor_slice] = self.remote_action

                if self.left_gripper_index is not None:
                    action[self.left_gripper_index] = (
                        self.remote_fsr[0]
                        / 100.0
                        * self.robot.joint_limits["left_gripper_rack"][1]
                    )

                if self.right_gripper_index is not None:
                    action[self.right_gripper_index] = (
                        self.remote_fsr[1]
                        / 100.0
                        * self.robot.joint_limits["right_gripper_rack"][1]
                    )

                self.is_logging = self.remote_is_logging

        # Log the data
        if self.is_logging:
            t1 = time.time()
            if self.camera is not None:
                camera_frame = self.camera.get_state()
            else:
                camera_frame = None

            fsrL, fsrR = self.remote_fsr[0], self.remote_fsr[1]
            t2 = time.time()
            n_logs = len(self.dataset_logger.data_dict["episode_ends"])
            print(
                f"Logging traj {n_logs} (starts at 0): camera_frame: {t2 - t1:.2f} s, current_time: {obs.time: .2f} s"
            )
            self.dataset_logger.log_entry(
                obs.time, obs.motor_pos, [fsrL, fsrR], camera_frame
            )
        else:
            # clean up the log when not logging.
            self.dataset_logger.maintain_log()

        return action
