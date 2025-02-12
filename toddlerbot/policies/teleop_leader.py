import os
import time
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sensing.FSR import FSR
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.comm_utils import ZMQMessage, ZMQNode
from toddlerbot.utils.math_utils import interpolate_action

# Connect the leader arms to the remote controller and run this script on the remote controller
# during the teleoperation data collection session.


class TeleopLeaderPolicy(BasePolicy, policy_name="teleop_leader"):
    """Teleoperation leader policy for the leader arms of ToddlerBot."""

    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        joystick: Optional[Joystick] = None,
        ip: str = "",
        task: str = "",
    ):
        """Initializes the object with specified parameters, setting up the robot's motor positions, joystick, and task-specific configurations.

        Args:
            name (str): The name of the object.
            robot (Robot): The robot instance to be controlled.
            init_motor_pos (npt.NDArray[np.float32]): Initial motor positions for the robot.
            joystick (Optional[Joystick]): An optional joystick for manual control. Defaults to None.
            ip (str): The IP address for the ZMQNode. Defaults to an empty string.
            task (str): The task to be performed, affecting motor position setup. Defaults to an empty string.

        Raises:
            ValueError: If the task-specific motion file does not exist.
        """
        super().__init__(name, robot, init_motor_pos)

        self.zmq = ZMQNode(type="sender", ip=ip)

        self.fsr = None
        try:
            self.fsr = FSR()
        except Exception as e:
            print(e)

        self.is_running = False
        self.toggle_motor = True
        self.is_button_pressed = False

        if joystick is None:
            self.joystick = Joystick()
        else:
            self.joystick = joystick

        self.reset_duration = 5.0
        self.reset_end_time = 1.0
        self.reset_time = None

        self.left_sho_pitch_idx = robot.motor_ordering.index("left_sho_pitch")
        self.right_sho_pitch_idx = robot.motor_ordering.index("right_sho_pitch")

        self.is_prepared = False

        if len(task) > 0:
            self.manip_duration = 2.0

            prep = "kneel" if task == "pick" else "hold"

            motion_file_path = os.path.join("motion", f"{prep}.pkl")
            if os.path.exists(motion_file_path):
                data_dict = joblib.load(motion_file_path)
            else:
                raise ValueError(f"No data files found in {motion_file_path}")

            self.manip_motor_pos = np.array(data_dict["action_traj"], dtype=np.float32)[
                -1
            ][16:30]  # hard coded for toddlerbot_arms

            if task == "hug":
                self.manip_motor_pos[self.left_sho_pitch_idx] -= 0.2
                self.manip_motor_pos[self.right_sho_pitch_idx] += 0.2
        else:
            self.manip_motor_pos = self.default_motor_pos.copy()

    # note: calibrate zero at: toddlerbot/tools/calibration/calibrate_zero.py --robot toddlerbot_arms
    # note: zero points can be accessed in config_motors.json

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Processes the current observation and determines the appropriate action and control inputs.

        Args:
            obs (Obs): The current observation containing time and motor positions.
            is_real (bool, optional): Flag indicating if the operation is in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing the control inputs and the action to be taken.
        """
        if not self.is_prepared:
            self.is_prepared = True
            self.prep_duration = 2.0
            self.prep_time, self.prep_action = self.move(
                -self.control_dt,
                self.init_motor_pos,
                self.manip_motor_pos,
                self.prep_duration,
            )

        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return {}, action

        control_inputs = self.joystick.get_controller_input()
        for task, input in control_inputs.items():
            if task == "teleop":
                if abs(input) > 0.5:
                    # Button is pressed
                    if not self.is_button_pressed:
                        self.is_button_pressed = True  # Mark the button as pressed
                        self.is_running = not self.is_running  # Toggle logging
                        self.toggle_motor = True

                        print(
                            f"\nLogging is now {'enabled' if self.is_running else 'disabled'}.\n"
                        )
                else:
                    # Button is released
                    self.is_button_pressed = False  # Reset button pressed state

        fsrL, fsrR = 0.0, 0.0
        action = self.manip_motor_pos.copy()
        if self.is_running:
            action = obs.motor_pos
            if self.fsr is not None:
                try:
                    fsrL, fsrR = self.fsr.get_state()
                    # print(f"FSR: {fsrL:.2f}, {fsrR:.2f}")
                except Exception as e:
                    print(e)
        else:
            if self.is_button_pressed and self.reset_time is None:
                self.reset_time, self.reset_action = self.move(
                    obs.time - self.control_dt,
                    obs.motor_pos,
                    self.manip_motor_pos,
                    self.reset_duration,
                    end_time=self.reset_end_time,
                )

            if self.reset_time is not None:
                if obs.time < self.reset_time[-1]:
                    action = np.asarray(
                        interpolate_action(obs.time, self.reset_time, self.reset_action)
                    )
                else:
                    self.reset_time = None

        # compile data to send to follower
        msg = ZMQMessage(
            time=time.time(),
            control_inputs=control_inputs,
            action=action,
            fsr=np.array([fsrL, fsrR]),
        )
        # print(f"Sending: {msg}")
        self.zmq.send_msg(msg)

        # time_curr = time.time()
        # print(f"Loop time: {1000 * (time_curr - self.last_time):.2f} ms")
        # self.last_time = time.time()

        return control_inputs, action
