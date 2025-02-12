import os
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.policies.balance_pd import BalancePDPolicy
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.policies.reset_pd import ResetPDPolicy
from toddlerbot.policies.teleop_follower_pd import TeleopFollowerPDPolicy
from toddlerbot.policies.walk import WalkPolicy
from toddlerbot.sensing.camera import Camera
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.comm_utils import ZMQNode
from toddlerbot.policies.dp_policy import DPPolicy
from toddlerbot.policies.push_cart import PushCartPolicy
from toddlerbot.policies.replay import ReplayPolicy

# Run this script on Jetson and run toddlerbot/tools/teleoperate.py on the remote controller
# in the puppeteering mode to control the robot.


class TeleopJoystickPolicy(BasePolicy, policy_name="teleop_joystick"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ip: str,
        run_name: str = "",
    ):
        """Initializes the class with the specified parameters and sets up various components and policies for robot control.

        Args:
            name (str): The name of the instance.
            robot (Robot): The robot object to be controlled.
            init_motor_pos (npt.NDArray[np.float32]): Initial motor positions for the robot.
            ip (str): IP address for the ZMQ communication.
            run_name (str, optional): Name of the run for loading control input data. Defaults to an empty string.

        Raises:
            ValueError: If the specified run_name does not correspond to an existing data file.
        """
        super().__init__(name, robot, init_motor_pos)

        self.control_inputs_list = None
        if len(run_name) > 0:
            pickle_file_path = os.path.join(
                "results",
                f"{robot.name}_teleop_joystick_real_world_{run_name}",
                "log_data.pkl",
            )

            if os.path.exists(pickle_file_path):
                data_dict = joblib.load(pickle_file_path)
            else:
                raise ValueError(f"No data files found in {pickle_file_path}")

            self.control_inputs_list = data_dict["control_inputs_list"]

        self.joystick = None
        try:
            self.joystick = Joystick()
        except Exception:
            pass

        self.zmq_receiver = ZMQNode(type="receiver")
        self.zmq_sender = ZMQNode(type="sender", ip=ip)

        self.left_eye = None
        self.right_eye = None
        try:
            self.left_eye = Camera("left")
            self.right_eye = Camera("right")
        except Exception:
            pass

        self.walk_policy = WalkPolicy(
            "walk", robot, init_motor_pos, joystick=self.joystick
        )
        balance_kwargs: Dict[str, Any] = dict(
            joystick=self.joystick,
            cameras=[self.left_eye, self.right_eye],
            zmq_receiver=self.zmq_receiver,
            zmq_sender=self.zmq_sender,
            ip=ip,
        )
        self.teleop_policy = TeleopFollowerPDPolicy(
            "teleop_follower_pd", robot, init_motor_pos, **balance_kwargs
        )
        self.reset_policy = ResetPDPolicy(
            "reset_pd", robot, init_motor_pos, **balance_kwargs
        )
        if robot.has_gripper:
            self.pick_policy = DPPolicy(
                "pick",
                robot,
                init_motor_pos,
                ckpt="20250110_204554",
                task="pick",
                **balance_kwargs,
            )
            self.hug_policy = self.teleop_policy
            self.push_cart_policy = self.teleop_policy
        else:
            self.pick_policy = self.teleop_policy  # type: ignore
            self.hug_policy = DPPolicy(  # type: ignore
                "hug",
                robot,
                init_motor_pos,
                ckpt="20250109_235450",
                task="hug",
                **balance_kwargs,
            )
            self.push_cart_policy = PushCartPolicy(  # type: ignore
                "push_cart",
                robot,
                init_motor_pos,
                ckpt="20250106_232754",
                **balance_kwargs,
            )
        self.cuddle_policy = ReplayPolicy(
            "cuddle", robot, init_motor_pos, run_name="cuddle"
        )
        self.policies = {
            "walk": self.walk_policy,
            "teleop": self.teleop_policy,
            "reset": self.reset_policy,
            "hug": self.hug_policy,
            "pick": self.pick_policy,
            "push_cart": self.push_cart_policy,
            "cuddle": self.cuddle_policy,
        }

        self.need_reset = False
        self.policy_prev = "teleop"
        self.last_control_inputs: Dict[str, float] = {}

        self.step_curr = 0

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Processes the current observation and determines the appropriate control inputs and motor targets based on the active policy.

        Args:
            obs (Obs): The current observation data.
            is_real (bool, optional): Flag indicating whether the operation is in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing the control inputs and the motor target values.
        """
        assert self.zmq_receiver is not None
        msg = self.zmq_receiver.get_msg()

        control_inputs = self.last_control_inputs
        if self.control_inputs_list is not None:
            step_idx = min(self.step_curr, len(self.control_inputs_list) - 1)
            control_inputs = self.control_inputs_list[step_idx]
        elif msg is not None and msg.control_inputs is not None:
            control_inputs = msg.control_inputs
        elif self.joystick is not None:
            control_inputs = self.joystick.get_controller_input()

        if msg is not None and msg.control_inputs is not None:
            msg.control_inputs = control_inputs

        self.last_control_inputs = control_inputs

        command_scale = {key: 0.0 for key in self.policies}
        command_scale["teleop"] = 1e-6

        if len(control_inputs) > 0:
            for task, input in control_inputs.items():
                for key in self.policies:
                    if key in task:
                        command_scale[key] += abs(input)
                        break

        policy_curr = max(command_scale, key=command_scale.get)  # type: ignore
        if policy_curr != self.policy_prev:
            last_policy = self.policies[self.policy_prev]
            curr_policy = self.policies[policy_curr]

            if (
                isinstance(last_policy, ResetPDPolicy)
                or isinstance(curr_policy, ResetPDPolicy)
                or self.need_reset
            ):
                if isinstance(curr_policy, BalancePDPolicy) or isinstance(
                    curr_policy, ReplayPolicy
                ):
                    curr_policy.time_start = obs.time
                elif isinstance(curr_policy, PushCartPolicy):
                    curr_policy.time_start = obs.time
                    curr_policy.grasp_policy.time_start = obs.time
            else:
                if isinstance(last_policy, MJXPolicy) and not last_policy.is_standing:
                    # Not ready for switching policy
                    policy_curr = self.policy_prev
                    for k, v in control_inputs.items():
                        control_inputs[k] = 0.0
                else:
                    self.need_reset = True
                    self.reset_policy.is_button_pressed = True

                    last_policy.reset()
                    curr_policy.reset()

        if self.need_reset:
            policy_curr = "reset"

        selected_policy = self.policies[policy_curr]

        if isinstance(selected_policy, BalancePDPolicy):
            selected_policy.msg = msg

        elif isinstance(selected_policy, MJXPolicy):
            selected_policy.control_inputs = control_inputs

        elif isinstance(selected_policy, PushCartPolicy):
            selected_policy.walk_policy.control_inputs = control_inputs
            selected_policy.grasp_policy.msg = msg

        _, motor_target = selected_policy.step(obs, is_real)

        print(f"policy: {policy_curr}")
        # print(f"control_inputs: {control_inputs}")
        # print(f"need_reset: {self.need_reset}")

        if self.reset_policy.reset_time is None:
            self.need_reset = False

        self.policy_prev = policy_curr

        self.step_curr += 1

        return control_inputs, motor_target
