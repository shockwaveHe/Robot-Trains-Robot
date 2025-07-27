from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.policies.balance_pd import BalancePDPolicy
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.policies.reset_pd import ResetPDPolicy
from toddlerbot.policies.stand import StandPolicy
from toddlerbot.policies.walk import WalkPolicy
from toddlerbot.sensing.camera import Camera
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.comm_utils import ZMQNode
from toddlerbot.utils.math_utils import interpolate_action


class ArmTreadmillFollowerPolicy(BasePolicy, policy_name="at_follower"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ip: str = "192.168.0.137",
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
        **kwargs: Any,
    ):
        super().__init__(name, robot, init_motor_pos)

        self.zmq_receiver = ZMQNode(type="receiver")
        self.zmq_sender = ZMQNode(type="sender", ip=ip)

        self.camera = None
        self.walk_policy = WalkPolicy(
            "walk", robot, init_motor_pos, joystick=None
        )
        balance_kwargs: Dict[str, Any] = dict(
            joystick=None,
            cameras=self.camera, 
            zmq_receiver=self.zmq_receiver,
            zmq_sender=self.zmq_sender,
            ip=ip,
            fixed_command=fixed_command,
        )
        self.stand_policy = StandPolicy(
            "stand", robot, init_motor_pos
        )
        self.reset_policy = ResetPDPolicy(
            "reset_pd", robot, init_motor_pos, **balance_kwargs
        )
        self.policies = {
            "walk": self.walk_policy,
            "stand": self.stand_policy,
            "reset": self.reset_policy,
        }

        self.need_reset = False
        self.policy_prev = "stand"
        self.last_control_inputs: Dict[str, float] = {}


    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return {}, action

        assert self.zmq_receiver is not None
        msg = self.zmq_receiver.get_msg()

        control_inputs = self.last_control_inputs
        if msg is not None and msg.control_inputs is not None:
            control_inputs = msg.control_inputs

        self.last_control_inputs = control_inputs

        command_scale = {key: 0.0 for key in self.policies}
        command_scale["stand"] = 1e-6

        if len(control_inputs) > 0:
            for task, input in control_inputs.items():
                for key in self.policies:
                    if key in task:
                        command_scale[key] += abs(input)
                        break

        policy_curr = max(command_scale, key=command_scale.get)  # type: ignore
        if policy_curr != self.policy_prev:
            last_policy = self.policies[self.policy_prev]
            is_reset_mask = np.abs((obs.motor_pos - self.default_motor_pos)) < 0.1

            policy_type_differs = isinstance(last_policy, MJXPolicy) != isinstance(
                self.policies[policy_curr], MJXPolicy
            )

            if (
                policy_type_differs
                and not self.need_reset
                and not np.all(is_reset_mask)
                and not isinstance(last_policy, StandPolicy)
            ):
                if isinstance(last_policy, MJXPolicy) and not last_policy.is_standing:
                    # Not ready for switching policy
                    policy_curr = self.policy_prev
                    for k, v in control_inputs.items():
                        control_inputs[k] = 0.0
                else:
                    self.need_reset = True
                    self.reset_policy.is_button_pressed = True

                    last_policy.reset()

        if self.need_reset:
            policy_curr = "reset"

        current_policy = self.policies[policy_curr]
        if isinstance(current_policy, StandPolicy):
            print(f"Stand policy: {control_inputs}")
        elif isinstance(current_policy, BalancePDPolicy):
            current_policy.msg = msg
        elif isinstance(current_policy, MJXPolicy):
            print(f"MJX policy: {control_inputs}")
            current_policy.control_inputs = control_inputs
        else:
            print(f"Policy: {policy_curr}, {current_policy}")
            raise NotImplementedError

        control_inputs, motor_target = current_policy.step(obs, is_real)

        print(f"policy: {policy_curr}")
        print(f"need_reset: {self.need_reset}")

        if self.reset_policy.reset_time is None:
            self.need_reset = False

        self.policy_prev = policy_curr

        return control_inputs, motor_target
