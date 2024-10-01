import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.policies.reset_pd import ResetPDPolicy
from toddlerbot.policies.teleop_follower_pd import TeleopFollowerPDPolicy
from toddlerbot.policies.turn import TurnPolicy
from toddlerbot.policies.walk import WalkPolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.comm_utils import ZMQNode
from toddlerbot.utils.math_utils import interpolate_action


class TeleopJoystickPolicy(BasePolicy, policy_name="teleop_joystick"):
    def __init__(
        self, name: str, robot: Robot, init_motor_pos: npt.NDArray[np.float32]
    ):
        super().__init__(name, robot, init_motor_pos)

        self.default_motor_pos = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )

        self.joystick = None
        try:
            self.joystick = Joystick()
        except Exception:
            pass

        self.zmq_node = ZMQNode(type="receiver")

        self.walk_policy = WalkPolicy(
            "walk", robot, init_motor_pos, joystick=self.joystick
        )
        self.turn_policy = TurnPolicy(
            "turn", robot, init_motor_pos, joystick=self.joystick
        )
        self.balance_policy = TeleopFollowerPDPolicy(
            "teleop_follower_pd",
            robot,
            init_motor_pos,
            joystick=self.joystick,
            zmq_node=self.zmq_node,
        )
        self.reset_policy = ResetPDPolicy("reset_pd", robot, init_motor_pos)

        self.policies = {
            "walk": self.walk_policy,
            "turn": self.turn_policy,
            "balance": self.balance_policy,
            "reset": self.reset_policy,
        }

        self.prep_duration = 2.0
        self.prep_time, self.prep_action = self.move(
            -self.control_dt,
            init_motor_pos,
            self.default_motor_pos,
            self.prep_duration,
            end_time=0.0,
        )

        self.need_reset = False
        self.policy_prev = "balance"
        self.last_control_inputs = None

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return action

        msg = self.zmq_node.get_msg()
        self.balance_policy.msg = msg

        print(f"msg: {msg}")

        control_inputs = self.last_control_inputs
        if self.joystick is not None:
            control_inputs = self.joystick.get_controller_input()
        elif msg is not None:
            control_inputs = msg.control_inputs

        self.last_control_inputs = control_inputs

        command_scale = {key: 0 for key in self.policies}
        command_scale["balance"] = 1e-6

        if control_inputs is not None:
            for task, input in control_inputs.items():
                for key in self.policies:
                    if key in task:
                        command_scale[key] += abs(input)
                        break

        policy_curr = max(command_scale, key=command_scale.get)
        if policy_curr != self.policy_prev:
            if (
                not self.need_reset
                and self.policy_prev == "balance"
                and isinstance(self.policies[policy_curr], MJXPolicy)
                and not np.allclose(
                    self.balance_policy.last_motor_target,
                    self.default_motor_pos,
                    atol=0.1,
                )
            ):
                self.need_reset = True
                self.reset_policy.last_motor_target = (
                    self.balance_policy.last_motor_target.copy()
                )
                self.balance_policy.reset()

            last_policy = self.policies[self.policy_prev]
            if (
                isinstance(last_policy, MJXPolicy)
                and not last_policy.is_double_support()
            ):
                policy_curr = self.policy_prev

        if self.need_reset:
            policy_curr = "reset"

        motor_target = self.policies[policy_curr].step(obs, is_real)

        print(f"Policy: {policy_curr}")
        print(f"need_reset: {self.need_reset}")

        if self.reset_policy.reset_time is None:
            self.need_reset = False
            self.reset_policy.reset()

        self.policy_prev = policy_curr

        return motor_target
