import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.policies.reset_pd import ResetPDPolicy
from toddlerbot.policies.teleop_follower_pd import TeleopFollowerPDPolicy
from toddlerbot.policies.turn import TurnPolicy
from toddlerbot.policies.walk import WalkPolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.math_utils import interpolate_action

WALK_CKPT = "20240929_133322"
TURN_CKPT = "20240929_220646"


class TeleopJoystickPolicy(BasePolicy, policy_name="teleop_joystick"):
    def __init__(
        self, name: str, robot: Robot, init_motor_pos: npt.NDArray[np.float32]
    ):
        super().__init__(name, robot, init_motor_pos)

        self.default_motor_pos = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )
        self.joystick = Joystick()

        self.walk_policy = WalkPolicy(
            "walk", robot, init_motor_pos, WALK_CKPT, self.joystick
        )
        self.turn_policy = TurnPolicy(
            "turn", robot, init_motor_pos, TURN_CKPT, self.joystick
        )
        self.balance_policy = TeleopFollowerPDPolicy(
            "teleop_follower_pd", robot, init_motor_pos, self.joystick
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
        self.last_policy = "balance"

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return action

        # TODO: Finish the current gait cycle before transitioning
        command_scale = {key: 0 for key in self.policies}
        command_scale["balance"] = 1e-6
        control_inputs = self.joystick.get_controller_input()
        for task, input in control_inputs.items():
            for key in self.policies:
                if key in task:
                    command_scale[key] += abs(input)
                    break

        policy_curr = max(command_scale, key=command_scale.get)

        if (
            not self.need_reset
            and policy_curr != self.last_policy
            and self.last_policy != "reset"
            and policy_curr in ["walk", "turn"]
        ):
            self.need_reset = True
            self.reset_policy.last_motor_target = (
                self.balance_policy.last_motor_target.copy()
            )
            self.balance_policy.reset()

        if self.need_reset:
            policy_curr = "reset"

        motor_target = self.policies[policy_curr].step(obs, is_real)

        # print(f"Policy: {policy_curr}")
        # print(f"need_reset: {self.need_reset}")

        if self.reset_policy.reset_time is None:
            self.need_reset = False
            self.reset_policy.reset()

        self.last_policy = policy_curr

        return motor_target
