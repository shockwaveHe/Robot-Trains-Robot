from typing import Dict, List

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.policies.look_pd import LookPDPolicy
from toddlerbot.policies.reset_pd import ResetPDPolicy
from toddlerbot.policies.squat_pd import SquatPDPolicy
from toddlerbot.policies.turn import TurnPolicy
from toddlerbot.policies.walk import WalkPolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.math_utils import interpolate_action

WALK_CKPT = "20240929_103548"
TURN_CKPT = "20240929_112114"


class TeleopJoystickPolicy(BasePolicy, policy_name="teleop_joystick"):
    def __init__(
        self, name: str, robot: Robot, init_motor_pos: npt.NDArray[np.float32]
    ):
        super().__init__(name, robot, init_motor_pos)

        self.default_motor_pos = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )

        self.joystick = Joystick()

        self.policies: Dict[str, BasePolicy] = {
            "walk": WalkPolicy("walk", robot, init_motor_pos, WALK_CKPT, self.joystick),
            "turn": TurnPolicy("turn", robot, init_motor_pos, TURN_CKPT, self.joystick),
            "squat": SquatPDPolicy("squat", robot, init_motor_pos, self.joystick),
            "look": LookPDPolicy("look", robot, init_motor_pos, self.joystick),
            "reset_pd": ResetPDPolicy("reset_pd", robot, init_motor_pos),
            "balance_pd": PDPolicy("balance_pd", robot, init_motor_pos),
        }

        self.prep_duration = 2.0
        self.prep_time, self.prep_action = self.move(
            -self.control_dt,
            init_motor_pos,
            self.default_motor_pos,
            self.prep_duration,
            end_time=0.0,
        )

        self.is_reset = False
        self.last_policy = "reset_pd"

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return action

        if np.allclose(obs.motor_pos, self.default_motor_pos, atol=0.05):
            self.is_reset = True

        command_scale = {key: 0 for key in self.policies}
        command_scale["reset_pd"] = 1e-6
        control_inputs = self.joystick.get_controller_input()
        for task, input in control_inputs.items():
            for key in self.policies:
                if key in task:
                    command_scale[key] += abs(input)
                    break

        policy_curr = max(command_scale, key=command_scale.get)

        if policy_curr != self.last_policy and not self.is_reset:
            policy_curr = "reset_pd"
            if self.last_policy != "reset_pd":
                self.policies[self.last_policy].reset()

        motor_target = self.policies[policy_curr].step(obs, is_real)

        self.last_policy = policy_curr
        self.is_reset = False

        return motor_target
