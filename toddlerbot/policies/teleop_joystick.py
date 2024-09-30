from typing import Dict

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
        self.default_torso_z = robot.config["general"]["offsets"]["default_torso_z"]

        self.joystick = Joystick()

        self.policies: Dict[str, BasePolicy] = {
            "walk": WalkPolicy("walk", robot, init_motor_pos, WALK_CKPT, self.joystick),
            "turn": TurnPolicy("turn", robot, init_motor_pos, TURN_CKPT, self.joystick),
            "squat": SquatPDPolicy("squat_pd", robot, init_motor_pos, self.joystick),
            "look": LookPDPolicy("look_pd", robot, init_motor_pos, self.joystick),
            "reset": ResetPDPolicy("reset_pd", robot, init_motor_pos),
        }

        self.prep_duration = 2.0
        self.prep_time, self.prep_action = self.move(
            -self.control_dt,
            init_motor_pos,
            self.default_motor_pos,
            self.prep_duration,
            end_time=0.0,
        )

        self.is_reset = True
        self.last_policy = "squat"

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return action

        command_scale = {key: 0 for key in self.policies}
        command_scale["squat"] = 1e-6
        control_inputs = self.joystick.get_controller_input()
        for task, input in control_inputs.items():
            for key in self.policies:
                if key in task:
                    command_scale[key] += abs(input)
                    break

        policy_curr = max(command_scale, key=command_scale.get)

        self.is_reset = (
            self.last_policy != "reset" and policy_curr not in ["walk", "turn"]
        ) or np.allclose(obs.pos[2], self.default_torso_z, atol=1e-2)

        if not self.is_reset:
            policy_curr = "reset"
            self.policies["squat"].reset()

        if policy_curr != self.last_policy and self.last_policy != "squat":
            self.policies[self.last_policy].reset()

        motor_target = self.policies[policy_curr].step(obs, is_real)

        # print(f"Policy: {policy_curr}")
        # print(f"is_reset: {self.is_reset}")

        self.last_policy = policy_curr

        return motor_target
