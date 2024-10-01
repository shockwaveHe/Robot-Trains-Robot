from typing import Dict, Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.policies.look_pd import LookPDPolicy
from toddlerbot.policies.reset_pd import ResetPDPolicy
from toddlerbot.policies.squat_pd import SquatPDPolicy
from toddlerbot.policies.teleop_follower_pd import TeleopFollowerPDPolicy
from toddlerbot.policies.turn import TurnPolicy
from toddlerbot.policies.walk import WalkPolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick
from toddlerbot.utils.math_utils import interpolate_action

WALK_CKPT = "20240929_133322"
TURN_CKPT = "20240929_220646"


class TeleopJoystickFollowerPDPolicy(
    TeleopFollowerPDPolicy, policy_name="teleop_joystick_follower_pd"
):
    def __init__(
        self, name: str, robot: Robot, init_motor_pos: npt.NDArray[np.float32]
    ):
        super().__init__(name, robot, init_motor_pos)

        self.default_motor_pos = np.array(
            list(robot.default_motor_angles.values()), dtype=np.float32
        )
        self.default_torso_z = robot.config["general"]["offsets"]["default_torso_z"]

        self.joystick = None
        try:
            self.joystick = Joystick()
        except Exception:
            pass

        self.policies: Dict[str, BasePolicy] = {
            "walk": WalkPolicy("walk", robot, init_motor_pos, WALK_CKPT),
            "turn": TurnPolicy("turn", robot, init_motor_pos, TURN_CKPT),
            "squat": SquatPDPolicy("squat_pd", robot, init_motor_pos),
            "look": LookPDPolicy("look_pd", robot, init_motor_pos),
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

    def step(
        self,
        obs: Obs,
        is_real: bool = False,
        control_inputs: Optional[Dict[str, float]] = None,
    ) -> npt.NDArray[np.float32]:
        motor_target = super().step(obs, is_real, control_inputs)

        if self.joystick is not None:
            control_inputs = self.joystick.get_controller_input()
        elif self.msg is not None:
            control_inputs = self.msg.control_inputs
        else:
            return motor_target

        # TODO: Finish the current gait cycle before transitioning
        command_scale = {key: 0 for key in self.policies}
        command_scale["squat"] = 1e-6
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

        motor_target = self.policies[policy_curr].step(obs, is_real, control_inputs)

        # print(f"Policy: {policy_curr}")
        # print(f"is_reset: {self.is_reset}")

        self.last_policy = policy_curr

        return motor_target
