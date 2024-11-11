from typing import Dict, Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.locomotion.mjx_config import get_env_config
from toddlerbot.motion.balance_pd_ref import BalancePDReference
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick


class BalancePolicy(MJXPolicy, policy_name="balance"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ckpt: str,
        joystick: Optional[Joystick] = None,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        env_cfg = get_env_config("balance")
        motion_ref = BalancePDReference(
            robot,
            env_cfg.sim.timestep * env_cfg.action.n_frames,
        )

        self.command_range = env_cfg.commands.command_range

        super().__init__(
            name,
            robot,
            init_motor_pos,
            ckpt,
            joystick,
            fixed_command,
            env_cfg,
            motion_ref,
        )

    def get_command(self, control_inputs: Dict[str, float]) -> npt.NDArray[np.float32]:
        command = np.zeros(self.num_commands, dtype=np.float32)
        for task, input in control_inputs.items():
            if task == "look_left" and input > 0:
                command[0] = input * self.command_range[0][1]
            elif task == "look_right" and input > 0:
                command[0] = input * self.command_range[0][0]
            elif task == "look_up" and input > 0:
                command[1] = input * self.command_range[1][1]
            elif task == "look_down" and input > 0:
                command[1] = input * self.command_range[1][0]
            elif task == "lean_left" and input > 0:
                command[3] = input * self.command_range[3][0]
            elif task == "lean_right" and input > 0:
                command[3] = input * self.command_range[3][1]
            elif task == "twist_left" and input > 0:
                command[4] = input * self.command_range[4][0]
            elif task == "twist_right" and input > 0:
                command[4] = input * self.command_range[4][1]
            elif task == "squat":
                command[5] = np.interp(
                    input,
                    [-1, 0, 1],
                    [self.command_range[5][1], 0.0, self.command_range[5][0]],
                )

        return command
