from typing import Dict, Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.locomotion.turn_env import TurnCfg
from toddlerbot.motion.walk_zmp_ref import WalkZMPReference
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick


class TurnPolicy(MJXPolicy, policy_name="turn"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ckpt: str = "",
        joystick: Optional[Joystick] = None,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        env_cfg = TurnCfg()
        motion_ref = WalkZMPReference(
            robot,
            env_cfg.sim.timestep * env_cfg.action.n_frames,
            env_cfg.action.cycle_time,
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
            if task == "turn":
                command[7] = np.interp(
                    input,
                    [-1, 0, 1],
                    [self.command_range[0][1], 0.0, self.command_range[0][0]],
                )

        return command
