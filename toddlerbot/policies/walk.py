from typing import Dict, Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.locomotion.walk_env import WalkCfg
from toddlerbot.motion.walk_zmp_ref import WalkZMPReference
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick


class WalkPolicy(MJXPolicy, policy_name="walk"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ckpt: str = "",
        joystick: Optional[Joystick] = None,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        env_cfg = WalkCfg()
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
        command = 0.5 * np.ones(self.num_commands, dtype=np.float32)
        command[7] = 0.0
        for task, input in control_inputs.items():
            axis = None
            if task == "walk_vertical":
                axis = 5
            elif task == "walk_horizontal":
                axis = 6

            if axis is not None:
                command[axis] = np.interp(
                    input,
                    [-1, 0, 1],
                    [self.command_range[axis][1], 0.0, self.command_range[axis][0]],
                )

        return command
