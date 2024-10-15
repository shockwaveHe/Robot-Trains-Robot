from typing import Dict, Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.envs.squat_env import SquatCfg
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.ref_motion.squat_ref import SquatReference
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.joystick import Joystick


class SquatPolicy(MJXPolicy, policy_name="squat"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ckpt: str = "",
        joystick: Optional[Joystick] = None,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        env_cfg = SquatCfg()
        motion_ref = SquatReference(
            robot, env_cfg.sim.timestep * env_cfg.action.n_frames
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
        for task, input in control_inputs.items():
            if task == "squat":
                command[5] = np.interp(
                    input,
                    [-1, 0, 1],
                    [self.command_range[5][1], 0.0, self.command_range[5][0]],
                )

        return command
