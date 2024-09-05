from typing import Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.envs.squat_env import SquatCfg
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.sim.robot import Robot


class SquatPolicy(MJXPolicy):
    def __init__(
        self,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ckpt: str,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        env_cfg = SquatCfg()
        command_ranges = [env_cfg.commands.lin_vel_z_range]

        super().__init__(
            "squat", robot, init_motor_pos, ckpt, command_ranges, fixed_command
        )
