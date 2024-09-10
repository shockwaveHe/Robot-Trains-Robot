from typing import Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.envs.balance_env import BalanceCfg
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.ref_motion.balance_ref import BalanceReference
from toddlerbot.sim.robot import Robot


class BalancePolicy(MJXPolicy):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ckpt: str,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        env_cfg = BalanceCfg()
        command_ranges = [[0.0, 1.0]]
        motion_ref = BalanceReference(robot)

        super().__init__(
            name,
            robot,
            init_motor_pos,
            env_cfg,
            motion_ref,
            ckpt,
            command_ranges,
            fixed_command,
        )
