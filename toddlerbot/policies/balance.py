from typing import Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.envs.balance_env import BalanceCfg
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.ref_motion.balance_ref import BalanceReference
from toddlerbot.sim.robot import Robot


class BalancePolicy(MJXPolicy, policy_name="balance"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ckpt: str,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        super().__init__(
            name,
            robot,
            init_motor_pos,
            ckpt,
            fixed_command,
            BalanceCfg(),
            BalanceReference(robot),
        )
