from typing import Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.envs.squat_env import SquatCfg
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.ref_motion.squat_ref import SquatReference
from toddlerbot.sim.robot import Robot


class SquatPolicy(MJXPolicy, policy_name="squat"):
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
            SquatCfg(),
            SquatReference(robot),
        )
