from typing import Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.envs.rotate_torso_env import RotateTorsoCfg
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.ref_motion.rotate_torso_ref import RotateTorsoReference
from toddlerbot.sim.robot import Robot


class RotateTorsoPolicy(MJXPolicy, policy_name="rotate_torso"):
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
            RotateTorsoCfg(),
            RotateTorsoReference(robot),
        )
