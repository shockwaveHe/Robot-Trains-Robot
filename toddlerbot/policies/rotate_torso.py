from typing import Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.envs.rotate_torso_env import RotateTorsoCfg
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.ref_motion.rotate_torso_ref import RotateTorsoReference
from toddlerbot.sim.robot import Robot


class RotateTorsoPolicy(MJXPolicy):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ckpt: str,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        env_cfg = RotateTorsoCfg()
        command_ranges = [
            env_cfg.commands.ang_vel_x_range,
            env_cfg.commands.ang_vel_z_range,
        ]
        motion_ref = RotateTorsoReference(robot)

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
