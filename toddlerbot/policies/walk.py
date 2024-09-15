from typing import Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.envs.walk_env import WalkCfg
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.ref_motion.walk_zmp_ref import WalkZMPReference
from toddlerbot.sim.robot import Robot


class WalkPolicy(MJXPolicy, policy_name="walk"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ckpt: str,
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        env_cfg = WalkCfg()
        command_ranges = [
            env_cfg.commands.lin_vel_x_range,
            env_cfg.commands.lin_vel_y_range,
            env_cfg.commands.ang_vel_z_range,
        ]
        motion_ref = WalkZMPReference(robot, env_cfg.action.cycle_time, command_ranges)

        super().__init__(
            name, robot, init_motor_pos, ckpt, fixed_command, env_cfg, motion_ref
        )
