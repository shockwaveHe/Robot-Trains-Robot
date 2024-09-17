from typing import Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.envs.balance_env import BalanceCfg
from toddlerbot.policies.mjx_policy import MJXPolicy
from toddlerbot.ref_motion.balance_ref import BalanceReference
from toddlerbot.sim.robot import Robot

default_pose = np.array(
    [
        -0.6028545,
        -0.90198064,
        0.01840782,
        1.2379225,
        0.52615595,
        0.4985056,
        -1.1320779,
        0.5031457,
        -0.9372623,
        -0.248505,
        1.2179809,
        -0.35434943,
        -0.6473398,
        -1.1581556,
    ]
)


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

        teleop_default_motor_pos = self.default_motor_pos.copy()
        arm_motor_slice = slice(
            robot.motor_ordering.index("left_sho_pitch"),
            robot.motor_ordering.index("right_wrist_roll") + 1,
        )
        teleop_default_motor_pos[arm_motor_slice] = default_pose
        self.prep_duration = 7.0
        self.prep_time, self.prep_action = self.move(
            -self.control_dt,
            init_motor_pos,
            teleop_default_motor_pos,
            self.prep_duration,
            end_time=5.0,
        )
