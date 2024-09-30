import numpy as np
import numpy.typing as npt

from toddlerbot.policies.balance_pd import BalancePDPolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate_action


class ResetPDPolicy(BalancePDPolicy, policy_name="reset_pd"):
    def __init__(
        self, name: str, robot: Robot, init_motor_pos: npt.NDArray[np.float32]
    ):
        super().__init__(name, robot, init_motor_pos)

        self.reset_duration = 7.0
        self.reset_end_time = 2.0
        self.reset_time = None

    def get_joint_target(self, obs: Obs, time_curr: float) -> npt.NDArray[np.float32]:
        if self.reset_time is None:
            self.reset_time, self.reset_action = self.move(
                obs.time - self.control_dt,
                obs.motor_pos,
                self.default_motor_pos,
                self.reset_duration,
                end_time=self.reset_end_time,
            )

        joint_target = self.default_joint_pos.copy()
        if obs.time < self.reset_time[-1]:
            motor_target = np.asarray(
                interpolate_action(obs.time, self.reset_time, self.reset_action)
            )
            joint_angles = self.robot.motor_to_joint_angles(
                dict(zip(self.robot.motor_ordering, motor_target))
            )
            joint_target = np.array(list(joint_angles.values()), dtype=np.float32)
        else:
            self.reset_time = None

        return joint_target
