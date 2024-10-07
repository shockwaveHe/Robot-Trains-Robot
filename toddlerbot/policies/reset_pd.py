import numpy as np
import numpy.typing as npt

from toddlerbot.policies.balance_pd import BalancePDPolicy
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import interpolate_action


class ResetPDPolicy(BalancePDPolicy, policy_name="reset_pd"):
    def __init__(
        self, name: str, robot: Robot, init_motor_pos: npt.NDArray[np.float32]
    ):
        super().__init__(name, robot, init_motor_pos)

        self.reset_duration = 5.0
        self.reset_end_time = 1.0
        self.reset_time = None

    def plan(self) -> npt.NDArray[np.float32]:
        if self.reset_time is None:
            self.reset_time, self.reset_action = self.move(
                -self.control_dt,
                self.last_motor_target,
                self.default_motor_pos,
                self.reset_duration,
                end_time=self.reset_end_time,
            )

        time_curr = self.step_curr * self.control_dt
        if time_curr < self.reset_time[-1]:
            motor_target = np.asarray(
                interpolate_action(time_curr, self.reset_time, self.reset_action)
            )
            joint_target = np.array(
                list(
                    self.robot.motor_to_joint_angles(
                        dict(zip(self.robot.motor_ordering, motor_target))
                    ).values()
                ),
                dtype=np.float32,
            )
        else:
            self.reset_time = None
            joint_target = self.default_joint_pos.copy()

        return joint_target
