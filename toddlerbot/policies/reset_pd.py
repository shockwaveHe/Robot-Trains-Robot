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

        self.reset_duration = 5.0
        self.reset_end_time = 1.0
        self.reset_time = None

    def reset(self):
        super().reset()
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
            joint_target = np.array(
                list(
                    self.robot.motor_to_joint_angles(
                        dict(zip(self.robot.motor_ordering, motor_target))
                    ).values()
                ),
                dtype=np.float32,
            )

        joint_angles = self.robot.motor_to_joint_angles(
            dict(zip(self.robot.motor_ordering, obs.motor_pos))
        )
        joint_target[self.neck_yaw_idx] = joint_angles["neck_yaw_driven"]
        joint_target[self.neck_pitch_idx] = joint_angles["neck_pitch_driven"]

        return joint_target
