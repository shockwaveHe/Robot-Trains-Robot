import numpy as np
import numpy.typing as npt

from toddlerbot.arm_policies import BaseArm, Obs
from toddlerbot.arm_policies.ee_pd import EEPDArmPolicy


class EEFollowArmPolicy(EEPDArmPolicy, arm_policy_name="ee_follow"):
    def __init__(
        self,
        name: str,
        arm: BaseArm,
        init_joint_pos: npt.NDArray[np.float32],
        control_dt: float = 0.02,
    ):
        super().__init__(name, arm, init_joint_pos, control_dt)
        self.follow_diff = np.array([0.0, 0.0], dtype=np.float32)
        self.follow_height = 0.8

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        # TODO: make a specific obs class
        obs.mocap_pos[:2] = obs.torso_pos[:2] + self.follow_diff  # type: ignore
        obs.mocap_pos[2] = self.follow_height  # type: ignore
        # TODO: add robot_command later
        # if command is not None:
        #     obs.mocap_pos += command[:3] * self.control_dt
        return super().step(obs, is_real)
