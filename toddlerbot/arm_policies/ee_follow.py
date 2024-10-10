import numpy as np
import numpy.typing as npt
from toddlerbot.arm_policies.ee_pd import EEPDArmPolicy
from toddlerbot.arm_policies import Obs, BaseArm

class EEFollowArmPolicy(EEPDArmPolicy, arm_policy_name="ee_follow"):
    def __init__(self, name: str, arm: BaseArm, init_joint_pos: npt.NDArray[np.float32], control_dt: float = 0.02):
        super().__init__(name, arm, init_joint_pos, control_dt)
        self.follow_diff = np.array([0.0, 0.0], dtype=np.float32)
        self.follow_height = 0.8

    def step(self, obs: Obs, command: npt.NDArray[np.float32] | None = None, is_real: bool = False) -> npt.NDArray[np.float32]:
        obs.mocap_pos[:2] = obs.torso_pos[:2] + self.follow_diff
        obs.mocap_pos[2] = self.follow_height
        if command is not None:
            obs.mocap_pos += command[:3] * self.control_dt
        return super().step(obs, is_real)