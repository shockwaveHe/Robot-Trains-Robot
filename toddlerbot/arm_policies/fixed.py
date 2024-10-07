import numpy as np
import numpy.typing as npt
from toddlerbot.arm_policies import Obs, BaseArm, BaseArmPolicy

class FixedArmPolicy(BaseArmPolicy, arm_policy_name="fixed"):
    def __init__(
        self,
        name: str,
        arm: BaseArm,
        init_joint_pos: npt.NDArray[np.float32],
        control_dt: float = 0.02,
    ):
        super().__init__(name, arm, init_joint_pos, control_dt)

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        return self.init_joint_pos