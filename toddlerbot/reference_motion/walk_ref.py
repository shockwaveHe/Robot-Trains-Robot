from typing import Optional

import numpy as np
import numpy.typing as npt

from toddlerbot.reference_motion.ref_motion import RefMotionGenerator


class WalkRefMotionGenerator(RefMotionGenerator):
    def __init__(self, init_state: npt.NDArray[np.float32]):
        super().__init__("periodic", init_state)

    def get_state(
        self,
        path_frame: npt.NDArray[np.float32],
        phase: Optional[npt.NDArray[np.float32]] = None,
        command: Optional[npt.NDArray[np.float32]] = None,
    ) -> npt.NDArray[np.float32]:
        if phase is None:
            raise ValueError(f"phase is required for {self.motion_type} motion")

        if command is None:
            raise ValueError(f"command is required for {self.motion_type} motion")

        # pos: 3
        # quat: 4
        # linear_vel: 3
        # angular_vel: 3
        # joint_pos: 30
        # joint_vel: 30
        # left_contact: 1
        # right_contact: 1

        sin_pos = np.sin(2 * np.pi * phase)
        sin_pos_l = sin_pos.copy()
        sin_pos_r = sin_pos.copy()

        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        # > or < depends on the robot configuration
        # When sin_pos > 0, right foot is in stance phase
        # When sin_pos < 0, left foot is in stance phase
        sin_pos_l[sin_pos_l < 0] = 0
        self.ref_dof_pos[:, 2] = sin_pos_l * scale_1
        self.ref_dof_pos[:, 3] = sin_pos_l * scale_2
        self.ref_dof_pos[:, 4] = sin_pos_l * scale_1
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r > 0] = 0
        self.ref_dof_pos[:, 8] = sin_pos_r * scale_1
        self.ref_dof_pos[:, 9] = sin_pos_r * scale_2
        self.ref_dof_pos[:, 10] = sin_pos_r * scale_1
        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask
