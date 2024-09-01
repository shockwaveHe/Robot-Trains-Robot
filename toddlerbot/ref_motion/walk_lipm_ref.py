from typing import List, Optional, Tuple

from toddlerbot.algorithms.lipm.lipm_3d_copy import LIPM3DPlanner
from toddlerbot.ref_motion import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_add, inplace_update
from toddlerbot.utils.array_utils import array_lib as np


class WalkLIPMReference(MotionReference):
    def __init__(
        self,
        robot: Robot,
        default_joint_pos: Optional[ArrayType] = None,
        default_joint_vel: Optional[ArrayType] = None,
        max_knee_pitch: float = np.pi / 3,
    ):
        super().__init__("walk_lipm", "periodic", robot)

        self.default_joint_pos = default_joint_pos
        self.default_joint_vel = default_joint_vel
        self.max_knee_pitch = max_knee_pitch
        self.com_z = 0.336  # float(robot.config["general"]["offsets"]["torso_z"])
        self.foot_to_com_x = float(robot.data_dict["offsets"]["foot_to_com_x"])
        self.foot_to_com_y = float(robot.data_dict["offsets"]["foot_to_com_y"])

        if self.default_joint_pos is None:
            self.default_target_z = 0.0
        else:
            self.default_target_z = (
                float(robot.config["general"]["offsets"]["torso_z"]) - self.com_z
            )

        COM_pos_0: list[float] = [0.0, 0.0, self.com_z]
        COM_v0 = [0.1, 0.0]

        left_foot_pos = [self.foot_to_com_x, self.foot_to_com_y, 0]
        right_foot_pos = [self.foot_to_com_x, -self.foot_to_com_y, 0]

        # COM_pos_0 = [-0.4, 0.2, 1.0]
        # COM_v0 = [1.0, -0.01]
        # left_foot_pos = [-0.2, 0.3, 0]
        # right_foot_pos = [0.2, -0.3, 0]

        delta_t = 0.012

        self.planner = LIPM3DPlanner(
            dt=delta_t,
            T_sup=0.34,
            support_leg="left_leg",
        )
        self.planner.initialize_model(COM_pos_0, left_foot_pos, right_foot_pos)

        self.planner.support_leg = "left_leg"
        if self.planner.support_leg == "left_leg":
            support_foot_pos = self.planner.left_foot_pos
            self.planner.p_x = self.planner.left_foot_pos[0]
            self.planner.p_y = self.planner.left_foot_pos[1]
        else:
            support_foot_pos = self.planner.right_foot_pos
            self.planner.p_x = self.planner.right_foot_pos[0]
            self.planner.p_y = self.planner.right_foot_pos[1]

        self.planner.x_0 = self.planner.COM_pos[0] - support_foot_pos[0]
        self.planner.y_0 = self.planner.COM_pos[1] - support_foot_pos[1]
        self.planner.vx_0 = COM_v0[0]
        self.planner.vy_0 = COM_v0[1]

        self.init_joint_pos = np.array(  # type: ignore
            list(robot.init_joint_angles.values()), dtype=np.float32
        )

        self.reset()

    def get_state_ref(
        self,
        path_pos: ArrayType,
        path_quat: ArrayType,
        phase: Optional[float | ArrayType] = None,
        command: Optional[ArrayType] = None,
    ) -> ArrayType:
        if phase is None:
            raise ValueError(f"phase is required for {self.name}")

        if command is None:
            raise ValueError(f"command is required for {self.name}")

        if not self.planned:
            self.plan(path_pos, path_quat, command, 10)

        linear_vel = np.array([command[0], command[1], 0.0], dtype=np.float32)  # type: ignore
        angular_vel = np.array([0.0, 0.0, command[2]], dtype=np.float32)  # type: ignore

        sin_phase_signal = np.sin(2 * np.pi * phase)  # type: ignore

        assert self.default_joint_pos is not None
        joint_pos = self.default_joint_pos.copy()  # type: ignore
        joint_pos = inplace_update(
            joint_pos, slice(4, 16), self.ref_joint_pos_traj[self.idx]
        )

        if self.default_joint_vel is None:
            joint_vel = np.zeros(len(self.robot.joint_ordering), dtype=np.float32)  # type: ignore
        else:
            joint_vel = self.default_joint_vel.copy()  # type: ignore

        # Update
        stance_mask = np.zeros(2, dtype=np.float32)  # type: ignore
        stance_mask = inplace_update(stance_mask, 0, np.any(sin_phase_signal >= 0))  # type: ignore
        stance_mask = inplace_update(stance_mask, 1, np.any(sin_phase_signal < 0))  # type: ignore

        self.idx += 1

        return np.concatenate(  # type: ignore
            (
                path_pos,
                path_quat,
                linear_vel,
                angular_vel,
                joint_pos,
                joint_vel,
                stance_mask,
            )
        )

    def reset(self):
        self.idx = 0
        self.planned = False

    def plan(
        self,
        path_pos: ArrayType,
        path_quat: ArrayType,
        command: ArrayType,
        duration: float,
    ):
        swing_height = 0.03

        s_x = 0.12
        s_y = 0.03
        a = 1.0
        b = 1.0
        theta = 0.0

        step_num = 0
        global_time = 0

        COM_pos_x: List[float] = []
        COM_pos_y: List[float] = []
        left_foot_pos_x: List[float] = []
        left_foot_pos_y: List[float] = []
        left_foot_pos_z: List[float] = []
        right_foot_pos_x: List[float] = []
        right_foot_pos_y: List[float] = []
        right_foot_pos_z: List[float] = []

        swing_data_len = int(self.planner.T_sup / self.planner.dt)
        swing_foot_pos = np.zeros((swing_data_len, 3))
        j = 0

        switch_index = swing_data_len

        for i in range(int(np.ceil(duration / self.planner.dt))):
            global_time += self.planner.dt

            self.planner.step()

            if step_num >= 1:
                if self.planner.support_leg == "left_leg":
                    self.planner.right_foot_pos = [
                        swing_foot_pos[j, 0],
                        swing_foot_pos[j, 1],
                        swing_foot_pos[j, 2],
                    ]
                else:
                    self.planner.left_foot_pos = [
                        swing_foot_pos[j, 0],
                        swing_foot_pos[j, 1],
                        swing_foot_pos[j, 2],
                    ]
                j += 1

            # record data
            COM_pos_x.append(self.planner.x_t + self.planner.support_foot_pos[0])
            COM_pos_y.append(self.planner.y_t + self.planner.support_foot_pos[1])
            left_foot_pos_x.append(self.planner.left_foot_pos[0])
            left_foot_pos_y.append(self.planner.left_foot_pos[1])
            left_foot_pos_z.append(self.planner.left_foot_pos[2])
            right_foot_pos_x.append(self.planner.right_foot_pos[0])
            right_foot_pos_y.append(self.planner.right_foot_pos[1])
            right_foot_pos_z.append(self.planner.right_foot_pos[2])

            # switch the support leg
            if (i > 0) and (i % switch_index == 0):
                j = 0
                # Switch the support leg / Update current body state (self.x_0, self.y_0, self.vx_0, self.vy_0)
                self.planner.switch_support_leg()
                step_num += 1

                if self.planner.support_leg == "left_leg":
                    self.planner.support_foot_pos = self.planner.left_foot_pos
                    self.planner.p_x = self.planner.left_foot_pos[0]
                    self.planner.p_y = self.planner.left_foot_pos[1]
                else:
                    self.planner.support_foot_pos = self.planner.right_foot_pos
                    self.planner.p_x = self.planner.right_foot_pos[0]
                    self.planner.p_y = self.planner.right_foot_pos[1]

                # calculate the next foot locations, with modification, stable
                x_0, vx_0, y_0, vy_0 = self.planner.calculate_Xt_Vt(
                    self.planner.T_sup
                )  #

                if self.planner.support_leg == "left_leg":
                    x_0 = (
                        x_0 + self.planner.left_foot_pos[0]
                    )  # need the absolute position for next step
                    y_0 = (
                        y_0 + self.planner.left_foot_pos[1]
                    )  # need the absolute position for next step
                else:
                    x_0 = (
                        x_0 + self.planner.right_foot_pos[0]
                    )  # need the absolute position for next step
                    y_0 = (
                        y_0 + self.planner.right_foot_pos[1]
                    )  # need the absolute position for next step

                self.planner.calculate_foot_location_for_next_step(
                    s_x, s_y, a, b, theta, x_0, vx_0, y_0, vy_0
                )

                # calculate the foot positions for swing phase
                if self.planner.support_leg == "left_leg":
                    right_foot_target_pos = [
                        self.planner.p_x_star,
                        self.planner.p_y_star,
                        0,
                    ]
                    swing_foot_pos[:, 0] = np.linspace(
                        self.planner.right_foot_pos[0],
                        right_foot_target_pos[0],
                        swing_data_len,
                    )
                    swing_foot_pos[:, 1] = np.linspace(
                        self.planner.right_foot_pos[1],
                        right_foot_target_pos[1],
                        swing_data_len,
                    )
                    swing_foot_pos[:, 2] = (
                        swing_height
                        / (swing_data_len // 2 - 1)
                        * np.concatenate(  # type: ignore
                            (
                                np.arange(swing_data_len // 2, dtype=np.float32),  # type: ignore
                                np.arange(
                                    swing_data_len // 2 - 1, -1, -1, dtype=np.float32
                                ),
                            )
                        )
                    )
                else:
                    left_foot_target_pos = [
                        self.planner.p_x_star,
                        self.planner.p_y_star,
                        0,
                    ]
                    swing_foot_pos[:, 0] = np.linspace(
                        self.planner.left_foot_pos[0],
                        left_foot_target_pos[0],
                        swing_data_len,
                    )
                    swing_foot_pos[:, 1] = np.linspace(
                        self.planner.left_foot_pos[1],
                        left_foot_target_pos[1],
                        swing_data_len,
                    )
                    swing_foot_pos[:, 2] = (
                        swing_height
                        / (swing_data_len // 2 - 1)
                        * np.concatenate(  # type: ignore
                            (
                                np.arange(swing_data_len // 2, dtype=np.float32),  # type: ignore
                                np.arange(
                                    swing_data_len // 2 - 1, -1, -1, dtype=np.float32
                                ),
                            )
                        )
                    )

        self.ref_joint_pos_traj = self.solve_ik(
            COM_pos_x,
            COM_pos_y,
            left_foot_pos_x,
            left_foot_pos_y,
            left_foot_pos_z,
            right_foot_pos_x,
            right_foot_pos_y,
            right_foot_pos_z,
        )

        self.planned = True

    def solve_ik(
        self,
        COM_pos_x: List[float],
        COM_pos_y: List[float],
        left_foot_pos_x: List[float],
        left_foot_pos_y: List[float],
        left_foot_pos_z: List[float],
        right_foot_pos_x: List[float],
        right_foot_pos_y: List[float],
        right_foot_pos_z: List[float],
    ):
        left_leg_joint_pos = []
        for pos in zip(
            COM_pos_x, COM_pos_y, left_foot_pos_x, left_foot_pos_y, left_foot_pos_z
        ):
            joint_pos = self.foot_ik(
                [
                    pos[2] - pos[0] - self.foot_to_com_x,
                    pos[3] - pos[1] - self.foot_to_com_y,
                    pos[4],
                ],
                side="left",
            )
            left_leg_joint_pos.append(joint_pos)

        right_leg_joint_pos = []
        for pos in zip(
            COM_pos_x, COM_pos_y, right_foot_pos_x, right_foot_pos_y, right_foot_pos_z
        ):
            joint_pos = self.foot_ik(
                [
                    pos[2] - pos[0] - self.foot_to_com_x,
                    pos[3] - pos[1] + self.foot_to_com_y,
                    pos[4],
                ],
                side="right",
            )
            right_leg_joint_pos.append(joint_pos)

        joint_pos_traj = np.zeros(  # type: ignore
            (len(left_leg_joint_pos), 12), dtype=np.float32
        )

        for i, (left, right) in enumerate(zip(left_leg_joint_pos, right_leg_joint_pos)):
            joint_pos_traj = inplace_update(joint_pos_traj, (i, slice(None, 6)), left)
            joint_pos_traj = inplace_update(joint_pos_traj, (i, slice(6, None)), right)

        return joint_pos_traj

    def foot_ik(
        self,
        target_foot_pos: List[float],
        target_foot_ori: List[float] = [0.0, 0.0, 0.0],
        side: str = "left",
    ) -> ArrayType:
        target_x, target_y, target_z = target_foot_pos
        ank_roll, ank_pitch, hip_yaw = target_foot_ori

        offsets = self.robot.data_dict["offsets"]

        transformed_x = target_x * np.cos(hip_yaw) + target_y * np.sin(hip_yaw)  # type: ignore
        transformed_y = -target_x * np.sin(hip_yaw) + target_y * np.cos(hip_yaw)  # type: ignore
        transformed_z = (
            offsets["hip_pitch_to_knee_z"]
            + offsets["knee_to_ank_pitch_z"]
            - target_z
            - self.default_target_z
        )

        hip_roll = np.arctan2(  # type: ignore
            transformed_y, transformed_z + offsets["hip_roll_to_pitch_z"]
        )

        leg_projected_yz_length = np.sqrt(transformed_y**2 + transformed_z**2)  # type: ignore
        leg_length = np.sqrt(transformed_x**2 + leg_projected_yz_length**2)  # type: ignore
        leg_pitch = np.arctan2(transformed_x, leg_projected_yz_length)  # type: ignore
        hip_disp_cos = (
            leg_length**2
            + offsets["hip_pitch_to_knee_z"] ** 2
            - offsets["knee_to_ank_pitch_z"] ** 2
        ) / (2 * leg_length * offsets["hip_pitch_to_knee_z"])
        hip_disp = np.arccos(min(max(hip_disp_cos, -1.0), 1.0))  # type: ignore
        ank_disp = np.arcsin(  # type: ignore
            offsets["hip_pitch_to_knee_z"]
            / offsets["knee_to_ank_pitch_z"]
            * np.sin(hip_disp)  # type: ignore
        )
        hip_pitch = -leg_pitch - hip_disp
        knee_pitch = hip_disp + ank_disp
        ank_pitch += knee_pitch + hip_pitch

        if side == "left":
            return np.array(  # type: ignore
                [
                    hip_yaw,
                    -hip_roll,
                    hip_pitch,
                    knee_pitch,
                    -ank_roll + hip_roll,
                    -ank_pitch,
                ],
                dtype=np.float32,
            )
        else:
            return np.array(  # type: ignore
                [
                    hip_yaw,
                    hip_roll,
                    -hip_pitch,
                    -knee_pitch,
                    -ank_roll + hip_roll,
                    -ank_pitch,
                ],
                dtype=np.float32,
            )
