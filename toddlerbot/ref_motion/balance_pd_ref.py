import os
from typing import List, Tuple

import joblib
import mujoco
from mujoco import mjx
from mujoco.mjx._src import support  # type: ignore

from toddlerbot.ref_motion import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_add, inplace_update
from toddlerbot.utils.array_utils import array_lib as np
from toddlerbot.utils.file_utils import find_robot_file_path


class BalancePDReference(MotionReference):
    def __init__(
        self,
        robot: Robot,
        dt: float,
        arm_playback_speed: float = 1.0,
        com_z_lower_limit_offset: float = 0.01,
        com_kp: List[float] = [2000.0, 2000.0],
    ):
        super().__init__("balance_pd", "perceptual", robot, dt)

        self._setup_neck()
        self._setup_arm(arm_playback_speed)
        self._setup_waist()
        self._setup_leg(com_z_lower_limit_offset)
        self._setup_mjx(com_kp)

    def _get_gear_ratios(self, motor_names: List[str]) -> ArrayType:
        gear_ratios = np.ones(len(motor_names), dtype=np.float32)
        for i, motor_name in enumerate(motor_names):
            motor_config = self.robot.config["joints"][motor_name]
            if motor_config["transmission"] in ["gear", "rack_and_pinion"]:
                gear_ratios = inplace_update(
                    gear_ratios, i, -motor_config["gear_ratio"]
                )
        return gear_ratios

    def _setup_neck(self):
        neck_motor_names = [
            self.robot.motor_ordering[i] for i in self.neck_actuator_indices
        ]
        self.neck_gear_ratio = self._get_gear_ratios(neck_motor_names)
        self.neck_joint_limits = np.array(
            [
                self.robot.joint_limits["neck_yaw_driven"],
                self.robot.joint_limits["neck_pitch_driven"],
            ],
            dtype=np.float32,
        ).T

    def _setup_arm(self, arm_playback_speed: float):
        arm_motor_names = [
            self.robot.motor_ordering[i] for i in self.arm_actuator_indices
        ]
        self.arm_gear_ratio = self._get_gear_ratios(arm_motor_names)

        # Load the balance dataset
        data_path = os.path.join("toddlerbot", "ref_motion", "balance_dataset.lz4")
        data_dict = joblib.load(data_path)
        # state_array: [time(1), motor_pos(14), fsrL(1), fsrR(1), camera_frame_idx(1)]
        state_arr = data_dict["state_array"]
        self.arm_time_ref = (
            np.array(state_arr[:, 0] - state_arr[0, 0], dtype=np.float32)
            / arm_playback_speed
        )
        self.arm_joint_pos_ref = np.array(
            [
                self.arm_fk(arm_motor_pos)
                for arm_motor_pos in state_arr[:, 1 + self.arm_actuator_indices]
            ],
            dtype=np.float32,
        )
        self.arm_ref_size = len(self.arm_time_ref)

    def _setup_waist(self):
        self.waist_coef = np.array(
            [
                self.robot.config["general"]["offsets"]["waist_roll_coef"],
                self.robot.config["general"]["offsets"]["waist_yaw_coef"],
            ],
            dtype=np.float32,
        )
        self.waist_joint_limits = np.array(
            [
                self.robot.joint_limits["waist_roll"],
                self.robot.joint_limits["waist_yaw"],
            ],
            dtype=np.float32,
        ).T

    def _setup_leg(self, com_z_lower_limit_offset: float):
        self.knee_pitch_default = self.default_joint_pos[
            self.robot.joint_ordering.index("left_knee_pitch")
        ]
        self.hip_pitch_to_knee_z = self.robot.data_dict["offsets"][
            "hip_pitch_to_knee_z"
        ]
        self.knee_to_ank_pitch_z = self.robot.data_dict["offsets"][
            "knee_to_ank_pitch_z"
        ]
        self.hip_pitch_to_ank_pitch_z = np.sqrt(
            self.hip_pitch_to_knee_z**2
            + self.knee_to_ank_pitch_z**2
            - 2
            * self.hip_pitch_to_knee_z
            * self.knee_to_ank_pitch_z
            * np.cos(np.pi - self.knee_pitch_default)
        )
        self.shin_thigh_ratio = self.knee_to_ank_pitch_z / self.hip_pitch_to_knee_z

        self.com_z_limits = np.array(
            [
                self.leg_fk(self.robot.joint_limits["left_knee_pitch"][1]).item()
                + com_z_lower_limit_offset,
                0.0,
            ],
            dtype=np.float32,
        )
        self.left_knee_pitch_idx = self.robot.joint_ordering.index("left_knee_pitch")
        self.leg_pitch_joint_indicies = np.array(
            [
                self.robot.joint_ordering.index(joint_name)
                for joint_name in [
                    "left_hip_pitch",
                    "left_knee_pitch",
                    "left_ank_pitch",
                    "right_hip_pitch",
                    "right_knee_pitch",
                    "right_ank_pitch",
                ]
            ]
        )
        self.leg_roll_joint_indicies = np.array(
            [
                self.robot.joint_ordering.index(joint_name)
                for joint_name in [
                    "left_hip_roll",
                    "left_ank_roll",
                    "right_hip_roll",
                    "right_ank_roll",
                ]
            ]
        )

    def _setup_mjx(self, com_kp: List[float]):
        xml_path = find_robot_file_path(self.robot.name, suffix="_scene.xml")
        model = mujoco.MjModel.from_xml_path(xml_path)
        self.default_qpos = np.array(model.keyframe("home").qpos)
        self.q_start_idx = 7  # Account for the free joint
        self.mj_joint_indices = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.joint_ordering
            ]
        )
        self.mj_joint_indices -= 1  # Account for the free joint

        if self.use_jax:
            self.model = mjx.put_model(model)

            def jac_subtree_com(d, body):
                jacp = np.zeros((3, self.model.nv))
                # Forward pass starting from body
                for b in range(body, self.model.nbody):
                    # End of body subtree, break from the loop
                    if b > body and self.model.body_parentid[b] < body:
                        break

                    # b is in the body subtree, add mass-weighted Jacobian into jacp
                    jacp_b, _ = support.jac(self.model, d, d.xipos[b], b)
                    # print(f"jacp_b: {jacp_b}")
                    jacp = jacp.at[:].add(jacp_b.T * self.model.body_mass[b])

                # Normalize by subtree mass
                jacp /= self.model.body_subtreemass[body]

                return jacp

        else:
            self.model = model

            def jac_subtree_com(d, body):
                jacp = np.zeros((3, self.model.nv))
                mujoco.mj_jacSubtreeCom(self.model, d, jacp, body)
                return jacp

        self.jac_subtree_com = jac_subtree_com
        self.com_pos_init = np.array(
            [0.01, 0.0, 0.313], dtype=np.float32
        )  # self.data.subtree_com[0].copy()
        self.com_kp = np.array(com_kp, dtype=np.float32)

    def get_phase_signal(self, time_curr: float | ArrayType) -> ArrayType:
        return np.zeros(1, dtype=np.float32)

    def get_vel(self, command: ArrayType) -> Tuple[ArrayType, ArrayType]:
        lin_vel = np.array([0.0, 0.0, command[5]], dtype=np.float32)
        ang_vel = np.array([-command[3], 0.0, -command[4]], dtype=np.float32)
        return lin_vel, ang_vel

    def get_state_ref(
        self, state_curr: ArrayType, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        torso_state = self.integrate_torso_state(
            state_curr[:3], state_curr[3:7], command
        )
        joint_pos_curr = state_curr[13 : 13 + self.robot.nu]

        # neck yaw, neck pitch, arm, waist roll, waist yaw, squat
        neck_joint_pos = np.clip(
            joint_pos_curr[self.neck_joint_indices] + self.dt * command[:2],
            self.neck_joint_limits[0],
            self.neck_joint_limits[1],
        )

        command_ref_idx = (command[2] * (self.arm_ref_size - 2)).astype(int)
        time_ref = self.arm_time_ref[command_ref_idx] + time_curr
        ref_idx = np.minimum(
            np.searchsorted(self.arm_time_ref, time_ref, side="right") - 1,
            self.arm_ref_size - 2,
        )
        # Linearly interpolate between p_start and p_end
        arm_joint_pos_start = self.arm_joint_pos_ref[ref_idx]
        arm_joint_pos_end = self.arm_joint_pos_ref[ref_idx + 1]
        arm_duration = self.arm_time_ref[ref_idx + 1] - self.arm_time_ref[ref_idx]
        arm_joint_pos = arm_joint_pos_start + (
            arm_joint_pos_end - arm_joint_pos_start
        ) * ((time_ref - self.arm_time_ref[ref_idx]) / arm_duration)

        waist_joint_pos = np.clip(
            joint_pos_curr[self.waist_joint_indices] + self.dt * command[3:5],
            self.waist_joint_limits[0],
            self.waist_joint_limits[1],
        )

        com_z_curr = self.leg_fk(joint_pos_curr[self.left_knee_pitch_idx])
        com_z_target = np.clip(
            com_z_curr + self.dt * command[5],
            self.com_z_limits[0],
            self.com_z_limits[1],
        )
        leg_pitch_joint_pos = self.leg_ik(com_z_target)

        joint_pos = self.default_joint_pos.copy()
        joint_pos = inplace_update(joint_pos, self.neck_joint_indices, neck_joint_pos)
        joint_pos = inplace_update(joint_pos, self.arm_joint_indices, arm_joint_pos)
        joint_pos = inplace_update(joint_pos, self.waist_joint_indices, waist_joint_pos)
        joint_pos = inplace_update(
            joint_pos, self.leg_pitch_joint_indicies, leg_pitch_joint_pos
        )

        qpos = inplace_update(
            self.default_qpos, self.q_start_idx + self.mj_joint_indices, joint_pos
        )
        if self.use_jax:
            data = mjx.make_data(self.model)
            data = data.replace(qpos=qpos)
            data = mjx.forward(self.model, data)
        else:
            data = mujoco.MjData(self.model)
            data.qpos = qpos
            mujoco.mj_forward(self.model, data)

        # Get the center of mass position
        com_pos = data.subtree_com[0].copy()
        # PD controller on CoM position
        com_pos_error = com_pos[:2] - self.com_pos_init[:2]
        com_ctrl = self.com_kp * com_pos_error
        com_jacp = self.jac_subtree_com(data, 0)

        # print(f"com_pos: {com_pos}")
        # print(f"com_pos_init: {self.com_pos_init}")
        # print(f"com_pos_error: {com_pos_error}")
        # print(f"com_ctrl: {com_ctrl}")
        # print(f"com_jacp: {com_jacp}")

        # Update joint positions based on the PD controller command
        joint_pos = inplace_add(
            joint_pos,
            self.leg_pitch_joint_indicies,
            -com_ctrl[0]
            * com_jacp[
                0,
                self.q_start_idx + self.mj_joint_indices[self.leg_pitch_joint_indicies],
            ],
        )
        joint_pos = inplace_add(
            joint_pos,
            self.leg_roll_joint_indicies,
            -com_ctrl[1]
            * com_jacp[
                1,
                self.q_start_idx + self.mj_joint_indices[self.leg_roll_joint_indicies],
            ],
        )

        joint_vel = self.default_joint_vel.copy()

        stance_mask = np.ones(2, dtype=np.float32)

        return np.concatenate((torso_state, joint_pos, joint_vel, stance_mask))

    def override_motor_target(
        self, motor_target: ArrayType, state_ref: ArrayType
    ) -> ArrayType:
        neck_joint_pos = state_ref[13 + self.neck_actuator_indices]
        neck_motor_pos = self.neck_ik(neck_joint_pos)
        motor_target = inplace_update(
            motor_target,
            self.neck_actuator_indices,
            neck_motor_pos,
        )
        arm_joint_pos = state_ref[13 + self.arm_actuator_indices]
        arm_motor_pos = self.arm_ik(arm_joint_pos)
        motor_target = inplace_update(
            motor_target,
            self.arm_actuator_indices,
            arm_motor_pos,
        )
        waist_joint_pos = state_ref[13 + self.waist_actuator_indices]
        waist_motor_pos = self.waist_ik(waist_joint_pos)
        motor_target = inplace_update(
            motor_target,
            self.waist_actuator_indices,
            waist_motor_pos,
        )

        return motor_target

    def neck_ik(self, neck_joint_pos: ArrayType) -> ArrayType:
        neck_motor_pos = neck_joint_pos / self.neck_gear_ratio
        return neck_motor_pos

    def arm_fk(self, arm_motor_pos: ArrayType) -> ArrayType:
        arm_joint_pos = arm_motor_pos * self.arm_gear_ratio
        return arm_joint_pos

    def arm_ik(self, arm_joint_pos: ArrayType) -> ArrayType:
        arm_motor_pos = arm_joint_pos / self.arm_gear_ratio
        return arm_motor_pos

    def waist_ik(self, waist_joint_pos: ArrayType) -> ArrayType:
        waist_roll, waist_yaw = waist_joint_pos / self.waist_coef
        waist_act_1 = (-waist_roll + waist_yaw) / 2
        waist_act_2 = (waist_roll + waist_yaw) / 2
        return np.array([waist_act_1, waist_act_2], dtype=np.float32)

    def leg_fk(self, knee_angle: float | ArrayType) -> ArrayType:
        # Compute the length from hip pitch to ankle pitch along the z-axis
        com_z_target = np.array(
            np.sqrt(
                self.hip_pitch_to_knee_z**2
                + self.knee_to_ank_pitch_z**2
                - 2
                * self.hip_pitch_to_knee_z
                * self.knee_to_ank_pitch_z
                * np.cos(np.pi - knee_angle)
            )
            - self.hip_pitch_to_ank_pitch_z,
            dtype=np.float32,
        )
        return com_z_target

    def leg_ik(self, com_z_target: ArrayType) -> ArrayType:
        knee_angle_cos = (
            self.hip_pitch_to_knee_z**2
            + self.knee_to_ank_pitch_z**2
            - (self.hip_pitch_to_ank_pitch_z + com_z_target) ** 2
        ) / (2 * self.hip_pitch_to_knee_z * self.knee_to_ank_pitch_z)
        knee_angle_cos = np.clip(knee_angle_cos, -1.0, 1.0)
        knee_angle = np.abs(np.pi - np.arccos(knee_angle_cos))

        ank_pitch_angle = np.arctan2(
            np.sin(knee_angle),
            np.cos(knee_angle) + self.shin_thigh_ratio,
        )
        hip_pitch_angle = knee_angle - ank_pitch_angle

        return np.array(
            [
                -hip_pitch_angle,
                knee_angle,
                -ank_pitch_angle,
                hip_pitch_angle,
                -knee_angle,
                -ank_pitch_angle,
            ],
            dtype=np.float32,
        )
