from typing import List, Tuple

import mujoco
from mujoco import mjx

from toddlerbot.motion.motion_ref import MotionReference
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.array_utils import ArrayType, inplace_update
from toddlerbot.utils.array_utils import array_lib as np
from toddlerbot.utils.file_utils import find_robot_file_path


class SquatReference(MotionReference):
    def __init__(self, robot: Robot, dt: float, com_kp: List[float] = [1.0, 1.0]):
        super().__init__("squat", "perceptual", robot, dt)

        self.com_kp = np.array(com_kp, dtype=np.float32)

        self._setup_mjx()

    def _setup_mjx(self):
        xml_path = find_robot_file_path(self.robot.name, suffix="_scene.xml")
        model = mujoco.MjModel.from_xml_path(xml_path)
        self.default_qpos = np.array(model.keyframe("home").qpos)
        self.mj_joint_indices = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.joint_ordering
            ]
        )
        self.mj_joint_indices -= 1  # Account for the free joint

        self.left_foot_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "left_foot_center"
        )
        self.right_foot_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "right_foot_center"
        )

        if self.use_jax:
            self.model = mjx.put_model(model)

            def forward(qpos):
                data = mjx.make_data(self.model)
                data = data.replace(qpos=qpos)
                return mjx.forward(self.model, data)

        else:
            self.model = model

            def forward(qpos):
                data = mujoco.MjData(self.model)
                data.qpos = qpos
                mujoco.mj_forward(self.model, data)
                return data

        self.forward = forward

        data = self.forward(self.default_qpos)
        self.desired_com = np.array(data.subtree_com[0], dtype=np.float32)

    def get_vel(self, command: ArrayType) -> Tuple[ArrayType, ArrayType]:
        lin_vel = np.array([0.0, 0.0, command[5]], dtype=np.float32)
        ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return lin_vel, ang_vel

    def get_state_ref(
        self, state_curr: ArrayType, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        torso_state = self.integrate_torso_state(state_curr, command)
        joint_pos_curr = state_curr[13 : 13 + self.robot.nu]

        joint_pos = self.default_joint_pos.copy()

        neck_yaw_pos = np.interp(
            command[0],
            np.array([-1, 0, 1]),
            np.array([self.neck_joint_limits[0, 0], 0.0, self.neck_joint_limits[1, 0]]),
        )
        neck_pitch_pos = np.interp(
            command[1],
            np.array([-1, 0, 1]),
            np.array([self.neck_joint_limits[0, 1], 0.0, self.neck_joint_limits[1, 1]]),
        )
        neck_joint_pos = np.array([neck_yaw_pos, neck_pitch_pos])
        joint_pos = inplace_update(joint_pos, self.neck_joint_indices, neck_joint_pos)

        ref_idx = (command[2] * (self.arm_ref_size - 2)).astype(int)
        # Linearly interpolate between p_start and p_end
        arm_joint_pos = self.arm_joint_pos_ref[ref_idx]
        joint_pos = inplace_update(joint_pos, self.arm_joint_indices, arm_joint_pos)

        waist_roll_pos = np.interp(
            command[3],
            np.array([-1, 0, 1]),
            np.array(
                [self.waist_joint_limits[0, 0], 0.0, self.waist_joint_limits[1, 0]]
            ),
        )
        waist_yaw_pos = np.interp(
            command[4],
            np.array([-1, 0, 1]),
            np.array(
                [self.waist_joint_limits[0, 1], 0.0, self.waist_joint_limits[1, 1]]
            ),
        )
        waist_joint_pos = np.array([waist_roll_pos, waist_yaw_pos])
        joint_pos = inplace_update(joint_pos, self.waist_joint_indices, waist_joint_pos)

        qpos = self.default_qpos.copy()
        qpos = inplace_update(qpos, slice(3, 7), torso_state[3:7])
        qpos = inplace_update(qpos, 7 + self.mj_joint_indices, joint_pos)
        data = self.forward(qpos)

        com_pos = np.array(data.subtree_com[0], dtype=np.float32)
        # PD controller on CoM position
        com_pos_error = self.desired_com[:2] - com_pos[:2]
        com_ctrl = self.com_kp * com_pos_error
        com_curr = self.com_fk(
            joint_pos_curr[self.left_knee_pitch_idx],
            joint_pos_curr[self.left_hip_pitch_idx],
        )

        com_z_target = np.clip(
            com_curr[2] + self.dt * command[5],
            self.com_z_limits[0],
            self.com_z_limits[1],
        )

        # print(f"command: {command[5]}")
        # print(f"com_pos: {com_pos}")
        # print(f"desired_com: {self.desired_com}")
        # print(f"com_pos_error: {com_pos_error}")
        # print(f"com_ctrl: {com_ctrl}")
        # print(f"com_curr: {com_curr}")
        # print(f"com_z_target: {com_z_target}")

        joint_pos = inplace_update(
            joint_pos,
            self.leg_joint_indices,
            self.com_ik(com_z_target, com_ctrl[0], com_ctrl[1]),
        )

        joint_vel = self.default_joint_vel.copy()

        stance_mask = np.ones(2, dtype=np.float32)

        return np.concatenate((torso_state, joint_pos, joint_vel, stance_mask))

    def override_motor_target(
        self, motor_target: ArrayType, state_ref: ArrayType
    ) -> ArrayType:
        neck_joint_pos = state_ref[13 + self.neck_motor_indices]
        neck_motor_pos = self.neck_ik(neck_joint_pos)
        motor_target = inplace_update(
            motor_target,
            self.neck_motor_indices,
            neck_motor_pos,
        )
        waist_joint_pos = state_ref[13 + self.waist_motor_indices]
        waist_motor_pos = self.waist_ik(waist_joint_pos)
        motor_target = inplace_update(
            motor_target,
            self.waist_motor_indices,
            waist_motor_pos,
        )
        leg_joint_pos = state_ref[13 + self.leg_motor_indices]
        leg_motor_pos = self.leg_ik(leg_joint_pos)
        motor_target = inplace_update(
            motor_target,
            self.leg_motor_indices,
            leg_motor_pos,
        )

        return motor_target
