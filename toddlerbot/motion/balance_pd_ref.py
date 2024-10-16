from typing import List, Tuple

import mujoco
from mujoco import mjx
from mujoco.mjx._src import support  # type: ignore

from toddlerbot.motion.motion_ref import MotionReference
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
        com_kp: List[float] = [250.0, 250.0, 0.0],
    ):
        super().__init__("balance_pd", "perceptual", robot, dt)

        self.arm_playback_speed = arm_playback_speed
        if self.arm_playback_speed > 0.0:
            self.arm_time_ref /= arm_playback_speed

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

        if self.use_jax:
            self.model = mjx.put_model(model)

            def forward(qpos):
                data = mjx.make_data(self.model)
                data = data.replace(qpos=qpos)
                return mjx.forward(self.model, data)

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

            def forward(qpos):
                data = mujoco.MjData(self.model)
                data.qpos = qpos
                mujoco.mj_forward(self.model, data)
                return data

            def jac_subtree_com(d, body):
                jacp = np.zeros((3, self.model.nv))
                mujoco.mj_jacSubtreeCom(self.model, d, jacp, body)
                return jacp

        self.forward = forward
        self.jac_subtree_com = jac_subtree_com

        qpos = self.default_qpos.copy()
        if self.arm_playback_speed > 0:
            qpos = inplace_update(
                qpos,
                7 + self.mj_joint_indices[self.arm_joint_indices],
                self.arm_joint_pos_ref[0],
            )
        data = self.forward(qpos)

        self.desired_com = np.array(data.subtree_com[0], dtype=np.float32)

        # foot_names = [
        #     f"{self.robot.foot_name}_collision",
        #     f"{self.robot.foot_name}_2_collision",
        # ]
        # self.desired_com = np.zeros(3, dtype=np.float32)
        # for i, foot_name in enumerate(foot_names):
        #     foot_geom_pos = np.array(data.geom(foot_name).xpos)
        #     foot_geom_mat = np.array(data.geom(foot_name).xmat).reshape(3, 3)
        #     foot_geom_size = np.array(model.geom(foot_name).size)
        #     # Define the local coordinates of the bounding box corners
        #     local_bbox_corners = np.array(
        #         [
        #             [-foot_geom_size[0], -foot_geom_size[1], 0.0],
        #             [foot_geom_size[0], -foot_geom_size[1], 0.0],
        #             [-foot_geom_size[0], foot_geom_size[1], 0.0],
        #             [foot_geom_size[0], foot_geom_size[1], 0.0],
        #         ]
        #     )

        #     # Transform local bounding box corners to world coordinates
        #     world_bbox_corners = (
        #         foot_geom_mat @ local_bbox_corners.T
        #     ).T + foot_geom_pos

        #     self.desired_com += np.mean(world_bbox_corners, axis=0) / 2

        # self.desired_com = inplace_update(self.desired_com, 2, qpos[2])

    def get_vel(self, command: ArrayType) -> Tuple[ArrayType, ArrayType]:
        lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        ang_vel = np.array([-command[3], 0.0, -command[4]], dtype=np.float32)
        return lin_vel, ang_vel

    def get_state_ref(
        self, state_curr: ArrayType, time_curr: float | ArrayType, command: ArrayType
    ) -> ArrayType:
        torso_state = self.integrate_torso_state(
            state_curr[:3], state_curr[3:7], command
        )
        joint_pos_curr = state_curr[13 : 13 + self.robot.nu]
        joint_pos = self.default_joint_pos.copy()

        # neck yaw, neck pitch, arm, waist roll, waist yaw
        neck_joint_pos = np.clip(
            joint_pos_curr[self.neck_joint_indices] + self.dt * command[:2],
            self.neck_joint_limits[0],
            self.neck_joint_limits[1],
        )
        joint_pos = inplace_update(joint_pos, self.neck_joint_indices, neck_joint_pos)

        if self.arm_playback_speed > 0:
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
            joint_pos = inplace_update(joint_pos, self.arm_joint_indices, arm_joint_pos)
        else:
            joint_pos = inplace_update(
                joint_pos,
                self.arm_joint_indices,
                joint_pos_curr[self.arm_joint_indices],
            )

        waist_joint_pos = np.clip(
            joint_pos_curr[self.waist_joint_indices] + self.dt * command[3:5],
            self.waist_joint_limits[0],
            self.waist_joint_limits[1],
        )
        joint_pos = inplace_update(joint_pos, self.waist_joint_indices, waist_joint_pos)

        com_z_target = self.com_z_limits[0] + command[5] * (
            self.com_z_limits[1] - self.com_z_limits[0]
        )
        leg_pitch_joint_pos = self.com_ik(com_z_target)
        joint_pos = inplace_update(
            joint_pos, self.leg_pitch_joint_indicies, leg_pitch_joint_pos
        )

        qpos = self.default_qpos.copy()
        qpos = inplace_update(qpos, slice(3, 7), torso_state[3:7])
        qpos = inplace_update(qpos, 7 + self.mj_joint_indices, joint_pos)
        data = self.forward(qpos)

        # Get the center of mass position
        com_pos = np.array(data.subtree_com[0], dtype=np.float32)
        # PD controller on CoM position
        com_pos_error = com_pos - self.desired_com
        com_ctrl = self.com_kp * com_pos_error
        com_jacp = self.jac_subtree_com(data, 0)

        # print(f"com_pos: {com_pos}")
        # print(f"desired_com: {self.desired_com}")
        # print(f"com_pos_error: {com_pos_error}")
        # print(f"com_ctrl: {com_ctrl}")

        # Update joint positions based on the PD controller command
        joint_pos = inplace_add(
            joint_pos,
            self.leg_pitch_joint_indicies,
            -com_ctrl[0]
            * com_jacp[0, 6 + self.mj_joint_indices[self.leg_pitch_joint_indicies]],
        )
        joint_pos = inplace_add(
            joint_pos,
            self.leg_roll_joint_indicies,
            -com_ctrl[1]
            * com_jacp[1, 6 + self.mj_joint_indices[self.leg_roll_joint_indicies]],
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
        arm_joint_pos = state_ref[13 + self.arm_motor_indices]
        arm_motor_pos = self.arm_ik(arm_joint_pos)
        motor_target = inplace_update(
            motor_target,
            self.arm_motor_indices,
            arm_motor_pos,
        )
        waist_joint_pos = state_ref[13 + self.waist_motor_indices]
        waist_motor_pos = self.waist_ik(waist_joint_pos)
        motor_target = inplace_update(
            motor_target,
            self.waist_motor_indices,
            waist_motor_pos,
        )

        return motor_target
