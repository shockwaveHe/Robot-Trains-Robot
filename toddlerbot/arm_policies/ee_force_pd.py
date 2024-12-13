import mujoco
import numpy as np
import numpy.typing as npt

from toddlerbot.arm_policies import BaseArm, BaseArmPolicy, Obs


class EEForcePDArmPolicy(BaseArmPolicy, arm_policy_name="ee_force_pd"):
    def __init__(
        self,
        name: str,
        arm: BaseArm,
        init_joint_pos: npt.NDArray[np.float32],
        control_dt: float = 0.02,  # TODO: should this be dt or integration_dt?
    ):
        super().__init__(name, arm, init_joint_pos, control_dt)
        self.arm_model = mujoco.MjModel.from_xml_path(arm.xml_path.as_posix())
        self.arm_data = mujoco.MjData(self.arm_model)

        self.gravity_compensation: bool = False
        self.arm_model.body_gravcomp[:] = float(self.gravity_compensation)
        ## =================== ##
        ## Setup IK.
        ## =================== ##
        self.integration_dt: float = 0.1
        self.damping: float = 1e-4
        self.dt: float = 0.002
        # Gains for the twist computation. These should be between 0 and 1. 0 means no
        # movement, 1 means move the end-effector to the target in one integration step.
        self.Kforce: float = 0.000095
        self.Ktorque: float = 0.95
        # Nullspace P gain.
        self.Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
        # Maximum allowable joint velocity in rad/s.
        self.max_angvel = 0.785

        self.dt = control_dt
        self.solver = "quadprog"
        self.pos_threshold = 1e-4
        self.ori_threshold = 1e-4
        self.max_iters = 10
        self.arm_data.qpos[:] = init_joint_pos
        self.arm_data.ctrl[:] = init_joint_pos
        mujoco.mj_forward(self.arm_model, self.arm_data)

        self.ee_jac = np.zeros((6, self.arm.arm_dofs))
        self.diag = self.damping * np.eye(6)
        self.eye = np.eye(self.arm.arm_dofs)
        self.twist = np.zeros(6)
        self.target_force = np.array([0.0, 0.0, 27.0], dtype=np.float32)
        self.target_xyz = np.array([0.01766377, 0.0, 1.52856866], dtype=np.float32)
        self.site_id = self.arm_model.site("attachment_site").id
        self.q0 = init_joint_pos

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        self.arm_data.qpos[:] = obs.arm_joint_pos
        self.arm_data.qvel[:] = obs.arm_joint_vel
        mujoco.mj_step(self.arm_model, self.arm_data)
        # import ipdb; ipdb.set_trace()
        # the force the end-effector is applying to the toddlerbot in the world frame
        force_world_frame = (
            self.arm_data.site(self.site_id).xmat.reshape(3, 3) @ obs.ee_force
        )
        error_force = self.target_force - force_world_frame
        # error_force[:2] = 0.0
        self.twist[:3] = self.Kforce * error_force / self.integration_dt
        print(force_world_frame, self.twist[:3])
        # TODO: how to handle this error_quat from ee_torque?
        # TODO: how to make sure the robot not going too far?
        # TODO: axis transform matrix for force orientation
        # mujoco.mju_mat2Quat(self.site_quat, self.arm_data.site(self.site_id).xmat)
        # mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
        # mujoco.mju_mulQuat(self.error_quat, obs.mocap_quat, self.site_quat_conj)
        # mujoco.mju_quat2Vel(self.twist[3:], self.error_quat, 1.0)
        # self.twist[3:] *= self.Kori / self.integration_dt

        mujoco.mj_jacSite(
            self.arm_model,
            self.arm_data,
            self.ee_jac[:3],
            self.ee_jac[3:],
            self.site_id,
        )  # TODO: is this changing?

        dq = self.ee_jac.T @ np.linalg.solve(
            self.ee_jac @ self.ee_jac.T + self.diag, self.twist
        )
        if self.arm.arm_dofs > 6:
            dq += (self.eye - np.linalg.pinv(self.ee_jac) @ self.ee_jac) @ (
                self.Kn * (self.q0 - self.arm_data.qpos)
            )

        dq_abs_max = np.abs(dq).max()
        # import ipdb; ipdb.set_trace()
        if dq_abs_max > self.max_angvel:
            dq *= self.max_angvel / dq_abs_max

        q: npt.NDArray[np.float32] = self.arm_data.qpos.copy()
        mujoco.mj_integratePos(self.arm_model, q, dq, self.integration_dt)
        np.clip(q, *self.arm_model.jnt_range.T, out=q)  # type: ignore
        # pos_achieved = np.linalg.norm(err[:3]) <= self.pos_threshold
        # ori_achieved = np.linalg.norm(err[3:]) <= self.ori_threshold
        # if pos_achieved and ori_achieved:
        #     break
        return q
