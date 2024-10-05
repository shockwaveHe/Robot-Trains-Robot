import mink
import mujoco
import numpy as np
import numpy.typing as npt
from loop_rate_limiters import RateLimiter
from toddlerbot.policies import Obs, BaseArm, BaseArmPolicy

class EEArmPolicy(BaseArmPolicy, arm_policy_name="ee_arm"):
    def __init__(
        self,
        name: str,
        arm: BaseArm,
        init_joint_pos: npt.NDArray[np.float32],
        control_dt: float = 0.02,
    ):
        super().__init__(name, arm, init_joint_pos, control_dt)
        self.arm_model = mujoco.MjModel.from_xml_path(arm.xml_path.as_posix())
        self.arm_data = mujoco.MjData(self.arm_model)
        self.configuration = mink.Configuration(self.arm_model)
        self.end_effector_task = mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(model=self.arm_model, cost=1e-3)
        self.tasks = [
            self.end_effector_task,
            self.posture_task,
        ]
        self.dt = control_dt
        self.solver = "quadprog"
        self.pos_threshold = 1e-4
        self.ori_threshold = 1e-4
        self.max_iters = 10
        self.arm_data.qpos[:] = init_joint_pos
        self.arm_data.ctrl[:] = init_joint_pos
        self.configuration.update(self.arm_data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)
        mujoco.mj_forward(self.arm_model, self.arm_data)
        mink.move_mocap_to_frame(self.arm_model, self.arm_data, "target", "attachment_site", "site")

    def step(self, obs: Obs, is_real: bool = False) -> npt.NDArray[np.float32]:
        T_wt = mink.SE3.from_mocap_name(self.arm_model, self.arm_data, "target")
        period = 1
        amplitude = 0.2
        # import ipdb; ipdb.set_trace()
        T_wt.translation()[0] += amplitude * np.sin(2 * np.pi * self.arm_data.time / period)
        self.end_effector_task.set_target(T_wt)
        for i in range(self.max_iters):
            self.configuration.update(obs.arm_joint_pos)
            vel = mink.solve_ik(self.configuration, self.tasks, self.dt, self.solver, 1e-3)
            self.configuration.integrate_inplace(vel, self.dt)
            err = self.end_effector_task.compute_error(self.configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= self.pos_threshold
            ori_achieved = np.linalg.norm(err[3:]) <= self.ori_threshold
            if pos_achieved and ori_achieved:
                break
        self.arm_data.ctrl = self.configuration.q
        mujoco.mj_step(self.arm_model, self.arm_data)
        return self.configuration.q

