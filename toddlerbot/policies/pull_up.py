import os
from typing import Optional

import joblib
import mink
import mujoco
import numpy as np
import numpy.typing as npt

from toddlerbot.policies.balance_pd import BalancePDPolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_robot_file_path

# from toddlerbot.sim.mujoco_utils import mj_render


class PullUpPolicy(BalancePDPolicy, policy_name="pull_up"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ip: str = "127.0.0.1",
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        super().__init__(
            name, robot, init_motor_pos, ip=ip, fixed_command=fixed_command
        )

        xml_file_path = find_robot_file_path(robot.name, "_fixed_scene.xml")
        model = mujoco.MjModel.from_xml_path(xml_file_path)

        self.default_qpos = np.array(model.keyframe("home").qpos)
        self.mj_joint_indices = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.joint_ordering
            ]
        )
        self.mj_motor_indices = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.motor_ordering
            ]
        )
        self.mj_passive_indices = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.passive_joint_names
            ]
        )

        self.posture_task = mink.PostureTask(model, cost=1.0)
        # pelvis_orientation_task = mink.FrameTask(
        #     frame_name="waist_link",
        #     frame_type="body",
        #     position_cost=0.0,
        #     orientation_cost=10.0,
        # )
        # com_task = mink.ComTask(cost=200.0)

        self.ee_tasks = []
        for side in ["left", "right"]:
            ee_site_name = f"{side}_ee_center"
            task = mink.FrameTask(
                frame_name=ee_site_name,
                frame_type="site",
                position_cost=50.0,
                orientation_cost=10.0,
                lm_damping=1.0,
            )
            self.ee_tasks.append(task)

        # self.foot_tasks = []
        # for side in ["left", "right"]:
        #     foot_site_name = f"{side}_foot_center"
        #     task = mink.FrameTask(
        #         frame_name=foot_site_name,
        #         frame_type="site",
        #         position_cost=20.0,
        #         orientation_cost=10.0,
        #         lm_damping=1.0,
        #     )
        #     self.foot_tasks.append(task)

        self.configuration = mink.Configuration(model)
        self.configuration.update_from_keyframe("home")

        self.posture_task.set_target_from_configuration(self.configuration)
        # pelvis_orientation_task.set_target_from_configuration(self.configuration)
        # com_task.set_target_from_configuration(self.configuration)
        self.ee_tasks[0].set_target_from_configuration(self.configuration)
        self.ee_tasks[1].set_target_from_configuration(self.configuration)
        # self.foot_tasks[0].set_target_from_configuration(self.configuration)
        # self.foot_tasks[1].set_target_from_configuration(self.configuration)

        self.tasks = [
            self.posture_task,
            # pelvis_orientation_task,
            # com_task,
            *self.ee_tasks,
            # *self.foot_tasks,
        ]

        velocity_limits = {}
        for i in range(model.nv):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name is None or len(joint_name) == 0:
                continue
            else:
                velocity_limits[joint_name] = np.pi / 4
        self.limits = [
            mink.ConfigurationLimit(model=model),
            mink.VelocityLimit(model, velocity_limits),
        ]

        self.solver = "quadprog"
        self.pos_threshold = 5e-3
        self.ori_threshold = 5e-3
        self.max_iter = 10

        motion_file_path = os.path.join("toddlerbot", "motion", "pull_up.pkl")
        if os.path.exists(motion_file_path):
            data_dict = joblib.load(motion_file_path)
        else:
            raise ValueError(f"No data files found in {motion_file_path}")

        self.time_arr = np.array(data_dict["time"])
        self.action_arr = np.array(data_dict["trajectory"], dtype=np.float32)

        start_idx = 0
        for idx in range(len(self.action_arr)):
            if np.allclose(self.default_motor_pos, self.action_arr[idx], atol=1e-1):
                start_idx = idx
                print(f"Truncating dataset at index {start_idx}")
                break

        self.time_arr = self.time_arr[start_idx:]
        self.action_arr = self.action_arr[start_idx:]

        self.step_curr = 0

    def get_arm_motor_pos(self, obs: Obs) -> npt.NDArray[np.float32]:
        left_ee_target = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_rpy_radians(-np.pi / 2, 0, np.pi / 2),
            np.array([0.1, 0.1, 0.2]),
        )
        right_ee_target = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_rpy_radians(np.pi / 2, 0, np.pi / 2),
            np.array([0.1, -0.1, 0.2]),
        )
        self.ee_tasks[0].set_target(left_ee_target)
        self.ee_tasks[1].set_target(right_ee_target)

        motor_angles = dict(zip(self.robot.motor_ordering, obs.motor_pos))
        joint_angles = self.robot.motor_to_joint_angles(motor_angles)
        passive_angles = self.robot.joint_to_passive_angles(joint_angles)
        qpos = self.default_qpos.copy()

        qpos[self.mj_motor_indices] = np.array(list(motor_angles.values()))
        qpos[self.mj_joint_indices] = np.array(list(joint_angles.values()))
        qpos[self.mj_passive_indices] = np.array(list(passive_angles.values()))

        self.configuration.update(qpos)
        mujoco.mj_forward(self.configuration.model, self.configuration.data)
        self.posture_task.set_target_from_configuration(self.configuration)

        for _ in range(self.max_iter):
            vel = mink.solve_ik(
                self.configuration,
                self.tasks,
                self.control_dt,
                self.solver,
                1e-12,
                limits=self.limits,
            )
            self.configuration.integrate_inplace(vel, self.control_dt)
            # mujoco.mj_camlight(self.configuration.model, self.configuration.data)
            mujoco.mj_step(self.configuration.model, self.configuration.data)

            # mj_render(self.configuration.model, self.configuration.data, lib="cv2")

            l_err = self.ee_tasks[0].compute_error(self.configuration)
            l_pos_achieved = np.linalg.norm(l_err[:3]) <= self.pos_threshold
            l_ori_achieved = np.linalg.norm(l_err[3:]) <= self.ori_threshold
            r_err = self.ee_tasks[1].compute_error(self.configuration)
            r_pos_achieved = np.linalg.norm(r_err[:3]) <= self.pos_threshold
            r_ori_achieved = np.linalg.norm(r_err[3:]) <= self.ori_threshold

            print(f"Left EE error: {l_err}")
            print(f"Right EE error: {r_err}")

            if l_pos_achieved and l_ori_achieved and r_pos_achieved and r_ori_achieved:
                break

        motor_pos = np.asarray(
            self.configuration.q[self.mj_motor_indices],
            dtype=np.float32,
        )

        return motor_pos[self.arm_motor_indices]
