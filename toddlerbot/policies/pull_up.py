import os
from typing import Dict, Optional, Tuple

import joblib
import mink
import mujoco
import numpy as np
import numpy.typing as npt

from toddlerbot.policies.balance_pd import BalancePDPolicy
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.math_utils import euler2mat, quat2mat

# from toddlerbot.utils.misc_utils import profile


def visualize_ee_trajectory(before_traj, after_traj):
    """
    Visualize the EE position trajectories before and after the update.

    Parameters:
    - before_traj (np.ndarray): The trajectory before the update, shape (n_waypoints, 10).
    - after_traj (np.ndarray): The trajectory after the update, shape (n_waypoints, 10).
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 6))

    # 3D plot for left EE trajectory
    ax_left = fig.add_subplot(121, projection="3d")
    ax_left.plot(
        before_traj[:, 0],
        before_traj[:, 1],
        before_traj[:, 2],
        label="Before Update",
        color="blue",
        linestyle="--",
    )
    ax_left.plot(
        after_traj[:, 0],
        after_traj[:, 1],
        after_traj[:, 2],
        label="After Update",
        color="red",
    )
    ax_left.set_title("Left EE Trajectory")
    ax_left.set_xlabel("X")
    ax_left.set_ylabel("Y")
    ax_left.set_zlabel("Z")
    ax_left.legend()
    ax_left.grid(True)

    # 3D plot for right EE trajectory
    ax_right = fig.add_subplot(122, projection="3d")
    ax_right.plot(
        before_traj[:, 7],
        before_traj[:, 8],
        before_traj[:, 9],
        label="Before Update",
        color="blue",
        linestyle="--",
    )
    ax_right.plot(
        after_traj[:, 7],
        after_traj[:, 8],
        after_traj[:, 9],
        label="After Update",
        color="red",
    )
    ax_right.set_title("Right EE Trajectory")
    ax_right.set_xlabel("X")
    ax_right.set_ylabel("Y")
    ax_right.set_zlabel("Z")
    ax_right.legend()
    ax_right.grid(True)

    # Ensure uniform scale for both subplots
    def set_uniform_scale(ax, trajectories):
        """
        Adjust the 3D axes of a Matplotlib plot to have uniform scaling.

        Parameters:
        - ax: The 3D axes to adjust.
        - trajectories (list of np.ndarray): List of trajectory arrays to consider for scaling.
        """
        all_points = np.vstack(trajectories)
        x_limits = [np.min(all_points[:, 0]), np.max(all_points[:, 0])]
        y_limits = [np.min(all_points[:, 1]), np.max(all_points[:, 1])]
        z_limits = [np.min(all_points[:, 2]), np.max(all_points[:, 2])]

        max_range = (
            max(
                x_limits[1] - x_limits[0],
                y_limits[1] - y_limits[0],
                z_limits[1] - z_limits[0],
            )
            / 2.0
        )

        mid_x = np.mean(x_limits)
        mid_y = np.mean(y_limits)
        mid_z = np.mean(z_limits)

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Apply uniform scaling to both subplots
    set_uniform_scale(ax_left, [before_traj[:, :3], after_traj[:, :3]])
    set_uniform_scale(ax_right, [before_traj[:, 7:10], after_traj[:, 7:10]])

    plt.tight_layout()
    plt.show()


class PullUpPolicy(BalancePDPolicy, policy_name="pull_up"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        ip: str = "",
        fixed_command: Optional[npt.NDArray[np.float32]] = None,
    ):
        super().__init__(
            name, robot, init_motor_pos, ip=ip, fixed_command=fixed_command
        )

        fixed_xml_file_path = find_robot_file_path(robot.name, "_fixed_scene.xml")
        fixed_model = mujoco.MjModel.from_xml_path(fixed_xml_file_path)

        self.default_qpos = np.array(fixed_model.keyframe("home").qpos)
        self.mj_joint_indices = np.array(
            [
                mujoco.mj_name2id(fixed_model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.joint_ordering
            ]
        )
        self.mj_motor_indices = np.array(
            [
                mujoco.mj_name2id(fixed_model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.motor_ordering
            ]
        )
        self.mj_passive_indices = np.array(
            [
                mujoco.mj_name2id(fixed_model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.passive_joint_names
            ]
        )

        self.default_torso_z = robot.config["general"]["offsets"]["default_torso_z"]

        self.posture_task = mink.PostureTask(fixed_model, cost=1.0)
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
                position_cost=100.0,
                orientation_cost=20.0,
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

        self.configuration = mink.Configuration(fixed_model)
        self.configuration.update_from_keyframe("home")

        self.model = self.configuration.model
        self.data = self.configuration.data

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
        for i in range(fixed_model.nv):
            joint_name = mujoco.mj_id2name(fixed_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name is None or len(joint_name) == 0:
                continue
            else:
                velocity_limits[joint_name] = np.pi / 4

        self.limits = [
            mink.ConfigurationLimit(model=fixed_model),
            mink.VelocityLimit(fixed_model, velocity_limits),
        ]

        self.solver = "quadprog"
        self.pos_threshold = 5e-3
        self.max_iter = 10

        self.root_to_left_eye_t = np.array([0.032, 0.017, 0.19], dtype=np.float32)
        self.grasp_motion_updated = False

        grasp_motion_path = os.path.join("toddlerbot", "motion", "pull_up_grasp.pkl")
        if os.path.exists(grasp_motion_path):
            grasp_data_dict = joblib.load(grasp_motion_path)
        else:
            raise ValueError(f"No data files found in {grasp_motion_path}")

        self.grasp_time_arr = np.array(grasp_data_dict["time"])
        self.grasp_action_arr = np.array(
            grasp_data_dict["action_traj"], dtype=np.float32
        )
        self.grasp_ee_arr = np.array(grasp_data_dict["ee_traj"], dtype=np.float32)
        self.grasp_ee_arr[:, 2] -= self.default_torso_z
        self.grasp_ee_arr[:, 9] -= self.default_torso_z

        pull_motion_path = os.path.join("toddlerbot", "motion", "pull_up_pull.pkl")
        if os.path.exists(pull_motion_path):
            pull_data_dict = joblib.load(pull_motion_path)
        else:
            raise ValueError(f"No data files found in {pull_motion_path}")

        self.pull_time_arr = np.array(pull_data_dict["time"])
        self.pull_action_arr = np.array(pull_data_dict["action_traj"], dtype=np.float32)
        self.pull_ee_arr = np.array(pull_data_dict["ee_traj"], dtype=np.float32)
        self.pull_ee_arr[:, 2] -= self.default_torso_z
        self.pull_ee_arr[:, 9] -= self.default_torso_z

        self.step_curr = 0

    # @profile()
    def get_arm_motor_pos(self, obs: Obs) -> npt.NDArray[np.float32]:
        step_idx = min(self.step_curr, self.grasp_ee_arr.shape[0] - 1)
        left_ee_target = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(
                np.asarray(quat2mat(self.grasp_ee_arr[step_idx][3:7]))
            ),
            self.grasp_ee_arr[step_idx][:3],
        )
        right_ee_target = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(
                np.asarray(quat2mat(self.grasp_ee_arr[step_idx][10:]))
            ),
            self.grasp_ee_arr[step_idx][7:10],
        )
        self.ee_tasks[0].set_target(left_ee_target)
        self.ee_tasks[1].set_target(right_ee_target)

        # print(f"Left EE target: {left_ee_target}")
        # print(f"Right EE target: {right_ee_target}")

        # motor_angles = dict(
        #     zip(self.robot.motor_ordering, self.grasp_action_arr[step_idx])
        # )
        motor_angles = dict(zip(self.robot.motor_ordering, obs.motor_pos))
        joint_angles = self.robot.motor_to_joint_angles(motor_angles)
        passive_angles = self.robot.joint_to_passive_angles(joint_angles)
        qpos = self.default_qpos.copy()

        qpos[self.mj_motor_indices] = np.array(list(motor_angles.values()))
        qpos[self.mj_joint_indices] = np.array(list(joint_angles.values()))
        qpos[self.mj_passive_indices] = np.array(list(passive_angles.values()))

        self.configuration.update(qpos)
        mujoco.mj_forward(self.model, self.data)
        # self.posture_task.set_target_from_configuration(self.configuration)

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
            mujoco.mj_step(self.model, self.data)
            # mujoco.mj_camlight(self.model, self.data)
            # from toddlerbot.sim.mujoco_utils import mj_render

            # mj_render(self.model, self.data, lib="cv2")

            left_pos_error = np.linalg.norm(
                self.data.site("left_ee_center").xpos - self.grasp_ee_arr[step_idx][:3]
            )
            right_pos_error = np.linalg.norm(
                self.data.site("right_ee_center").xpos
                - self.grasp_ee_arr[step_idx][7:10]
            )

            # print(f"Left EE error: {left_pos_error}")
            # print(f"Right EE error: {right_pos_error}")

            if (
                left_pos_error < self.pos_threshold
                and right_pos_error < self.pos_threshold
            ):
                break

        motor_pos = np.asarray(
            self.configuration.q[self.mj_motor_indices],
            dtype=np.float32,
        )

        return motor_pos[self.arm_motor_indices]

    def get_ee_pos_target(
        self, obs: Obs, grasp_delta_y: float = 0.1, grasp_delta_z: float = -0.1
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        left_eye_transform = np.eye(4, dtype=np.float32)
        left_eye_R = euler2mat(obs.euler)
        left_eye_transform[:3, :3] = left_eye_R
        left_eye_transform[:3, 3] = left_eye_R @ self.root_to_left_eye_t

        if self.left_eye is None or self.right_eye is None:
            tag_pose_avg = np.eye(4, dtype=np.float32)
            tag_pose_avg[:3, 3] = np.array([0.1, -0.017, 0.1], dtype=np.float32)
            tag_pose_avg_world = left_eye_transform @ tag_pose_avg
        else:
            left_tag_poses = self.left_eye.detect_tags()
            right_tag_poses = self.right_eye.detect_tags()

            tag_id = list(left_tag_poses.keys())[0]
            tag_pose_avg = np.mean(
                [left_tag_poses[tag_id], right_tag_poses[tag_id]], axis=0
            )
            tag_pose_avg_world = left_eye_transform @ tag_pose_avg

        left_ee_grasp_pos = (
            tag_pose_avg_world[:3, :3]
            @ np.array([0, grasp_delta_y, grasp_delta_z], dtype=np.float32)
            + tag_pose_avg_world[:3, 3]
        )

        right_ee_grasp_pos = (
            tag_pose_avg_world[:3, :3]
            @ np.array([0, -grasp_delta_y, grasp_delta_z], dtype=np.float32)
            + tag_pose_avg_world[:3, 3]
        )

        return left_ee_grasp_pos, right_ee_grasp_pos

    def update_grasp_ee_traj(self, obs: Obs):
        left_ee_grasp_pos, right_ee_grasp_pos = self.get_ee_pos_target(obs)
        left_ee_grasp_pos_prev = self.grasp_ee_arr[-1][:3]
        right_ee_grasp_pos_prev = self.grasp_ee_arr[-1][7:10]

        left_delta = left_ee_grasp_pos - left_ee_grasp_pos_prev
        right_delta = right_ee_grasp_pos - right_ee_grasp_pos_prev

        # Generate a smooth interpolation factor from 0 to 1
        n_waypoints = self.grasp_ee_arr.shape[0]
        interpolation_factors = np.linspace(0, 1, n_waypoints)

        # Apply the interpolation to adjust the trajectory
        grasp_ee_arr_updated = self.grasp_ee_arr.copy()
        for i in range(1, n_waypoints):  # Skip the first waypoint to keep it fixed
            grasp_ee_arr_updated[i][:3] += left_delta * interpolation_factors[i]
            grasp_ee_arr_updated[i][7:10] += right_delta * interpolation_factors[i]

        # Visualize the trajectories
        visualize_ee_trajectory(self.grasp_ee_arr, grasp_ee_arr_updated)

        self.grasp_ee_arr = grasp_ee_arr_updated

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        if not self.grasp_motion_updated:
            self.update_grasp_ee_traj(obs)
            self.grasp_motion_updated = True

        control_inputs, motor_target = super().step(obs, is_real)

        return control_inputs, motor_target
