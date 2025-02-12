import os
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import numpy.typing as npt

from toddlerbot.policies import BasePolicy
from toddlerbot.sensing.camera import Camera
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import euler2mat, interpolate_action, quat2euler

# from toddlerbot.utils.misc_utils import profile


class PullUpPolicy(BasePolicy, policy_name="pull_up"):
    """Policy for pulling up the robot."""

    def __init__(
        self, name: str, robot: Robot, init_motor_pos: npt.NDArray[np.float32]
    ):
        """Initializes the object with specified parameters and sets up camera and motion data.

        Args:
            name (str): The name of the object.
            robot (Robot): The robot instance associated with this object.
            init_motor_pos (npt.NDArray[np.float32]): Initial motor positions.

        Raises:
            ValueError: If required motion data files are not found.
        """
        super().__init__(name, robot, init_motor_pos)

        self.left_eye = None
        self.right_eye = None
        try:
            self.left_eye = Camera("left")
            self.right_eye = Camera("right")
        except Exception:
            pass

        self.root_to_left_sho_pitch = np.array(
            [-0.0035, 0.07, 0.1042], dtype=np.float32
        )
        self.root_to_right_sho_pitch = np.array(
            [-0.0035, -0.07, 0.1042], dtype=np.float32
        )
        self.elbow_roll_to_sho_pitch = 0.0876
        self.wrist_pitch_to_elbow_roll = 0.0806
        self.ee_center_to_wrist_pitch = 0.045

        self.root_to_left_eye_t = np.array([0.032, 0.017, 0.19], dtype=np.float32)
        self.root_to_neck_t = np.array([0.016, 0.0, 0.1419], dtype=np.float32)
        self.waist_motor_indices = np.array(
            [
                robot.motor_ordering.index("waist_act_1"),
                robot.motor_ordering.index("waist_act_2"),
            ]
        )
        self.neck_pitch_idx = robot.motor_ordering.index("neck_pitch_act")
        self.neck_pitch_vel = np.pi / 16
        self.neck_pitch_act_pos = 0.0
        self.tag_pose_avg: Optional[npt.NDArray[np.float32]] = None

        self.grasp_motion_updated = False
        grasp_motion_path = os.path.join("motion", "pull_up_grasp.pkl")
        if os.path.exists(grasp_motion_path):
            grasp_data_dict = joblib.load(grasp_motion_path)
        else:
            raise ValueError(f"No data files found in {grasp_motion_path}")

        grasp_time_arr = np.array(grasp_data_dict["time"], dtype=np.float32)
        grasp_action_arr = np.array(grasp_data_dict["action_traj"], dtype=np.float32)
        grasp_ee_arr = np.array(grasp_data_dict["ee_traj"], dtype=np.float32)
        grasp_root_arr = np.array(grasp_data_dict["root_traj"], dtype=np.float32)

        prepare_idx = int(1.0 / self.control_dt)
        self.prepare_action = grasp_action_arr[prepare_idx].copy()
        self.prepared_time = 0.0

        self.grasp_time_arr = grasp_time_arr[prepare_idx:]
        self.grasp_action_arr = grasp_action_arr[prepare_idx:]
        self.grasp_ee_arr = grasp_ee_arr[prepare_idx:]
        self.grasp_root_arr = grasp_root_arr[prepare_idx:]

        self.last_action = np.zeros(robot.nu, dtype=np.float32)
        self.grasped_count = 0
        self.grasped_time = 0.0
        self.grasped_action = np.zeros(robot.nu, dtype=np.float32)

        pull_motion_path = os.path.join("motion", "pull_up_pull.pkl")
        if os.path.exists(pull_motion_path):
            pull_data_dict = joblib.load(pull_motion_path)
        else:
            raise ValueError(f"No data files found in {pull_motion_path}")

        self.pull_time_arr = np.array(pull_data_dict["time"])
        self.pull_action_arr = np.array(pull_data_dict["action_traj"], dtype=np.float32)

        self.step_curr = 0

        self.is_prepared = False

    def search_for_tag(self, obs: Obs):
        """Searches for a tag using the robot's sensors and updates the robot's action based on detected tag positions.

        Args:
            obs (Obs): An observation object containing the current state of the robot, including time and motor positions.

        Returns:
            np.ndarray: An action array with updated neck pitch position based on the detected tag.
        """
        neck_pitch_act_pos = np.clip(
            self.neck_pitch_vel * (obs.time - self.prep_duration),
            self.robot.joint_limits["neck_pitch"][0],
            self.robot.joint_limits["neck_pitch"][1],
        )
        action = self.prepare_action.copy()
        action[self.neck_pitch_idx] = neck_pitch_act_pos

        assert self.left_eye is not None and self.right_eye is not None
        left_tag_poses = self.left_eye.detect_tags()
        right_tag_poses = self.right_eye.detect_tags()

        if len(left_tag_poses) > 0 and len(right_tag_poses) > 0:
            joint_angles = self.robot.motor_to_joint_angles(
                dict(zip(self.robot.motor_ordering, obs.motor_pos))
            )
            neck_yaw_pos = joint_angles["neck_yaw_driven"]
            neck_pitch_pos = joint_angles["neck_pitch"]
            head_R = euler2mat(
                np.array([0, -neck_pitch_pos, neck_yaw_pos], dtype=np.float32)
            )
            left_eye_transform = np.eye(4, dtype=np.float32)
            left_eye_transform[:3, :3] = head_R
            left_eye_transform[:3, 3] = (
                head_R @ (self.root_to_left_eye_t - self.root_to_neck_t)
                + self.root_to_neck_t
            )

            tag_id = list(left_tag_poses.keys())[0]
            self.tag_pose_avg = left_eye_transform @ np.mean(
                [left_tag_poses[tag_id], right_tag_poses[tag_id]], axis=0
            )
            self.neck_pitch_act_pos = neck_pitch_act_pos

        return action

    def update_grasp_traj(
        self, grasp_delta_x: float = -0.01, grasp_delta_z: float = 0.0
    ):
        """Updates the grasp trajectory by adjusting the end-effector positions and corresponding motor actions based on specified deltas.

        Args:
            grasp_delta_x (float): The change in the x-coordinate for the grasp position. Default is -0.01.
            grasp_delta_z (float): The change in the z-coordinate for the grasp position. Default is 0.0.

        Raises:
            AssertionError: If `self.tag_pose_avg` is None.

        Modifies:
            self.grasp_ee_arr: Updates the end-effector trajectory with interpolated adjustments.
            self.grasp_action_arr: Updates the motor actions to reflect the new trajectory.
        """
        assert self.tag_pose_avg is not None

        left_delta = np.array(
            [
                self.grasp_root_arr[-1][0]
                + self.tag_pose_avg[0, 3]
                + grasp_delta_x
                - self.grasp_ee_arr[-1][0],
                0,
                self.grasp_root_arr[-1][2]
                + self.tag_pose_avg[2, 3]
                + grasp_delta_z
                - self.grasp_ee_arr[-1][2],
            ],
            dtype=np.float32,
        )
        right_delta = np.array(
            [
                self.grasp_root_arr[-1][0]
                + self.tag_pose_avg[0, 3]
                + grasp_delta_x
                - self.grasp_ee_arr[-1][7],
                0,
                self.grasp_root_arr[-1][2]
                + self.tag_pose_avg[2, 3]
                + grasp_delta_z
                - self.grasp_ee_arr[-1][9],
            ],
            dtype=np.float32,
        )

        # Generate a smooth interpolation factor from 0 to 1
        n_waypoints = self.grasp_ee_arr.shape[0]
        interpolation_factors = np.linspace(0, 1, n_waypoints)

        # Apply the interpolation to adjust the trajectory
        grasp_ee_arr_updated = self.grasp_ee_arr.copy()
        for i in range(1, n_waypoints):  # Skip the first waypoint to keep it fixed
            grasp_ee_arr_updated[i][:3] += left_delta * interpolation_factors[i]
            grasp_ee_arr_updated[i][7:10] += right_delta * interpolation_factors[i]

        # import matplotlib.pyplot as plt

        # # Plot left arm trajectory
        # plt.plot(
        #     self.grasp_ee_arr[:, 0],
        #     self.grasp_ee_arr[:, 2],
        #     label="Left Arm Original",
        #     linestyle="--",
        # )
        # plt.plot(
        #     grasp_ee_arr_updated[:, 0],
        #     grasp_ee_arr_updated[:, 2],
        #     label="Left Arm Updated",
        # )

        # # Plot right arm trajectory
        # plt.plot(
        #     self.grasp_ee_arr[:, 7],
        #     self.grasp_ee_arr[:, 9],
        #     label="Right Arm Original",
        #     linestyle="--",
        # )
        # plt.plot(
        #     grasp_ee_arr_updated[:, 7],
        #     grasp_ee_arr_updated[:, 9],
        #     label="Right Arm Updated",
        # )

        # # Add labels, legend, and grid
        # plt.xlabel("X Coordinate")
        # plt.ylabel("Z Coordinate")
        # plt.title("End-Effector Trajectories for Left and Right Arms")
        # plt.legend()
        # plt.grid(True)

        # # Save the plot to a file
        # plt.savefig("test_both_arms.png")

        torso_pitch = quat2euler(self.grasp_root_arr[-1][3:])[1].item()
        root_to_left_ee_x = grasp_ee_arr_updated[-1][0] - self.grasp_root_arr[-1][0]
        root_to_left_ee_z = grasp_ee_arr_updated[-1][2] - self.grasp_root_arr[-1][2]

        root_to_left_sho_pitch_x = self.root_to_left_sho_pitch[0] * np.cos(
            -torso_pitch
        ) - self.root_to_left_sho_pitch[2] * np.sin(-torso_pitch)
        root_to_left_sho_pitch_z = self.root_to_left_sho_pitch[0] * np.sin(
            -torso_pitch
        ) + self.root_to_left_sho_pitch[2] * np.cos(-torso_pitch)

        left_ee_x_target = root_to_left_ee_x - root_to_left_sho_pitch_x
        left_ee_z_target = root_to_left_ee_z - root_to_left_sho_pitch_z

        left_ee_x_target_rotated = left_ee_x_target * np.cos(
            torso_pitch
        ) - left_ee_z_target * np.sin(torso_pitch)
        left_ee_z_target_rotated = left_ee_x_target * np.sin(
            torso_pitch
        ) + left_ee_z_target * np.cos(torso_pitch)

        root_to_right_ee_x = grasp_ee_arr_updated[-1][7] - self.grasp_root_arr[-1][0]
        root_to_right_ee_z = grasp_ee_arr_updated[-1][9] - self.grasp_root_arr[-1][2]
        root_to_right_sho_pitch_x = self.root_to_right_sho_pitch[0] * np.cos(
            -torso_pitch
        ) - self.root_to_right_sho_pitch[2] * np.sin(-torso_pitch)
        root_to_right_sho_pitch_z = self.root_to_right_sho_pitch[0] * np.sin(
            -torso_pitch
        ) + self.root_to_right_sho_pitch[2] * np.cos(-torso_pitch)

        right_ee_x_target = root_to_right_ee_x - root_to_right_sho_pitch_x
        right_ee_z_target = root_to_right_ee_z - root_to_right_sho_pitch_z

        right_ee_x_target_rotated = right_ee_x_target * np.cos(
            torso_pitch
        ) - right_ee_z_target * np.sin(torso_pitch)
        right_ee_z_target_rotated = right_ee_x_target * np.sin(
            torso_pitch
        ) + right_ee_z_target * np.cos(torso_pitch)

        ee_x_target_avg = (left_ee_x_target_rotated + right_ee_x_target_rotated) / 2
        ee_z_target_avg = (left_ee_z_target_rotated + right_ee_z_target_rotated) / 2
        ee_pitch_target = -np.pi / 2 - torso_pitch

        left_arm_motor_pos = self.arm_ik(
            ee_x_target_avg, ee_z_target_avg, ee_pitch_target, "left"
        )
        right_arm_motor_pos = self.arm_ik(
            ee_x_target_avg, ee_z_target_avg, ee_pitch_target, "right"
        )

        arm_motor_pos = np.concatenate(
            [left_arm_motor_pos, right_arm_motor_pos, np.zeros(2, dtype=np.float32)]
        )
        arm_action_delta = (
            arm_motor_pos - self.grasp_action_arr[-1][self.arm_motor_indices]
        )

        # Apply the interpolation to adjust the trajectory
        grasp_action_arr_updated = self.grasp_action_arr.copy()
        for i in range(1, n_waypoints):  # Skip the first waypoint to keep it fixed
            grasp_action_arr_updated[i][self.arm_motor_indices] += (
                arm_action_delta * interpolation_factors[i]
            )

        self.grasp_ee_arr = grasp_ee_arr_updated
        self.grasp_action_arr = grasp_action_arr_updated

        # print("Grasp trajectory updated!")

    def arm_ik(
        self, x: float, z: float, pitch: float, side: str
    ) -> npt.NDArray[np.float32]:
        """Calculates the inverse kinematics for a robotic arm to determine joint angles based on target position and orientation.

        Args:
            x (float): The target x-coordinate in the arm's plane.
            z (float): The target z-coordinate in the arm's plane.
            pitch (float): The desired end-effector pitch angle in radians.
            side (str): The side of the arm ('left' or 'right').

        Returns:
            npt.NDArray[np.float32]: An array of joint angles for the arm motors, in radians.
        """
        L1 = self.elbow_roll_to_sho_pitch
        L2 = self.wrist_pitch_to_elbow_roll
        L3 = self.ee_center_to_wrist_pitch

        d = np.sqrt(x**2 + z**2)
        if d > L1 + L2 + L3:
            # Project target onto reachable boundary
            x *= (L1 + L2 + L3) / d
            z *= (L1 + L2 + L3) / d
            d = L1 + L2 + L3

        L3_vec = np.array([L3 * np.cos(pitch), -L3 * np.sin(pitch)], dtype=np.float32)
        wrist_vec = np.array([x, z], dtype=np.float32) - L3_vec

        # Solve for θ2
        cos_theta2 = (np.linalg.norm(wrist_vec) ** 2 - L1**2 - L2**2) / (2 * L1 * L2)
        theta2 = np.arccos(np.clip(cos_theta2, -1, 1))

        # Solve for θ1
        phi1 = np.arctan2(wrist_vec[0], -wrist_vec[1])
        phi2 = np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
        theta1 = phi1 - phi2

        # Solve for θ3
        theta3 = -pitch + np.pi / 2 - (theta1 + theta2)

        arm_motor_pos = np.array(
            [
                -theta1 if side == "left" else theta1,
                0.0,
                np.pi / 2 if side == "left" else -np.pi / 2,
                theta2,
                -np.pi / 2 if side == "left" else np.pi / 2,
                -theta3 if side == "left" else theta3,
                0.0,
            ],
            dtype=np.float32,
        )
        return arm_motor_pos

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        """Executes a step in the control loop, determining the appropriate action based on the current observation and state.

        Args:
            obs (Obs): The current observation containing time and other relevant data.
            is_real (bool, optional): Flag indicating whether the operation is in a real environment. Defaults to False.

        Returns:
            Tuple[Dict[str, float], npt.NDArray[np.float32]]: A tuple containing an empty dictionary and the computed action as a NumPy array.
        """
        if not self.is_prepared:
            self.is_prepared = True
            self.prep_duration = 10.0 if is_real else 2.0
            self.prep_time, self.prep_action = self.move(
                -self.control_dt,
                self.init_motor_pos,
                self.prepare_action,
                self.prep_duration,
                end_time=8.0 if is_real else 0.0,
            )

        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return {}, action

        if is_real:
            if self.tag_pose_avg is None:
                action = self.search_for_tag(obs)

                return {}, action
        else:
            left_eye_transform = np.eye(4, dtype=np.float32)
            left_eye_transform[:3, 3] = self.root_to_left_eye_t
            tag_pose_avg = np.eye(4, dtype=np.float32)
            tag_pose_avg[:3, 3] = np.array([0.05, -0.017, 0.05], dtype=np.float32)
            self.tag_pose_avg = left_eye_transform @ tag_pose_avg

        if not self.grasp_motion_updated:
            self.update_grasp_traj()
            self.grasp_motion_updated = True

        if self.grasped_count < 1 / self.control_dt:
            if self.prepared_time == 0:
                self.prepared_time = obs.time

            # Calculate the interpolation factor
            time_elapsed = obs.time - self.prepared_time
            # Find the closest pull action index
            curr_idx = np.argmin(np.abs(self.grasp_time_arr - time_elapsed))
            grasp_action = self.grasp_action_arr[curr_idx]
            grasp_action[self.neck_pitch_idx] = self.neck_pitch_act_pos
            grasp_action[self.waist_motor_indices] = 0.0

            if curr_idx == len(self.grasp_time_arr) - 1:
                self.grasped_count += 1

            self.last_action = grasp_action

            return {}, grasp_action

        else:
            if self.grasped_time == 0:
                self.grasped_time = obs.time
                self.grasped_action = self.last_action

            # Calculate the interpolation factor
            time_elapsed = obs.time - self.grasped_time
            interp_duration = 1.0  # Duration in seconds for the transition
            interp_factor = min(time_elapsed / interp_duration, 1.0)  # Clamp to [0, 1]
            # Find the closest pull action index
            curr_idx = np.argmin(np.abs(self.pull_time_arr - time_elapsed))
            pull_action = (
                1 - interp_factor
            ) * self.grasped_action + interp_factor * self.pull_action_arr[curr_idx]
            pull_action[self.waist_motor_indices] = 0.0

            self.last_action = pull_action

            return {}, pull_action
