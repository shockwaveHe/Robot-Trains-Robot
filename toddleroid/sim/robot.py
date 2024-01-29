import math
from typing import Dict, List, Tuple

import numpy as np
import pybullet as p

from toddleroid.robot_descriptions.robot_configs import robot_configs


class HumanoidRobot:
    """Class representing a humanoid robot."""

    def __init__(self, robot_name: str):
        """
        Initialize a humanoid robot with the given name.

        Args:
            robot_name (str): The name of the robot.

        Raises:
            ValueError: If the robot name is not supported.
        """
        self.name = robot_name
        if robot_name not in robot_configs:
            raise ValueError(f"Robot '{robot_name}' is not supported.")
        self.config = robot_configs[robot_name]
        self.id = None

    @property
    def foot_names(self) -> Dict[str, str]:
        """
        Returns the names of the end effectors (feet).

        Returns:
            Dict[str, str]: Dictionary with keys 'left_foot_link' and 'right_foot_link' and their names.
        """
        foot_keys = ["left_foot_link", "right_foot_link"]
        return {
            k: self.config.canonical_name2link_name[k]
            for k in foot_keys
            if k in self.config.canonical_name2link_name
        }

    def _get_foot_indices(self) -> Dict[str, int]:
        """
        Retrieves the indices of the end effectors (feet).

        Returns:
            Dict[str, int]: Dictionary with keys 'left' and 'right' and joint indices as values.
        """
        name2indices = {}
        for name, name_in_urdf in self.foot_names.items():
            for id in range(p.getNumJoints(self.id)):
                if p.getJointInfo(self.id, id)[12].decode("UTF-8") == name_in_urdf:
                    name2indices[name] = id
                    break

        return name2indices

    def solve_ik(
        self,
        target_left_foot_pos: List[float],
        target_left_foot_ori: List[float],
        target_right_foot_pos: List[float],
        target_right_foot_ori: List[float],
        current_angles: List[float] = None,
    ) -> List[float]:
        """
        Solves inverse kinematics for the given foot positions and orientations.

        Args:
            target_left_foot_pos (List[float]): Target position for the left foot (x, y, z).
            target_left_foot_ori (List[float]): Target orientation for the left foot (quaternion).
            target_right_foot_pos (List[float]): Target position for the right foot (x, y, z).
            target_right_foot_ori (List[float]): Target orientation for the right foot (quaternion).
            current_angles (List[float], optional): Current joint angles. Defaults to None.

        Returns:
            List[float]: New joint angles to achieve the desired foot positions and orientations.

        Raises:
            ValueError: If the robot has not been loaded yet.
        """
        if self.id is None:
            raise ValueError("Robot has not been loaded yet.")

        # end_effector_indices = self._get_end_effector_indices()
        # left_foot_index = end_effector_indices["left_foot_link"]
        # right_foot_index = end_effector_indices["right_foot_link"]

        # joint_angles = p.calculateInverseKinematics2(
        #     self.id,
        #     [left_foot_index, right_foot_index],
        #     [target_left_foot_pos, target_right_foot_pos],
        #     [target_left_foot_ori, target_right_foot_ori],
        #     maxNumIterations=1000,

        if current_angles is None:
            joint_angles = []
            for id in range(p.getNumJoints(self.id)):
                if p.getJointInfo(self.id, id)[3] > -1:
                    joint_angles += [0]
        else:
            joint_angles = current_angles.copy()

        self._calculate_leg_angles(
            target_left_foot_pos, target_left_foot_ori, "left", joint_angles
        )
        self._calculate_leg_angles(
            target_right_foot_pos, target_right_foot_ori, "right", joint_angles
        )
        return joint_angles

    def _calculate_leg_angles(
        self,
        target_foot_pos: Tuple[float, float, float],
        target_foot_ori: Tuple[float, float, float],
        side: str,
        joint_angles: List[float],
    ) -> List[float]:
        """
        Calculates the leg angles based on the target foot position and orientation.

        Args:
            target_foot_pos (Tuple[float, float, float]): Target position for the foot (x, y, z).
            target_foot_ori (Tuple[float, float, float]): Target orientation for the foot (roll, pitch, yaw).
            side (str): The side of the leg ('left' or 'right').
            joint_angles (List[float]): Current list of joint angles.

        Returns:
            List[float]: Updated list of joint angles after calculation.
        """

        # Initialize dictionary to map joint names to their degrees of freedom index
        # TODO: Update this to use joint name instead of link_name
        index_dof = {p.getBodyInfo(self.id)[0].decode("UTF-8"): -1}
        for idx in range(p.getNumJoints(self.id)):
            joint_name = p.getJointInfo(self.id, idx)[1].decode("UTF-8")
            index_dof[joint_name] = p.getJointInfo(self.id, idx)[3] - 7

        # Define joint names and their corresponding angles
        n_dof = len(self.config.joint_names)
        if side == "left":
            joint_names = self.config.joint_names[: n_dof // 2]
        else:
            joint_names = self.config.joint_names[n_dof // 2 :]

        # Magic numbers extracted from robot configuration
        (
            z_offset_thigh,
            z_offset_knee,
            z_offset_shin,
            x_offset_foot_to_ankle,
            y_offset_foot_to_ankle,
        ) = (
            self.config.offsets[key]
            for key in [
                "z_offset_thigh",
                "z_offset_knee",
                "z_offset_shin",
                "x_offset_foot_to_ankle",
                "y_offset_foot_to_ankle",
            ]
        )

        # Decompose target position and orientation
        target_x, target_y, target_z = target_foot_pos
        ankle_roll, ankle_pitch, waist_yaw = target_foot_ori

        # Adjust positions based on offsets and calculate new coordinates
        target_x += x_offset_foot_to_ankle
        target_y += (
            -y_offset_foot_to_ankle if side == "left" else y_offset_foot_to_ankle
        )
        target_z = z_offset_thigh + z_offset_knee + z_offset_shin - target_z

        transformed_x = target_x * math.cos(waist_yaw) + target_y * math.sin(waist_yaw)
        transformed_y = -target_x * math.sin(waist_yaw) + target_y * math.cos(waist_yaw)
        transformed_z = target_z

        waist_roll = math.atan2(transformed_y, transformed_z)

        # Calculate leg angles
        if self.name == "sustaina_op":
            adjusted_leg_height_sq = (
                transformed_y**2 + transformed_z**2 - transformed_x**2
            )
            adjusted_leg_height = (
                math.sqrt(max(0.0, adjusted_leg_height_sq)) - z_offset_knee
            )
            leg_pitch = math.atan2(transformed_x, adjusted_leg_height)
            leg_length = math.sqrt(transformed_x**2 + adjusted_leg_height**2)
            knee_disp = math.acos(
                min(max(leg_length / (z_offset_thigh + z_offset_shin), -1.0), 1.0)
            )
            waist_pitch = -leg_pitch - knee_disp
            knee_pitch = -leg_pitch + knee_disp

            angles = [
                waist_yaw,
                waist_roll,
                waist_pitch,
                -waist_pitch,
                waist_pitch,
                knee_pitch,
                -knee_pitch,
                knee_pitch,
                ankle_pitch,
                ankle_roll - waist_roll,
            ]
        elif self.name == "robotis_op3":
            leg_projected_yz_length = math.sqrt(transformed_y**2 + transformed_z**2)
            leg_length = math.sqrt(transformed_x**2 + leg_projected_yz_length**2)
            leg_pitch = math.atan2(transformed_x, leg_projected_yz_length)
            wrist_disp_cos = (
                leg_length**2 + z_offset_shin**2 - z_offset_thigh**2
            ) / (2 * leg_length * z_offset_shin)
            wrist_disp = math.acos(min(max(wrist_disp_cos, -1.0), 1.0))
            ankle_disp = math.asin(
                z_offset_thigh / z_offset_shin * math.sin(wrist_disp)
            )
            waist_pitch = -leg_pitch - wrist_disp
            knee_pitch = wrist_disp + ankle_disp
            ankle_pitch += knee_pitch + waist_pitch

            angles = [
                waist_yaw,
                -waist_roll,
                waist_pitch if side == "left" else -waist_pitch,
                knee_pitch if side == "left" else -knee_pitch,
                ankle_pitch if side == "left" else -ankle_pitch,
                ankle_roll - waist_roll,
            ]
        else:
            raise ValueError(f"Robot '{self.name}' is not supported.")

        # Update joint angles based on calculations
        for joint_name, angle in zip(joint_names, angles):
            joint_index = index_dof[joint_name]
            if joint_index >= 0:  # Check if joint index is valid
                joint_angles[joint_index] = angle

        return joint_angles


# Example usage
if __name__ == "__main__":
    import numpy as np

    from toddleroid.sim.pybullet_sim import PyBulletSim
    from toddleroid.utils.data_utils import round_floats

    sim = PyBulletSim()
    robot = HumanoidRobot("sustaina_op")
    sim.load_robot(robot)
    sim.put_robot_on_ground(robot)

    # Define target positions and orientations for left and right feet
    target_left_foot_pos, target_left_foot_ori = [0.2, 0.1, -0.2], [
        0,
        0,
        0,
    ]
    target_right_foot_pos, target_right_foot_ori = [0.2, -0.1, -0.2], [0, 0, 0]

    joint_angles = robot.solve_ik(
        target_left_foot_pos,
        target_left_foot_ori,
        target_right_foot_pos,
        target_right_foot_ori,
    )

    rounded_joint_angles = round_floats(joint_angles, 3)
    print(f"Joint angles: {rounded_joint_angles}")

    expected_joint_angles = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.122,
            0,
            -0.51,
            0.51,
            -0.51,
            0.51,
            0,
            -0.122,
            -0.51,
            -0.51,
            0,
            -0.122,
            0,
            -0.51,
            0.51,
            -0.51,
            0.51,
            0,
            0.122,
            -0.51,
            -0.51,
        ]
    )

    assert np.allclose(
        rounded_joint_angles, expected_joint_angles
    ), "Joint Angles does not match expected values."
