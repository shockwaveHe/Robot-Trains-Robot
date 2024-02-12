import copy
import math
from typing import Dict, List, Tuple

import numpy as np

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
        self.joints_info = None

    def solve_ik(
        self,
        target_left_foot_pos: List[float],
        target_left_foot_ori: List[float],
        target_right_foot_pos: List[float],
        target_right_foot_ori: List[float],
        joint_angles_curr: List[float],
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
        if self.id is None or self.joints_info is None:
            raise ValueError("Robot has not been loaded yet.")

        joint_angles = copy.deepcopy(joint_angles_curr)

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
        joint_angles: Dict[str, float],
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
        # Magic numbers extracted from robot configuration
        (
            z_offset_thigh,
            z_offset_knee,
            z_offset_shin,
            x_offset_ankle_to_foot,
            y_offset_ankle_to_foot,
        ) = (
            self.config.offsets[key]
            for key in [
                "z_offset_thigh",
                "z_offset_knee",
                "z_offset_shin",
                "x_offset_ankle_to_foot",
                "y_offset_ankle_to_foot",
            ]
        )

        # Decompose target position and orientation
        target_x, target_y, target_z = target_foot_pos
        ankle_roll, ankle_pitch, hip_yaw = target_foot_ori

        # Adjust positions based on offsets and calculate new coordinates
        target_x += x_offset_ankle_to_foot
        target_y += (
            -y_offset_ankle_to_foot if side == "left" else y_offset_ankle_to_foot
        )
        target_z = z_offset_thigh + z_offset_knee + z_offset_shin - target_z

        transformed_x = target_x * math.cos(hip_yaw) + target_y * math.sin(hip_yaw)
        transformed_y = -target_x * math.sin(hip_yaw) + target_y * math.cos(hip_yaw)
        transformed_z = target_z

        hip_roll = math.atan2(transformed_y, transformed_z)

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
            hip_pitch = -leg_pitch - knee_disp
            knee_pitch = -leg_pitch + knee_disp

            angles_dict = {
                "waist_yaw_joint": hip_yaw,
                "waist_roll_joint": hip_roll,
                "waist_pitch_joint": hip_pitch,
                "knee_pitch_mimic_joint": -hip_pitch,
                "waist_pitch_mimic_joint": hip_pitch,
                "knee_pitch_joint": knee_pitch,
                "ankle_pitch_mimic_joint": -knee_pitch,
                "shin_pitch_mimic_joint": knee_pitch,
                "ankle_pitch_joint": ankle_pitch,
                "ankle_roll_joint": ankle_roll - hip_roll,
            }
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
            hip_pitch = -leg_pitch - wrist_disp
            knee_pitch = wrist_disp + ankle_disp
            ankle_pitch += knee_pitch + hip_pitch

            angles_dict = {
                "hip_yaw": hip_yaw,
                "hip_roll": -hip_roll,
                "hip_pitch": hip_pitch if side == "left" else -hip_pitch,
                "knee": knee_pitch if side == "left" else -knee_pitch,
                "ank_pitch": ankle_pitch if side == "left" else -ankle_pitch,
                "ank_roll": ankle_roll - hip_roll,
            }
        else:
            raise ValueError(f"Robot '{self.name}' is not supported.")

        # Update joint angles based on calculations
        for name, angle in angles_dict.items():
            if f"{side}_{name}" in joint_angles:
                joint_angles[f"{side}_{name}"] = angle
            elif f"{side[0]}_{name}" in joint_angles:
                joint_angles[f"{side[0]}_{name}"] = angle
            else:
                raise ValueError(f"Joint '{name}' not found in joint angles.")

        return joint_angles


# Example usage
if __name__ == "__main__":
    import numpy as np

    from toddleroid.sim.pybullet_sim import PyBulletSim
    from toddleroid.utils.data_utils import round_floats

    robot = HumanoidRobot("sustaina_op")
    sim = PyBulletSim(robot)

    # Define target positions and orientations for left and right feet
    target_left_foot_pos, target_left_foot_ori = [0.2, 0.1, -0.2], [
        0,
        0,
        0,
    ]
    target_right_foot_pos, target_right_foot_ori = [0.2, -0.1, -0.2], [0, 0, 0]

    # TODO: Get current joint angles from robot
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
