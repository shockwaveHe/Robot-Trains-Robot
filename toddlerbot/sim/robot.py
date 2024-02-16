import copy
import math
from typing import Dict, List, Tuple

import numpy as np

from toddlerbot.robot_descriptions.robot_configs import robot_configs


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

        self._solve_leg_ik(
            target_left_foot_pos, target_left_foot_ori, "left", joint_angles
        )
        self._solve_leg_ik(
            target_right_foot_pos, target_right_foot_ori, "right", joint_angles
        )
        return joint_angles

    def _solve_leg_ik(
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
        # Calculate leg angles
        angles_dict = self.config.compute_leg_angles(
            target_foot_pos, target_foot_ori, side, self.config.offsets
        )

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

    from toddlerbot.sim.pybullet_sim import PyBulletSim
    from toddlerbot.utils.data_utils import round_floats

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
