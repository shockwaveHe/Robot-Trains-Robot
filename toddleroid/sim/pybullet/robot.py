import math
from typing import List

import pybullet as p

from toddleroid.robot_descriptions.config import robots_config
from toddleroid.utils.data_utils import round_floats
from toddleroid.utils.file_utils import find_urdf_path


# TODO: Condier bringing kinematics.py into this file.
class HumanoidRobot:
    def __init__(self, robot_name: str):
        self.name = robot_name
        if robot_name not in robots_config:
            raise ValueError(f"Robot '{robot_name}' is not supported.")
        self.config = robots_config[robot_name]
        self.id = None

    def get_urdf_path(self):
        """
        Returns the path to the robot's URDF file.
        """
        return find_urdf_path(self.name)

    @property
    def foot_names(self):
        """
        Returns the names of the end effectors.
        """
        return {
            "left_foot_link": self.config.left_foot_link,
            "right_foot_link": self.config.right_foot_link,
        }

    def _get_foot_indices(self):
        """
        Retrieves the indices of the end effectors (feet).

        Returns:
            dict: A dictionary with keys 'left' and 'right' and joint indices as values.
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
        target_left_foot_pos: List,
        target_left_foot_ori: List,
        target_right_foot_pos: List,
        target_right_foot_ori: List,
        current_angles: List = None,
    ):
        """
        Solves inverse kinematics using PyBullet for the given foot positions and orientations.

        Args:
            target_left_foot_pos (list): Target position for the left foot (x, y, z).
            target_left_foot_ori (list): Target orientation for the left foot (quaternion).
            target_right_foot_pos (list): Target position for the right foot (x, y, z).
            target_right_foot_ori (list): Target orientation for the right foot (quaternion).

        Returns:
            list: New joint angles to achieve the desired foot positions and orientations.
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
        self, target_foot_pos, target_foot_ori, side, joint_angles
    ):
        L1 = self.config.magic_numbers["L1"]
        L12 = self.config.magic_numbers["L12"]
        L2 = self.config.magic_numbers["L2"]
        L3 = self.config.magic_numbers["L3"]
        OFFSET_W = self.config.magic_numbers["OFFSET_W"]
        OFFSET_X = self.config.magic_numbers["OFFSET_X"]

        x, y, z = target_foot_pos
        ankle_roll, ankle_pitch, waist_yaw = target_foot_ori
        x -= OFFSET_X
        y += -OFFSET_W if side == "left" else OFFSET_W
        z = L1 + L12 + L2 + L3 - z

        x2 = x * math.cos(waist_yaw) + y * math.sin(waist_yaw)
        y2 = -x * math.sin(waist_yaw) + y * math.cos(waist_yaw)
        z2 = z - L3

        waist_roll = math.atan2(y2, z2)
        l2 = y2**2 + z2**2
        z3 = math.sqrt(max(l2 - x2**2, 0.0)) - L12
        pitch_angle = math.atan2(x2, z3)
        l = math.sqrt(x2**2 + z3**2)
        knee_disp = math.acos(min(max(l / (2.0 * L1), -1.0), 1.0))
        waist_pitch = -pitch_angle - knee_disp
        knee_pitch = -pitch_angle + knee_disp

        return self._set_joint_angles(
            side,
            waist_yaw,
            waist_roll,
            waist_pitch,
            knee_pitch,
            ankle_roll,
            ankle_pitch,
            joint_angles,
        )

    def _set_joint_angles(
        self,
        side,
        waist_yaw,
        waist_roll,
        waist_pitch,
        knee_pitch,
        ankle_roll,
        ankle_pitch,
        joint_angles,
    ):
        index_dof = {p.getBodyInfo(self.id)[0].decode("UTF-8"): -1}
        for id in range(p.getNumJoints(self.id)):
            joint_name = p.getJointInfo(self.id, id)[12].decode("UTF-8")
            index_dof[joint_name] = p.getJointInfo(self.id, id)[3] - 7

        prefix = "left" if side == "left" else "right"

        joint_names = [
            "waist_yaw_link",
            "waist_roll_link",
            "waist_pitch_link",
            "knee_pitch_link",
            "waist_pitch_mimic_link",
            "shin_pitch_link",
            "independent_pitch_link",
            "shin_pitch_mimic_link",
            "ankle_pitch_link",
            "ankle_roll_link",
        ]

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

        for joint_name, angle in zip(joint_names, angles):
            joint_index = index_dof[f"{prefix}_{joint_name}"]
            if joint_index >= 0:  # Check if joint index is valid
                joint_angles[joint_index] = angle

        return joint_angles


# Example usage
if __name__ == "__main__":
    from toddleroid.sim.pybullet.simulation import PyBulletSim

    sim = PyBulletSim()
    robot = HumanoidRobot("Sustaina_OP")
    sim.load_robot(robot)

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

    print(f"Joint Angles {len(joint_angles)}: {round_floats(joint_angles, 3)}")
