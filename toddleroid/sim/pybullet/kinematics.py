import pybullet as p

from toddleroid.sim.robot import HumanoidRobot
from toddleroid.utils.data_utils import round_floats


class Kinematics:
    def __init__(self, robot):
        """
        Initialize Kinematics for a given robot.

        Args:
            robot_id (int): Unique ID of the robot in PyBullet simulation.
        """
        self.robot = robot
        self.end_effector_indices = self._get_end_effector_indices()

    def _get_end_effector_indices(self):
        """
        Retrieves the indices of the end effectors (feet).

        Returns:
            dict: A dictionary with keys 'left' and 'right' and joint indices as values.
        """
        name2indices = {}
        for name, name_in_urdf in self.robot.end_effector_names.items():
            for id in range(p.getNumJoints(self.robot.id)):
                if (
                    p.getJointInfo(self.robot.id, id)[12].decode("UTF-8")
                    == name_in_urdf
                ):
                    name2indices[name] = id
                    break
        return name2indices

    def solve_ik(
        self, target_pos_left, target_ori_left, target_pos_right, target_ori_right
    ):
        """
        Solves inverse kinematics using PyBullet for the given foot positions and orientations.

        Args:
            target_pos_left (list/tuple): Target position for the left foot (x, y, z).
            target_ori_left (list/tuple): Target orientation for the left foot (quaternion).
            target_pos_right (list/tuple): Target position for the right foot (x, y, z).
            target_ori_right (list/tuple): Target orientation for the right foot (quaternion).

        Returns:
            list: New joint angles to achieve the desired foot positions and orientations.
        """
        left_foot_index = self.end_effector_indices["left_foot_link"]
        right_foot_index = self.end_effector_indices["right_foot_link"]

        joint_angles_left = p.calculateInverseKinematics(
            self.robot.id, left_foot_index, target_pos_left, target_ori_left
        )
        joint_angles_right = p.calculateInverseKinematics(
            self.robot.id, right_foot_index, target_pos_right, target_ori_right
        )

        return joint_angles_left, joint_angles_right


# Example usage
if __name__ == "__main__":
    from toddleroid.sim.pybullet.simulation import PyBulletSim

    sim = PyBulletSim()
    robot = HumanoidRobot("Robotis_OP3")
    sim.load_robot(robot)

    kinematics = Kinematics(robot)

    # Define target positions and orientations for left and right feet
    target_pos_left, target_ori_left = [0.2, 0.1, -0.2], [0, 0, 0, 1]
    target_pos_right, target_ori_right = [0.2, -0.1, -0.2], [0, 0, 0, 1]

    joint_angles_left, joint_angles_right = kinematics.solve_ik(
        target_pos_left, target_ori_left, target_pos_right, target_ori_right
    )

    print(
        f"Left Foot Joint Angles {len(joint_angles_left)}: {round_floats(joint_angles_left, 3)}"
    )
    print(
        f"Right Foot Joint Angles {len(joint_angles_right)}: {round_floats(joint_angles_right, 3)}"
    )
