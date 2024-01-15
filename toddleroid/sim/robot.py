from toddleroid.robot_descriptions.config import robots_config
from toddleroid.utils.file_utils import find_urdf_path


class HumanoidRobot:
    def __init__(self, robot_name: str):
        self.name = robot_name
        self.config = robots_config.get(robot_name)
        self.id = None

    def get_urdf_path(self):
        """
        Returns the path to the robot's URDF file.
        """
        return find_urdf_path(self.name)

    @property
    def end_effector_names(self):
        """
        Returns the names of the end effectors.
        """
        return {
            "left_foot_link": self.config.left_foot_link,
            "right_foot_link": self.config.right_foot_link,
        }
