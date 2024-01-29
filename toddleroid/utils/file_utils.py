import os


def find_description_path(robot_name: str, suffix: str = ".urdf") -> str:
    """
    Dynamically finds the URDF file path for a given robot name.

    This function searches for a .urdf file in the directory corresponding to the given robot name.
    It raises a FileNotFoundError if no URDF file is found.

    Args:
        robot_name: The name of the robot (e.g., 'robotis_op3').

    Returns:
        The file path to the robot's URDF file.

    Raises:
        FileNotFoundError: If no URDF file is found in the robot's directory.

    Example:
        robot_urdf_path = find_urdf_path("robotis_op3")
        print(robot_urdf_path)
    """
    robot_dir = os.path.join("toddleroid", "robot_descriptions", robot_name)

    for file in os.listdir(robot_dir):
        if file.endswith(suffix):
            return os.path.join(robot_dir, file)

    raise FileNotFoundError(
        f"No URDF file found in the directory '{robot_dir}' for robot '{robot_name}'."
    )
