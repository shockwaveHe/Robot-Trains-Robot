import os
import platform
from typing import List, Optional

import serial.tools.list_ports as list_ports

from toddlerbot.utils.misc_utils import log


def find_ports(target: str) -> List[str]:
    """Find open network ports on a specified target.

    Args:
        target: The IP address or hostname of the target to scan for open ports.

    Returns:
        A list of strings representing the open ports on the target.
    """
    ports = list(list_ports.comports())
    target_ports: List[str] = []

    os_type = platform.system()

    for port, desc, hwid in ports:
        # Adjust the condition below according to your board's unique identifier or pattern
        print(port, desc, hwid)
        if target in desc:
            if os_type != "Windows":
                port = port.replace("cu", "tty")

            log(
                f"Found {target} board: {port} - {desc} - {hwid}",
                header="FileUtils",
                level="debug",
            )
            target_ports.append(port)

    if len(target_ports) == 0:
        raise ConnectionError(f"Could not find the {target} board.")
    else:
        return sorted(target_ports)


def find_last_result_dir(result_dir: str, prefix: str = "") -> Optional[str]:
    """
    Find the latest (most recent) result directory within a given directory.

    Args:
        result_dir: The path to the directory containing result subdirectories.
        prefix: The prefix of result directory names to consider.

    Returns:
        The path to the latest result directory, or None if no matching directory is found.
    """
    # Get a list of all items in the result directory
    try:
        dir_contents = os.listdir(result_dir)
    except FileNotFoundError:
        print(f"The directory {result_dir} was not found.")
        return None

    # Filter out directories that start with the specified prefix
    result_dirs = [
        d
        for d in dir_contents
        if os.path.isdir(os.path.join(result_dir, d)) and d.startswith(prefix)
    ]

    # Sort the directories based on name, assuming the naming convention includes a sortable date and time
    result_dirs.sort()

    # Return the last directory in the sorted list, if any
    if result_dirs:
        return os.path.join(result_dir, result_dirs[-1])
    else:
        print(f"No directories starting with '{prefix}' were found in {result_dir}.")
        return None


def find_arm_toddler_file_path(
    arm_name: str, toddler_name: str, suffix: str = ".urdf"
) -> str:
    robot_dir = os.path.join("toddlerbot", "descriptions", arm_name)
    if os.path.exists(robot_dir):
        file_path = os.path.join(robot_dir, "_".join([arm_name, toddler_name]) + suffix)
        if os.path.exists(file_path):
            return file_path

    raise FileNotFoundError(
        f"No {suffix} file found for arm '{arm_name}' and robot '{toddler_name}'."
    )


def find_robot_file_path(robot_name: str, suffix: str = ".urdf") -> str:
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
    robot_dir = os.path.join("toddlerbot", "descriptions", robot_name)
    if os.path.exists(robot_dir):
        file_path = os.path.join(robot_dir, robot_name + suffix)
        if os.path.exists(file_path):
            return file_path
    else:
        assembly_dir = os.path.join("toddlerbot", "descriptions", "assemblies")
        file_path = os.path.join(assembly_dir, robot_name + suffix)
        if os.path.exists(file_path):
            return file_path

    raise FileNotFoundError(f"No {suffix} file found for robot '{robot_name}'.")
