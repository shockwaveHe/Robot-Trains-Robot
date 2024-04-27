import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
from typing import Optional

import serial.tools.list_ports as list_ports

from toddlerbot.utils.misc_utils import log


def find_ports(target):
    ports = list(list_ports.comports())
    target_ports = []
    for port, desc, hwid in ports:
        # Adjust the condition below according to your board's unique identifier or pattern
        if target in desc:
            port = port.replace("cu", "tty")
            log(
                f"Found {target} board: {port} - {desc} - {hwid}",
                header="FileUtils",
                level="debug",
            )
            target_ports.append(port)

    if len(target_ports) == 0:
        raise ConnectionError(f"Could not find the {target} board.")
    elif len(target_ports) == 1:
        return target_ports[0]
    else:
        return target_ports


def find_last_result_dir(result_dir: str, prefix: str = "") -> Optional[str]:
    """
    Find the latest (most recent) result directory within a given directory.

    Args:
    - result_dir: The path to the directory containing result subdirectories.
    - prefix: The prefix of result directory names to consider.

    Returns:
    - The path to the latest result directory, or None if no matching directory is found.
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
    robot_dir = os.path.join("toddlerbot", "robot_descriptions", robot_name)
    if os.path.exists(robot_dir):
        file_path = os.path.join(robot_dir, robot_name + suffix)
        if os.path.exists(file_path):
            return file_path
    else:
        assembly_dir = os.path.join("toddlerbot", "robot_descriptions", "assemblies")
        file_path = os.path.join(assembly_dir, robot_name + suffix)
        if os.path.exists(file_path):
            return file_path

    raise FileNotFoundError(f"No {suffix} file found for robot '{robot_name}'.")


def is_xml_pretty_printed(file_path):
    """Check if an XML file is pretty-printed based on indentation and line breaks."""
    with open(file_path, "r") as file:
        lines = file.readlines()

        # Check if there's indentation in lines after the first non-empty one
        for line in lines[1:]:  # Skip XML declaration or root element line
            stripped_line = line.lstrip()
            # If any line starts with a tag and has leading whitespace, assume pretty-printing
            if stripped_line.startswith("<") and len(line) > len(stripped_line):
                return True

    return False


def prettify(elem, file_path):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, "utf-8")
    reparsed = xml.dom.minidom.parseString(rough_string)

    if is_xml_pretty_printed(file_path):
        return reparsed.toxml()
    else:
        return reparsed.toprettyxml(indent="  ", newl="")
