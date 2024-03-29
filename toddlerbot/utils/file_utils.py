import os
import re
import xml.dom.minidom
import xml.etree.ElementTree as ET
from typing import Optional


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
    robot_dir = os.path.join("toddlerbot", "robot_descriptions", robot_name)
    if os.path.exists(robot_dir):
        description_path = os.path.join(robot_dir, robot_name + suffix)
        if os.path.exists(description_path):
            return description_path
    else:
        assembly_dir = os.path.join("toddlerbot", "robot_descriptions", "assemblies")
        description_path = os.path.join(assembly_dir, robot_name + suffix)
        if os.path.exists(description_path):
            return description_path

    raise FileNotFoundError(f"No URDF file found for robot '{robot_name}'.")


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
