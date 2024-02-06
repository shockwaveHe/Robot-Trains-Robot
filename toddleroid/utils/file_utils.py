import os
import re
import xml.dom.minidom
import xml.etree.ElementTree as ET


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
