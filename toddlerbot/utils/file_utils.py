import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
from typing import List, Optional

import serial.tools.list_ports as list_ports
from PIL import Image

from toddlerbot.utils.misc_utils import log


def get_load_path(root: str, load_run: str = "", checkpoint: int = -1):
    try:
        runs = os.listdir(root)
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")

        last_run = os.path.join(root, runs[-1])

    except Exception:
        raise ValueError("No runs in this directory: " + root)

    if len(load_run) == 0:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if "model" in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)

    return load_path


def combine_images(image1_path: str, image2_path: str, output_path: str):
    # Open the two images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Get the dimensions of the images
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Create a new image with the combined width and the maximum height
    combined_image = Image.new("RGB", (width1 + width2, max(height1, height2)))

    # Paste the first image at the left
    combined_image.paste(image1, (0, 0))

    # Paste the second image at the right
    combined_image.paste(image2, (width1, 0))

    # Save the combined image
    combined_image.save(output_path)


def find_ports(target: str) -> List[str]:
    ports = list(list_ports.comports())
    target_ports: List[str] = []
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
    else:
        return sorted(target_ports)


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


def is_xml_pretty_printed(file_path: str) -> bool:
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


def prettify(elem: ET.Element, file_path: str):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, "utf-8")
    reparsed = xml.dom.minidom.parseString(rough_string)

    if is_xml_pretty_printed(file_path):
        return reparsed.toxml()
    else:
        return reparsed.toprettyxml(indent="  ", newl="")
