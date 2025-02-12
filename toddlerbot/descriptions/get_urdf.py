import argparse
import json
import os
import shutil
import subprocess
import xml.dom.minidom
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Set


def is_xml_pretty_printed(file_path: str) -> bool:
    """Check if an XML file is pretty-printed by examining indentation.

    Args:
        file_path (str): The path to the XML file to be checked.

    Returns:
        bool: True if the XML file is pretty-printed with indentation, False otherwise.
    """
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
    """Formats an XML element into a pretty-printed string.

    This function converts an XML element into a string and formats it with indentation for improved readability. If the XML file at the specified path is already pretty-printed, it returns the compact XML string; otherwise, it returns a pretty-printed version with indentation.

    Args:
        elem (ET.Element): The XML element to be formatted.
        file_path (str): The path to the XML file to check for pretty-printing.

    Returns:
        str: A string representation of the XML element, either compact or pretty-printed.
    """
    rough_string = ET.tostring(elem, "utf-8")
    reparsed = xml.dom.minidom.parseString(rough_string)

    if is_xml_pretty_printed(file_path):
        return reparsed.toxml()
    else:
        return reparsed.toprettyxml(indent="  ", newl="")


@dataclass
class OnShapeConfig:
    """Data class for storing OnShape configuration parameters."""

    doc_id_list: List[str]
    assembly_list: List[str]
    # The following are the default values for the config.json file
    mergeSTLs: str = "all"
    mergeSTLsCollisions: bool = True
    simplifySTLs: str = "all"
    maxSTLSize: int = 1


def process_urdf_and_stl_files(assembly_path: str):
    """Processes URDF and STL files within a specified assembly directory.

    This function performs several operations on URDF and STL files located in the given assembly path:
    1. Parses the URDF file and updates the robot name to match the directory name.
    2. Identifies and collects all referenced STL files from the URDF.
    3. Deletes any STL or PART files in the directory that are not referenced in the URDF.
    4. Moves referenced STL files to a 'meshes' directory, creating it if necessary.
    5. Updates the URDF file to reflect the new locations of the STL files.
    6. Renames the URDF file to match the base directory name if needed.

    Args:
        assembly_path (str): The path to the directory containing the URDF and STL files.

    Raises:
        ValueError: If no URDF file is found in the specified directory.
    """
    urdf_path = os.path.join(assembly_path, "robot.urdf")
    if not os.path.exists(urdf_path):
        raise ValueError("No URDF file found in the robot directory.")

    # Parse the URDF file
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    robot_name = os.path.basename(assembly_path)
    # Update robot name to match URDF file name
    if root.attrib["name"] != robot_name:
        root.attrib["name"] = robot_name

    # Find and update all mesh filenames
    referenced_stls: Set[str] = set()
    for mesh in root.findall(".//mesh"):
        filename_attr = mesh.get("filename")
        if filename_attr and filename_attr.startswith("package:///"):
            filename = os.path.basename(filename_attr)
            referenced_stls.add(filename)

    # Delete STL and PART files if not referenced
    for entry in os.scandir(assembly_path):
        if entry.is_file():  # Check if the entry is a file
            file = entry.name
            if file.endswith((".stl", ".part")) and file not in referenced_stls:
                file_path = os.path.join(assembly_path, file)
                os.remove(file_path)

    # Create 'meshes' directory if not exists
    meshes_dir = os.path.join(assembly_path, "meshes")
    if not os.path.exists(meshes_dir):
        os.makedirs(meshes_dir)

    # Move referenced STL files to 'meshes' directory
    for stl in referenced_stls:
        if "left" in robot_name and "left" not in stl:
            new_stl = "left_" + stl
        elif "right" in robot_name and "right" not in stl:
            new_stl = "right_" + stl
        else:
            new_stl = stl

        # Update the filename attribute in the URDF file
        for mesh in root.findall(".//mesh"):
            filename_attr = mesh.get("filename")
            if filename_attr and filename_attr.endswith(stl):
                mesh.set("filename", f"package:///meshes/{new_stl}")

        source_path = os.path.join(assembly_path, stl)
        if os.path.exists(source_path):
            shutil.move(source_path, os.path.join(meshes_dir, new_stl))

    pretty_xml = prettify(root, urdf_path)
    # Write the modified XML back to the URDF file
    with open(urdf_path, "w") as urdf_file:
        urdf_file.write(pretty_xml)

    # Rename URDF file to match the base directory name if necessary
    new_urdf_path = os.path.join(assembly_path, robot_name + ".urdf")
    if urdf_path != new_urdf_path:
        os.rename(urdf_path, new_urdf_path)


def run_onshape_to_robot(onshape_config: OnShapeConfig):
    """Processes OnShape assemblies and converts them to URDF format for robotic applications.

    This function iterates over a list of OnShape document IDs and corresponding assembly names, creating a directory for each assembly. It generates a configuration JSON file for each assembly, specifying parameters for URDF conversion. The function then executes a command to convert the assembly to URDF format and processes the resulting URDF and STL files.

    Args:
        onshape_config (OnShapeConfig): Configuration object containing lists of document IDs and assembly names, along with settings for STL merging, collision handling, simplification, and maximum STL size.
    """
    assembly_dir = os.path.join("toddlerbot", "descriptions", "assemblies")

    # Process each assembly in series
    for doc_id, assembly_name in zip(
        onshape_config.doc_id_list, onshape_config.assembly_list
    ):
        assembly_path = os.path.join(assembly_dir, assembly_name)

        if os.path.exists(assembly_path):
            shutil.rmtree(assembly_path)

        os.makedirs(assembly_path)
        json_file_path = os.path.join(assembly_path, "config.json")
        # Map the URDFConfig to the desired JSON structure
        json_data = {
            "documentId": doc_id,
            "outputFormat": "urdf",
            "assemblyName": assembly_name,
            "robotName": assembly_name,
            "mergeSTLs": onshape_config.mergeSTLs,
            "mergeSTLsCollisions": onshape_config.mergeSTLsCollisions,
            "simplifySTLs": onshape_config.simplifySTLs,
            "maxSTLSize": onshape_config.maxSTLSize,
        }

        # Write the JSON data to a file
        with open(json_file_path, "w") as json_file:
            json.dump(json_data, json_file, indent=4)

        # Execute the command
        subprocess.run(f"onshape-to-robot {assembly_path}", shell=True)

        process_urdf_and_stl_files(assembly_path)


def main():
    """Parses command-line arguments for document and assembly names and processes them using OnShape.

    This function sets up an argument parser to handle command-line inputs for document IDs and assembly names, ensuring they match the names in OnShape. It then invokes the `run_onshape_to_robot` function with the parsed arguments.

    Raises:
        SystemExit: If required command-line arguments are not provided.
    """
    parser = argparse.ArgumentParser(description="Process the urdf.")
    parser.add_argument(
        "--doc-id-list",
        type=str,
        nargs="+",  # Indicates that one or more arguments will be consumed.
        required=True,
        help="The names of the documents. Need to match the names in OnShape.",
    )
    parser.add_argument(
        "--assembly-list",
        type=str,
        nargs="+",  # Indicates that one or more arguments will be consumed.
        required=True,
        help="The names of the assemblies. Need to match the names in OnShape.",
    )
    args = parser.parse_args()

    run_onshape_to_robot(
        OnShapeConfig(doc_id_list=args.doc_id_list, assembly_list=args.assembly_list)
    )


if __name__ == "__main__":
    main()
