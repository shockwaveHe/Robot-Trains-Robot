import argparse
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

from toddlerbot.utils.file_utils import *


@dataclass
class OnShapeConfig:
    assembly_list: List[str]
    # The following are the default values for the config.json file
    documentId: str = "d2a8be5ce536cd2e18740efa"
    mergeSTLs: str = "all"
    mergeSTLsCollisions: bool = True
    simplifySTLs: str = "all"
    maxSTLSize: int = 1


def process_urdf_and_stl_files(assembly_path):
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
    referenced_stls = set()
    for mesh in root.findall(".//mesh"):
        filename_attr = mesh.get("filename")
        if filename_attr and filename_attr.startswith("package:///"):
            filename = os.path.basename(filename_attr)
            referenced_stls.add(filename)

    # Delete STL and PART files if not referenced
    for entry in os.scandir(assembly_path):
        if entry.is_file():  # Check if the entry is a file
            file = entry.name
            if (
                file.endswith((".stl", ".part"))
                and file not in referenced_stls
                and "simplified" not in file
            ):
                file_path = os.path.join(assembly_path, file)
                os.remove(file_path)

    # Create 'meshes' directory if not exists
    meshes_dir = os.path.join(assembly_path, "meshes")
    if not os.path.exists(meshes_dir):
        os.makedirs(meshes_dir)

    # Move referenced STL files to 'meshes' directory
    for stl in referenced_stls:
        if "left" in robot_name and not "left" in stl:
            new_stl = "left_" + stl
        elif "right" in robot_name and not "right" in stl:
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


def process_assembly(onshape_config, assembly_dir, assembly_name):
    assembly_path = os.path.join(assembly_dir, assembly_name)
    os.makedirs(assembly_path, exist_ok=True)
    json_file_path = os.path.join(assembly_path, "config.json")
    # Map the URDFConfig to the desired JSON structure
    json_data = {
        "documentId": onshape_config.documentId,
        "outputFormat": "urdf",
        "assemblyName": assembly_name,
        "robotName": assembly_name,
        "addDummyBaseLink": "body" in assembly_name.lower(),
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


def run_onshape_to_robot(onshape_config, parallel=False):
    assembly_dir = os.path.join("toddlerbot", "robot_descriptions", "assemblies")

    if parallel:
        # Use ThreadPoolExecutor to parallelize processing
        with ThreadPoolExecutor(
            max_workers=len(onshape_config.assembly_list)
        ) as executor:
            futures = [
                executor.submit(
                    process_assembly, onshape_config, assembly_dir, assembly_name
                )
                for assembly_name in onshape_config.assembly_list
            ]

            # Optionally, wait for all futures to complete and handle exceptions
            for future in futures:
                try:
                    future.result()  # This will raise exceptions if any occurred within a thread
                except Exception as e:
                    print(f"Error processing assembly: {e}")
    else:
        # Process each assembly in series
        for assembly_name in onshape_config.assembly_list:
            process_assembly(onshape_config, assembly_dir, assembly_name)


def main():
    parser = argparse.ArgumentParser(description="Process the urdf.")
    parser.add_argument(
        "--assembly-list",
        type=str,
        nargs="+",  # Indicates that one or more arguments will be consumed.
        required=True,
        help="The names of the assemblies. Need to match the names in OnShape.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Whether to run onshape-to-robot in parallel.",
    )
    args = parser.parse_args()

    run_onshape_to_robot(
        OnShapeConfig(assembly_list=args.assembly_list),
        args.parallel,
    )


if __name__ == "__main__":
    main()
