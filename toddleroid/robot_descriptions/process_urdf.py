import argparse
import os
import shutil
import xml.etree.ElementTree as ET

from toddleroid.utils.file_utils import *


def parse_urdf_for_closing_links(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    closing_links = {}
    for link in root.findall(".//link"):
        link_name = link.get("name")
        if link_name.startswith("closing_"):
            base_name = "_".join(link_name.split("_")[:-1])
            if base_name in closing_links:
                closing_links[base_name] = (closing_links[base_name], link_name)
            else:
                closing_links[base_name] = link_name

    return closing_links


def process_urdf_and_stl_files(robot_dir):
    robot_name = os.path.basename(robot_dir)

    # Check for URDF file in the base directory
    is_urdf_found = False
    urdf_names = ["robot.urdf", robot_name + ".urdf"]
    urdf_path = ""
    for urdf_name in urdf_names:
        urdf_path = os.path.join(robot_dir, urdf_name)
        if os.path.exists(urdf_path):
            is_urdf_found = True
            break

    if not is_urdf_found:
        raise ValueError("No URDF file found in the robot directory.")

    # Parse the URDF file
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Update robot name to match URDF file name
    if root.attrib["name"] != robot_name:
        root.attrib["name"] = robot_name

    # Find and update all mesh filenames
    referenced_stls = set()
    for mesh in root.findall(".//mesh"):
        filename_attr = mesh.get("filename")
        if filename_attr and filename_attr.startswith("package:///"):
            filename = os.path.basename(filename_attr)
            new_path = f"package:///meshes/{filename}"
            mesh.set("filename", new_path)
            referenced_stls.add(filename)

    # Check if the <mujoco> element already exists
    mujoco = root.find("./mujoco")
    if mujoco is None:
        # Create and insert the <mujoco> element
        mujoco = ET.Element("mujoco")
        compiler = ET.SubElement(mujoco, "compiler")
        compiler.set("meshdir", "./meshes/")
        compiler.set("balanceinertia", "true")
        compiler.set("discardvisual", "false")
        root.insert(0, mujoco)

    pretty_xml = prettify(root, urdf_path)

    # Write the modified XML back to the URDF file
    with open(urdf_path, "w") as urdf_file:
        urdf_file.write(pretty_xml)

    # Delete STL and PART files if not referenced
    for subdir, _, files in os.walk(robot_dir):
        for file in files:
            if file.endswith((".stl", ".part")) and file not in referenced_stls:
                file_path = os.path.join(subdir, file)
                os.remove(file_path)

    # Create 'meshes' directory if not exists
    meshes_dir = os.path.join(robot_dir, "meshes")
    if not os.path.exists(meshes_dir):
        os.makedirs(meshes_dir)

    # Move referenced STL files to 'meshes' directory
    for stl in referenced_stls:
        source_path = os.path.join(robot_dir, stl)
        target_path = os.path.join(meshes_dir, stl)
        if os.path.exists(source_path):
            shutil.move(source_path, target_path)

    # Rename URDF file to match the base directory name if necessary
    new_urdf_path = os.path.join(robot_dir, robot_name + ".urdf")
    if urdf_path != new_urdf_path:
        os.rename(urdf_path, new_urdf_path)


def main():
    parser = argparse.ArgumentParser(description="Process the urdf.")
    parser.add_argument(
        "--robot-name",
        type=str,
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    args = parser.parse_args()

    robot_dir = os.path.join(
        os.path.join("toddleroid", "robot_descriptions"), args.robot_name
    )
    process_urdf_and_stl_files(robot_dir)


if __name__ == "__main__":
    main()
