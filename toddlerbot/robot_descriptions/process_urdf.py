import argparse
import os
import shutil
import xml.etree.ElementTree as ET

from toddlerbot.utils.file_utils import *


def find_root_link(root):
    child_links = {joint.find("child").get("link") for joint in root.findall("joint")}
    all_links = {link.get("name") for link in root.findall("link")}

    # The root link is the one not listed as a child
    root_link = all_links - child_links
    if root_link:
        return root_link.pop()
    else:
        raise ValueError("Could not find root link in URDF")


def assemble_urdf(assembly_list, urdf_config, robot_dir):
    # Parse the target URDF
    target_urdf_path = os.path.join(robot_dir, urdf_config.robot_name + ".urdf")
    target_tree = ET.parse(target_urdf_path)
    target_root = target_tree.getroot()

    for joint in target_root.findall("joint"):
        child_link = joint.find("child")
        child_link_name = child_link.attrib.get("link")

        if not "leg" in child_link_name and not "arm" in child_link_name:
            continue

        for link in target_root.findall("link"):
            if link.attrib.get("name") == child_link_name.lower():
                target_root.remove(link)

        source_urdf_path = None
        for assembly_name in assembly_list:
            if assembly_name.lower() == child_link_name.lower():
                # Parse the source URDF
                source_urdf_path = os.path.join(
                    robot_dir, assembly_name, assembly_name + ".urdf"
                )
                break

        if source_urdf_path is None:
            raise ValueError(f"Could not find source URDF for link '{child_link_name}'")

        source_tree = ET.parse(source_urdf_path)
        source_root = source_tree.getroot()

        new_child_link_name = find_root_link(source_root)
        child_link.set("link", new_child_link_name)

        for element in list(source_root):
            target_root.append(element)

    # Check if the <mujoco> element already exists
    mujoco = target_root.find("./mujoco")
    if mujoco is None:
        # Create and insert the <mujoco> element
        mujoco = ET.Element("mujoco")
        compiler = ET.SubElement(mujoco, "compiler")
        compiler.set("meshdir", "./meshes/")
        compiler.set("balanceinertia", "true")
        compiler.set("discardvisual", "false")
        target_root.insert(0, mujoco)

    target_tree.write(target_urdf_path)


def main():
    parser = argparse.ArgumentParser(description="Process the urdf.")
    parser.add_argument(
        "--robot-name",
        type=str,
        default="base",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    args = parser.parse_args()

    robot_dir = os.path.join(
        os.path.join("toddlerbot", "robot_descriptions"), args.robot_name
    )
    process_urdf_and_stl_files(robot_dir)


if __name__ == "__main__":
    main()
