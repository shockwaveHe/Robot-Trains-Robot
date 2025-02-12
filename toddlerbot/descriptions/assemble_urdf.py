import argparse
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List


@dataclass
class URDFConfig:
    """Data class for storing URDF configuration parameters."""

    robot_name: str
    body_name: str
    arm_name: str
    leg_name: str


def find_root_link_name(root: ET.Element):
    """Updates link names in `part_root` to ensure uniqueness in `body_root` and updates all references.

    Args:
        root (ET.Element): The root XML element of the part URDF being merged.

    Returns:
        str: The new unique name for the root link.
    """
    child_links = {joint.find("child").get("link") for joint in root.findall("joint")}  # type: ignore
    all_links = {link.get("name") for link in root.findall("link")}

    # The root link is the one not listed as a child
    root_link = all_links - child_links
    if root_link:
        return str(root_link.pop())
    else:
        raise ValueError("Could not find root link in URDF")


def update_link_names_and_references(body_root: ET.Element, part_root: ET.Element):
    """
    Updates link names in part_root to ensure uniqueness in body_root and updates all references.

    Args:
        body_root: The root XML element of the main URDF body.
        part_root: The root XML element of the part URDF being merged.

    Returns:
        None: The function directly modifies part_root in-place.
    """
    existing_links = {link.attrib["name"] for link in body_root.findall("link")}

    # Function to find or generate a unique link name
    def get_unique_name(old_name: str):
        if old_name not in existing_links:
            return old_name
        i = 2
        old_name_words = old_name.split("_")
        if old_name_words[-1].isdigit() and int(old_name_words[-1]) < 100:
            old_name = "_".join(old_name_words[:-1])

        while f"{old_name}_{i}" in existing_links:
            i += 1

        return f"{old_name}_{i}"

    # Update link names in part_root and collect changes
    name_changes = {}
    for link in part_root.findall("link"):
        old_name = link.attrib["name"]
        new_name = get_unique_name(old_name)
        if old_name != new_name:
            link.attrib["name"] = new_name
            name_changes[old_name] = new_name
            existing_links.add(new_name)

    for joint in part_root.findall("joint"):
        for tag in ["parent", "child"]:
            link_element = joint.find(tag)
            if link_element is not None and link_element.attrib["link"] in name_changes:
                link_element.attrib["link"] = name_changes[link_element.attrib["link"]]


def assemble_urdf(urdf_config: URDFConfig):
    """Assembles a URDF file for a robot based on the provided configuration.

    This function constructs a complete URDF (Unified Robot Description Format) file by combining a base body URDF with optional arm and leg components specified in the configuration. It updates mesh file paths and ensures the correct structure for simulation.

    Args:
        urdf_config (URDFConfig): Configuration object containing the names of the robot, body, arms, and legs to be assembled.

    Raises:
        ValueError: If a source URDF for a specified link cannot be found.
    """
    # Parse the target URDF
    description_dir = os.path.join("toddlerbot", "descriptions")
    assembly_dir = os.path.join(description_dir, "assemblies")

    body_urdf_path = os.path.join(
        assembly_dir, urdf_config.body_name, urdf_config.body_name + ".urdf"
    )
    body_tree = ET.parse(body_urdf_path)
    body_root = body_tree.getroot()
    body_root.set("name", urdf_config.robot_name)

    assembly_list: List[str] = []
    if len(urdf_config.arm_name) > 0:
        assembly_list.append("left_" + urdf_config.arm_name)
        assembly_list.append("right_" + urdf_config.arm_name)

    if len(urdf_config.leg_name) > 0:
        assembly_list.append("left_" + urdf_config.leg_name)
        assembly_list.append("right_" + urdf_config.leg_name)

    for element in list(body_root):
        # Before appending, update the filename attribute in <mesh> tags
        for mesh in element.findall(".//mesh"):
            mesh.attrib["filename"] = mesh.attrib["filename"].replace(
                "package:///", f"../assemblies/{urdf_config.body_name}/"
            )

    for joint in body_root.findall("joint"):
        child_link = joint.find("child")
        if child_link is None:
            continue

        child_link_name = child_link.attrib.get("link")
        if child_link_name is None:
            continue

        if not ("leg" in child_link_name and len(urdf_config.leg_name) > 0) and not (
            "arm" in child_link_name and len(urdf_config.arm_name) > 0
        ):
            continue

        for link in body_root.findall("link"):
            if link.attrib.get("name") == child_link_name.lower():
                body_root.remove(link)

        source_urdf_path = None
        assembly_name = ""
        child_link_name_words = child_link_name.split("_")
        for name in assembly_list:
            name_words = name.split("_")
            if (
                name_words[0].lower() == child_link_name_words[0].lower()
                and name_words[1].lower() == child_link_name_words[1].lower()
            ):
                source_urdf_path = os.path.join(assembly_dir, name, name + ".urdf")
                assembly_name = name
                break

        if source_urdf_path is None:
            raise ValueError(f"Could not find source URDF for link '{child_link_name}'")

        part_tree = ET.parse(source_urdf_path)
        part_root = part_tree.getroot()

        update_link_names_and_references(body_root, part_root)
        child_link.set("link", find_root_link_name(part_root))

        for element in list(part_root):
            # Before appending, update the filename attribute in <mesh> tags
            for mesh in element.findall(".//mesh"):
                mesh.attrib["filename"] = mesh.attrib["filename"].replace(
                    "package:///", f"../assemblies/{assembly_name}/"
                )

            body_root.append(element)

    # Check if the <mujoco> element already exists
    mujoco = body_root.find("./mujoco")
    if mujoco is None:
        # Create and insert the <mujoco> element
        mujoco = ET.Element("mujoco")
        compiler = ET.SubElement(mujoco, "compiler")
        compiler.set("strippath", "false")
        compiler.set("balanceinertia", "true")
        compiler.set("discardvisual", "false")
        body_root.insert(0, mujoco)

    target_robot_dir = os.path.join(description_dir, urdf_config.robot_name)
    os.makedirs(target_robot_dir, exist_ok=True)
    target_urdf_path = os.path.join(target_robot_dir, urdf_config.robot_name + ".urdf")
    body_tree.write(target_urdf_path)


def main():
    """Parses command-line arguments to configure and assemble a URDF (Unified Robot Description Format) file.

    This function sets up an argument parser to accept parameters for robot configuration, including the robot's name, body, arm, and leg components. It then calls the `assemble_urdf` function with the parsed arguments to generate the URDF file.
    """
    parser = argparse.ArgumentParser(description="Assemble the urdf.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    parser.add_argument(
        "--body-name",
        type=str,
        default="4R_body",
        help="The name of the body.",
    )
    parser.add_argument(
        "--arm-name",
        type=str,
        default="",
        help="The name of the arm.",
    )
    parser.add_argument(
        "--leg-name",
        type=str,
        default="",
        help="The name of the leg.",
    )
    args = parser.parse_args()

    assemble_urdf(URDFConfig(args.robot, args.body_name, args.arm_name, args.leg_name))


if __name__ == "__main__":
    main()
