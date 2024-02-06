import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from itertools import combinations

import numpy as np

from toddleroid.sim.robot import HumanoidRobot
from toddleroid.utils.file_utils import *


def add_default_settings(root, act_type="position"):
    # Create or find the <default> element
    default = root.find("default")
    if default is not None:
        root.remove(default)

    default = ET.SubElement(root, "default")

    # Create or update the <joint> settings within <default>
    joint_default = default.find("joint")
    if joint_default is None:
        joint_default = ET.SubElement(default, "joint")
    joint_default.attrib = {
        "damping": "1.084",
        "armature": "0.045",
        "frictionloss": "0.03",
    }

    # Create or update the <position> settings within <default>
    position_default = default.find(act_type)
    if position_default is None:
        position_default = ET.SubElement(default, act_type)
    position_default.attrib = {"kv": "10", "forcelimited": "false"}


def add_contact_exclusion_to_mjcf(root):
    # Ensure there is a <contact> element
    contact = root.find("contact")
    if contact is not None:
        root.remove(contact)

    contact = ET.SubElement(root, "contact")

    # Collect all body names
    body_names = [
        body.get("name") for body in root.findall(".//body") if body.get("name")
    ]

    # Generate all unique pairs of body names
    body_pairs = combinations(body_names, 2)

    # Add an <exclude> element for each pair
    for body1, body2 in body_pairs:
        exclude = contact.find(f"./exclude[@body1='{body1}'][@body2='{body2}']")
        if exclude is None:
            ET.SubElement(contact, "exclude", body1=body1, body2=body2)


def add_actuators_to_mjcf(root, act_list, act_type="position"):
    # Create <actuator> element if it doesn't exist
    actuator = root.find("./actuator")
    if actuator is not None:
        root.remove(actuator)

    actuator = ET.SubElement(root, "actuator")

    for joint in root.findall(".//joint"):
        joint_name = joint.get("name")
        joint_type = joint.get("type")
        if joint_name in act_list:
            motor_name = f"{joint_name}_act"
            # Retrieve control range from joint, if available
            if act_type == "position":
                ctrlrange = joint.get("range", "-3.141592 3.141592")
            else:
                ctrlrange = "0 0"

            ET.SubElement(
                actuator,
                act_type,
                name=motor_name,
                joint=joint_name,
                kp="1e6" if joint_type == "slide" else "1e4",
                ctrlrange=ctrlrange,
            )


def add_equality_constraints_for_leaves(root, body_pairs):
    # Ensure there is an <equality> element
    equality = root.find("./equality")
    if equality is not None:
        root.remove(equality)

    equality = ET.SubElement(root, "equality")

    # Add equality constraints for each pair
    for body1, body2 in body_pairs:
        ET.SubElement(
            equality,
            "weld",
            body1=body1,
            body2=body2,
            solimp="0.9999 0.9999 0.001 0.5 2",
        )


def create_base_scene_xml(mjcf_path):
    robot_name = os.path.basename(mjcf_path).replace(".xml", "")

    # Create the root element
    mujoco = ET.Element("mujoco", attrib={"model": f"{robot_name}_scene"})

    # Include the robot model
    ET.SubElement(mujoco, "include", attrib={"file": f"{robot_name}.xml"})

    # Add statistic element
    ET.SubElement(mujoco, "statistic", attrib={"center": "0 0 0.2", "extent": "0.6"})

    # Visual settings
    visual = ET.SubElement(mujoco, "visual")
    ET.SubElement(
        visual,
        "headlight",
        attrib={
            "diffuse": "0.6 0.6 0.6",
            "ambient": "0.3 0.3 0.3",
            "specular": "0 0 0",
        },
    )
    ET.SubElement(visual, "rgba", attrib={"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(visual, "global", attrib={"azimuth": "160", "elevation": "-20"})

    # Asset settings
    asset = ET.SubElement(mujoco, "asset")
    ET.SubElement(
        asset,
        "texture",
        attrib={
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.3 0.5 0.7",
            "rgb2": "0 0 0",
            "width": "512",
            "height": "3072",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        attrib={
            "type": "2d",
            "name": "groundplane",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "0.2 0.3 0.4",
            "rgb2": "0.1 0.2 0.3",
            "markrgb": "0.8 0.8 0.8",
            "width": "300",
            "height": "300",
        },
    )
    ET.SubElement(
        asset,
        "material",
        attrib={
            "name": "groundplane",
            "texture": "groundplane",
            "texuniform": "true",
            "texrepeat": "5 5",
            "reflectance": "0.0",
        },
    )

    # Worldbody settings
    worldbody = ET.SubElement(mujoco, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        attrib={"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"},
    )
    ET.SubElement(
        worldbody,
        "geom",
        attrib={
            "name": "floor",
            "size": "0 0 0.05",
            "type": "plane",
            "material": "groundplane",
        },
    )

    # Create a tree from the root element and write it to a file
    tree = ET.ElementTree(mujoco)
    tree.write(os.path.join(os.path.dirname(mjcf_path), f"{robot_name}_scene.xml"))


def process_mjcf_files(robot_name):
    robot_dir = os.path.join("toddleroid", "robot_descriptions", robot_name)
    source_mjcf_path = os.path.join("mjmodel.xml")
    mjcf_path = os.path.join(robot_dir, robot_name + ".xml")
    if os.path.exists(source_mjcf_path):
        shutil.move(source_mjcf_path, mjcf_path)

    create_base_scene_xml(mjcf_path)

    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    robot = HumanoidRobot(robot_name)
    body_pairs = [
        ("ank_act_rod_head", "ank_act_rod"),
        ("ank_act_rod_head_2", "ank_act_rod_2"),
    ]
    act_type = "position"

    add_contact_exclusion_to_mjcf(root)
    add_actuators_to_mjcf(root, robot.config.joint_names, act_type)
    add_equality_constraints_for_leaves(root, body_pairs)
    add_default_settings(root, act_type)

    tree.write(mjcf_path)


def main():
    parser = argparse.ArgumentParser(description="Process the MJCF.")
    parser.add_argument(
        "--robot-name",
        type=str,
        default="base",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    args = parser.parse_args()

    process_mjcf_files(args.robot_name)


if __name__ == "__main__":
    main()
