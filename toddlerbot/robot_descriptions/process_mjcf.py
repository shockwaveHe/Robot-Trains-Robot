import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import fields
from itertools import combinations

import numpy as np
from transforms3d.euler import euler2quat

from toddlerbot.sim.robot import HumanoidRobot
from toddlerbot.utils.file_utils import *


def replace_mesh_file(root, old_file, new_file):
    # Find all mesh elements
    for mesh in root.findall(".//mesh"):
        # Check if the file attribute matches the old file name
        if mesh.get("file") == old_file:
            # Replace with the new file name
            mesh.set("file", new_file)


def update_joint_params(root, act_params):
    if act_params is None:
        return

    # Iterate over all joints in the XML
    for joint in root.findall(".//joint"):
        joint_name = joint.get("name")
        # Check if the joint name is in the provided armature dictionary
        if joint_name in act_params:
            for field in fields(act_params[joint_name]):
                if field.name in ["damping", "armature"]:
                    joint.set(
                        field.name, str(getattr(act_params[joint_name], field.name))
                    )


def update_geom_classes(root, geom_keys):
    for geom in root.findall(".//geom[@mesh]"):
        mesh_name = geom.get("mesh")

        # Determine the class based on the mesh name
        if "visual" in mesh_name:
            geom.set("class", "visual")
        elif "collision" in mesh_name:
            geom.set("class", "collision")
        else:
            raise ValueError(f"Unknown class for mesh: {mesh_name}")

        for attr in geom_keys:
            if attr in geom.attrib:
                del geom.attrib[attr]


def add_default_settings(root):
    # Create or find the <default> element
    default = root.find("default")
    if default is not None:
        root.remove(default)

    default = ET.SubElement(root, "default")

    # Set <joint> settings
    ET.SubElement(default, "joint", {"frictionloss": "0.03"})

    # Set <position> settings
    ET.SubElement(default, "position", {"forcelimited": "false"})

    # Set <geom> settings
    ET.SubElement(default, "geom", {"type": "mesh", "solref": ".004 1"})

    # Add <default class="visual"> settings
    visual_default = ET.SubElement(default, "default", {"class": "visual"})
    ET.SubElement(
        visual_default,
        "geom",
        {"contype": "0", "conaffinity": "0", "group": "2", "density": "0"},
    )

    # Add <default class="collision"> settings
    collision_default = ET.SubElement(default, "default", {"class": "collision"})
    # Group 3's visualization is diabled by default
    ET.SubElement(collision_default, "geom", {"group": "3"})


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
            solref="0.0001 1",
        )


def add_actuators_to_mjcf(root, act_params):
    # Create <actuator> element if it doesn't exist
    actuator = root.find("./actuator")
    if actuator is not None:
        root.remove(actuator)

    actuator = ET.SubElement(root, "actuator")

    for joint in root.findall(".//joint"):
        joint_name = joint.get("name")
        if joint_name in act_params:
            motor_name = f"{joint_name}_act"
            ctrlrange = joint.get("range", "-3.141592 3.141592")
            ET.SubElement(
                actuator,
                "position",
                name=motor_name,
                joint=joint_name,
                kp=str(act_params[joint_name].kp),
                kv=str(act_params[joint_name].kv),
                ctrlrange=ctrlrange,
            )


def parse_urdf_body_link(config, urdf_path):
    urdf_tree = ET.parse(urdf_path)
    urdf_root = urdf_tree.getroot()

    # Assuming you want to extract properties for 'body_link'
    body_link = urdf_root.find(
        f"link[@name='{config.canonical_name2link_name['body_link']}']"
    )
    inertial = body_link.find("inertial") if body_link is not None else None

    if inertial is None:
        return None
    else:
        origin = inertial.find("origin").attrib
        mass = inertial.find("mass").attrib["value"]
        inertia = inertial.find("inertia").attrib

        pos = [float(x) for x in origin["xyz"].split(" ")]
        quat = euler2quat(*[float(x) for x in origin["rpy"].split(" ")])
        diaginertia = [
            float(x) for x in [inertia["ixx"], inertia["iyy"], inertia["izz"]]
        ]
        properties = {
            "pos": " ".join([f"{x:.6f}" for x in pos]),
            "quat": " ".join([f"{x:.6f}" for x in quat]),
            "mass": f"{float(mass):.8f}",
            "diaginertia": " ".join(f"{x:.5e}" for x in diaginertia),
        }
        return properties


def add_body_link(root, config, urdf_path):
    properties = parse_urdf_body_link(config, urdf_path)
    if properties is None:
        print("No inertial properties found in URDF file.")
        return

    worldbody = root.find(".//worldbody")

    body_link = ET.Element(
        "body",
        name=config.canonical_name2link_name["body_link"],
        pos="0 0 0",
        quat="1 0 0 0",
    )

    ET.SubElement(
        body_link,
        "inertial",
        pos=properties["pos"],
        quat=properties["quat"],
        mass=properties["mass"],
        diaginertia=properties["diaginertia"],
    )
    ET.SubElement(body_link, "freejoint")

    existing_elements = list(worldbody)
    worldbody.insert(0, body_link)
    for element in existing_elements:
        worldbody.remove(element)
        body_link.append(element)


def update_actuator_types(root, act_params):
    # Create <actuator> element if it doesn't exist
    actuator = root.find("./actuator")
    if actuator is not None:
        root.remove(actuator)

    actuator = ET.SubElement(root, "actuator")

    for joint in root.findall(".//joint"):
        joint_name = joint.get("name")
        if joint_name in act_params:
            motor_name = f"{joint_name}_act"
            ET.SubElement(
                actuator,
                act_params[joint_name].type,
                name=motor_name,
                joint=joint_name,
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


def process_mjcf_debug_file(root, config):
    replace_mesh_file(
        root, "body_link_collision.stl", "body_link_collision_simplified.stl"
    )
    update_joint_params(root, config.act_params)
    update_geom_classes(root, ["type", "contype", "conaffinity", "group", "density"])
    add_contact_exclusion_to_mjcf(root)
    add_actuators_to_mjcf(root, config.act_params)
    add_equality_constraints_for_leaves(root, config.constraint_pairs)
    add_default_settings(root)


def process_mjcf_file(root, config, urdf_path):
    update_actuator_types(root, config.act_params)
    add_body_link(root, config, urdf_path)


def process_mjcf_files(robot_name):
    robot = HumanoidRobot(robot_name)

    robot_dir = os.path.join("toddlerbot", "robot_descriptions", robot_name)
    source_mjcf_path = os.path.join("mjmodel.xml")
    mjcf_debug_path = os.path.join(robot_dir, robot_name + "_debug.xml")
    if os.path.exists(source_mjcf_path):
        shutil.move(source_mjcf_path, mjcf_debug_path)

    xml_tree = ET.parse(mjcf_debug_path)
    xml_root = xml_tree.getroot()

    process_mjcf_debug_file(xml_root, robot.config)
    xml_tree.write(mjcf_debug_path)

    urdf_path = os.path.join(robot_dir, robot_name + ".urdf")
    mjcf_path = os.path.join(robot_dir, robot_name + ".xml")
    process_mjcf_file(xml_root, robot.config, urdf_path)
    xml_tree.write(mjcf_path)

    create_base_scene_xml(mjcf_path)


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
