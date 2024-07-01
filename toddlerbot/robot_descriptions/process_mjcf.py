import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple

from transforms3d.euler import euler2quat  # type: ignore

from toddlerbot.sim.robot import Robot


def find_root_link_name(root: ET.Element):
    child_links = {joint.find("child").get("link") for joint in root.findall("joint")}  # type: ignore
    all_links = {link.get("name") for link in root.findall("link")}

    # The root link is the one not listed as a child
    root_link = all_links - child_links
    if root_link:
        return str(root_link.pop())
    else:
        raise ValueError("Could not find root link in URDF")


def replace_mesh_file(root: ET.Element, old_file: str, new_file: str):
    # Find all mesh elements
    for mesh in root.findall(".//mesh"):
        # Check if the file attribute matches the old file name
        if mesh.get("file") == old_file:
            # Replace with the new file name
            mesh.set("file", new_file)


def add_torso_site(root: ET.Element):
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("No worldbody element found in the XML.")

    site_attributes = {
        "name": "torso",
        "fromto": "0.01 0 0.4 -0.01 0 0.4",
        "type": "cylinder",
        "size": "0.005 0.005 1",
        "group": "3",
    }

    site_element = ET.Element("site", site_attributes)
    worldbody.insert(0, site_element)


def add_imu_sensor(root: ET.Element):
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("No worldbody element found in the XML.")

    site_attributes = {"name": "imu", "size": "0.01", "pos": "0.0 0 0.0"}
    site_element = ET.Element("site", site_attributes)
    worldbody.insert(0, site_element)

    sensor = root.find("sensor")
    if sensor is not None:
        root.remove(sensor)

    sensor = ET.SubElement(root, "sensor")

    # Adding framequat sub-element
    ET.SubElement(
        sensor,
        "framequat",
        attrib={
            "name": "orientation",
            "objtype": "site",
            "noise": "0.001",
            "objname": "imu",
        },
    )

    # Adding framepos sub-element
    ET.SubElement(
        sensor,
        "framepos",
        attrib={
            "name": "position",
            "objtype": "site",
            "noise": "0.001",
            "objname": "imu",
        },
    )

    # Adding gyro sub-element
    ET.SubElement(
        sensor,
        "gyro",
        attrib={
            "name": "angular_velocity",
            "site": "imu",
            "noise": "0.005",
            "cutoff": "34.9",
        },
    )

    # Adding velocimeter sub-element
    ET.SubElement(
        sensor,
        "velocimeter",
        attrib={
            "name": "linear_velocity",
            "site": "imu",
            "noise": "0.001",
            "cutoff": "30",
        },
    )

    # Adding accelerometer sub-element
    ET.SubElement(
        sensor,
        "accelerometer",
        attrib={
            "name": "linear_acceleration",
            "site": "imu",
            "noise": "0.005",
            "cutoff": "157",
        },
    )

    # Adding magnetometer sub-element
    ET.SubElement(
        sensor, "magnetometer", attrib={"name": "magnetometer", "site": "imu"}
    )


def update_joint_params(root: ET.Element, config: Dict[str, Any]):
    # Iterate over all joints in the XML
    for joint in root.findall(".//joint"):
        joint_name = joint.get("name")
        if joint_name in config:
            for attr_name in config[joint_name]:
                if attr_name in ["damping", "armature", "frictionloss"]:
                    joint.set(attr_name, str(config[joint_name][attr_name]))


def update_geom_classes(root: ET.Element, geom_keys: List[str]):
    for geom in root.findall(".//geom[@mesh]"):
        mesh_name = geom.get("mesh")

        if mesh_name is None:
            continue

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


def add_default_settings(root: ET.Element):
    # Create or find the <default> element
    default = root.find("default")
    if default is not None:
        root.remove(default)

    default = ET.SubElement(root, "default")

    # Set <joint> settings
    # ET.SubElement(default, "joint", {"frictionloss": "0.03"})

    # Set <position> settings
    ET.SubElement(default, "position", {"forcelimited": "false"})

    # Set <geom> settings
    ET.SubElement(
        default,
        "geom",
        {"type": "mesh", "condim": "4", "solref": ".001 2", "friction": "0.9 0.2 0.2"},
    )

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


def exclude_all_contacts(root: ET.Element):
    contact = root.find("contact")
    if contact is not None:
        root.remove(contact)

    contact = ET.SubElement(root, "contact")

    for body1 in root.findall(".//body"):
        body1_name = body1.get("name")
        for body2 in root.findall(".//body"):
            body2_name = body2.get("name")
            if (
                body1_name is not None
                and body2_name is not None
                and body1_name != body2_name
            ):
                ET.SubElement(contact, "exclude", body1=body1_name, body2=body2_name)


def add_contact_exclusion_to_mjcf(root: ET.Element):
    # Ensure there is a <contact> element
    contact = root.find("contact")
    if contact is not None:
        root.remove(contact)

    contact = ET.SubElement(root, "contact")

    # Iterate through all bodies
    for body in root.findall(".//body"):
        parent_name = body.get("name")
        # Ensure the parent has a 'collision' class geom as a direct child
        if parent_name and body.find("./geom[@class='collision']") is not None:
            # Iterate over direct children bodies of the current body
            for child in body.findall(".//body"):
                child_name = child.get("name")
                # Ensure the child has a 'collision' class geom as a direct child
                if child_name and child.find("./geom[@class='collision']") is not None:
                    # Add exclusion since both parent and child meet the criteria
                    ET.SubElement(
                        contact, "exclude", body1=parent_name, body2=child_name
                    )


def add_equality_constraints_for_leaves(
    root: ET.Element, body_pairs: List[Tuple[str, str]]
):
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


def add_actuators_to_mjcf(root: ET.Element, config: Dict[str, Any]):
    # Create <actuator> element if it doesn't exist
    actuator = root.find("./actuator")
    if actuator is not None:
        root.remove(actuator)

    actuator = ET.SubElement(root, "actuator")

    for joint in root.findall(".//joint"):
        joint_name = joint.get("name")
        if joint_name in config:
            motor_name = f"{joint_name}_act"
            ctrlrange = joint.get("range", "-3.141592 3.141592")
            ET.SubElement(
                actuator,
                "position",
                name=motor_name,
                joint=joint_name,
                kp=str(config[joint_name]["kp_sim"]),
                kv=str(config[joint_name]["kd_sim"]),
                ctrlrange=ctrlrange,
            )


def parse_urdf_body_link(root: ET.Element, root_link_name: str):
    # Assuming you want to extract properties for 'body_link'
    body_link = root.find(f"link[@name='{root_link_name}']")
    inertial = body_link.find("inertial") if body_link is not None else None

    if inertial is None:
        return None
    else:
        origin = inertial.find("origin").attrib  # type: ignore
        mass = inertial.find("mass").attrib["value"]  # type: ignore
        inertia = inertial.find("inertia").attrib  # type: ignore

        pos = [float(x) for x in origin["xyz"].split(" ")]
        quat = euler2quat(*[float(x) for x in origin["rpy"].split(" ")])  # type: ignore
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


def add_body_link(root: ET.Element, urdf_path: str):
    urdf_tree = ET.parse(urdf_path)
    urdf_root = urdf_tree.getroot()
    root_link_name: str = find_root_link_name(urdf_root)
    properties = parse_urdf_body_link(urdf_root, root_link_name)
    if properties is None:
        print("No inertial properties found in URDF file.")
        return

    worldbody = root.find(".//worldbody")
    if worldbody is None:
        raise ValueError("No worldbody element found in the XML.")

    body_link = ET.Element(
        "body",
        name=root_link_name,
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


def update_actuator_types(root: ET.Element, config: Dict[str, Any]):
    # Create <actuator> element if it doesn't exist
    actuator = root.find("./actuator")
    if actuator is not None:
        root.remove(actuator)

    actuator = ET.SubElement(root, "actuator")

    for joint in root.findall(".//joint"):
        joint_name = joint.get("name")
        if joint_name in config:
            motor_name = f"{joint_name}_act"
            ET.SubElement(
                actuator,
                config[joint_name]["control_mode"],
                name=motor_name,
                joint=joint_name,
            )


def create_base_scene_xml(mjcf_path: str):
    robot_name = os.path.basename(mjcf_path).replace("_fixed", "").replace(".xml", "")

    # Create the root element
    mujoco = ET.Element("mujoco", attrib={"model": f"{robot_name}_scene"})

    # Include the robot model
    ET.SubElement(mujoco, "include", attrib={"file": mjcf_path})

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
    ET.SubElement(
        visual,
        "global",
        attrib={
            "azimuth": "160",
            "elevation": "-20",
            "offwidth": "1280",
            "offheight": "720",
        },
    )

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
            "condim": "1",
        },
    )

    # Create a tree from the root element and write it to a file
    tree = ET.ElementTree(mujoco)
    tree.write(os.path.join(os.path.dirname(mjcf_path), f"{robot_name}_scene.xml"))


def process_mjcf_fixed_file(root: ET.Element, config: Dict[str, Any]):
    if config["general"]["use_torso_site"]:
        add_torso_site(root)

    if config["general"]["has_imu"]:
        add_imu_sensor(root)

    update_joint_params(root, config)
    update_geom_classes(root, ["type", "contype", "conaffinity", "group", "density"])
    exclude_all_contacts(root)
    add_actuators_to_mjcf(root, config)

    if len(config["general"]["constraint_pairs"]) > 0:
        add_equality_constraints_for_leaves(root, config["general"]["constraint_pairs"])

    add_default_settings(root)
    add_contact_exclusion_to_mjcf(root)


def get_mjcf_files(robot_name: str):
    cache_file_path = os.path.join(
        "toddlerbot", "robot_descriptions", robot_name, f"{robot_name}_data.pkl"
    )
    if os.path.exists(cache_file_path):
        os.remove(cache_file_path)

    robot = Robot(robot_name)

    robot_dir = os.path.join("toddlerbot", "robot_descriptions", robot_name)
    urdf_path = os.path.join(robot_dir, robot_name + ".urdf")
    source_mjcf_path = os.path.join("mjmodel.xml")
    mjcf_fixed_path = os.path.join(robot_dir, robot_name + "_fixed.xml")
    if os.path.exists(source_mjcf_path):
        shutil.move(source_mjcf_path, mjcf_fixed_path)
    else:
        raise ValueError(
            "No MJCF file found. Remember to click the button save_xml to save the model to mjmodel.xml in the current directory."
        )

    xml_tree = ET.parse(mjcf_fixed_path)
    xml_root = xml_tree.getroot()

    process_mjcf_fixed_file(xml_root, robot.config)
    xml_tree.write(mjcf_fixed_path)

    if robot.config["general"]["is_fixed"]:
        mjcf_path = mjcf_fixed_path
    else:
        mjcf_path = os.path.join(robot_dir, robot_name + ".xml")
        add_body_link(xml_root, urdf_path)
        xml_tree.write(mjcf_path)

    create_base_scene_xml(mjcf_path)


def main():
    parser = argparse.ArgumentParser(description="Process the MJCF.")
    parser.add_argument(
        "--robot-name",
        type=str,
        default="sysID_XC430",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    args = parser.parse_args()

    get_mjcf_files(args.robot_name)


if __name__ == "__main__":
    main()
