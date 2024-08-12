import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple

from transforms3d.euler import euler2quat  # type: ignore

from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import round_to_sig_digits

# TODO: Implement the actuator model of MuJoCo in IsaacGym. How should I do frictionloss?
# What's the activation parameter? Damping and armature are known.


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


def update_compiler_settings(root: ET.Element):
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("No compiler element found in the XML.")

    compiler.set("autolimits", "true")


def add_option_settings(root: ET.Element):
    option = root.find("option")
    if option is not None:
        root.remove(option)

    option = ET.SubElement(root, "option")

    ET.SubElement(option, "flag", {"eulerdamp": "disable"})


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


def update_joint_params(root: ET.Element, joints_config: Dict[str, Any]):
    # Iterate over all joints in the XML
    for joint in root.findall(".//joint"):
        joint_name = joint.get("name")
        if joint_name in joints_config:
            for attr_name in joints_config[joint_name]:
                if attr_name in ["damping", "armature"]:  # , "frictionloss"]:
                    attr_value = round_to_sig_digits(
                        joints_config[joint_name][attr_name], 6
                    )
                    joint.set(attr_name, str(attr_value))


def update_geom_classes(root: ET.Element, geom_keys: List[str]):
    for geom in root.findall(".//geom[@mesh]"):
        mesh_name = geom.get("mesh")
        if mesh_name is None:
            continue

        # Determine the class based on the mesh name
        if "visual" in mesh_name:
            geom.set("class", "visual")
        else:
            raise ValueError(f"Not visual class for mesh: {mesh_name}")

        for attr in geom_keys:
            if attr in geom.attrib:
                del geom.attrib[attr]

    for geom in root.findall(".//geom[@name]"):
        name = geom.get("name")
        if name is None:
            continue

        # Determine the class based on the mesh name
        if "collision" in name:
            geom.set("class", "collision")
        else:
            raise ValueError(f"Not collision class for name: {name}")

        for attr in geom_keys:
            if attr in geom.attrib:
                del geom.attrib[attr]


def add_default_settings(root: ET.Element):
    # Create or find the <default> element
    default = root.find("default")
    if default is not None:
        root.remove(default)

    default = ET.SubElement(root, "default")

    XM430_default = ET.SubElement(default, "default", {"class": "XM430"})
    ET.SubElement(XM430_default, "position", {"forcerange": "-3 3"})

    XC430_default = ET.SubElement(default, "default", {"class": "XC430"})
    ET.SubElement(XC430_default, "position", {"forcerange": "-2 2"})

    two_XC430_default = ET.SubElement(default, "default", {"class": "2XC430"})
    ET.SubElement(two_XC430_default, "position", {"forcerange": "-2 2"})

    XL430_default = ET.SubElement(default, "default", {"class": "XL430"})
    ET.SubElement(XL430_default, "position", {"forcerange": "-2 2"})

    two_XL430_default = ET.SubElement(default, "default", {"class": "2XL430"})
    ET.SubElement(two_XL430_default, "position", {"forcerange": "-2 2"})

    XC330_default = ET.SubElement(default, "default", {"class": "XC330"})
    ET.SubElement(XC330_default, "position", {"forcerange": "-1 1"})

    # Set <geom> settings
    # ET.SubElement(default, "geom", {"type": "mesh", "solref": ".004 1"})

    # Add <default class="visual"> settings
    visual_default = ET.SubElement(default, "default", {"class": "visual"})
    ET.SubElement(
        visual_default,
        "geom",
        {"type": "mesh", "contype": "0", "conaffinity": "0", "group": "2"},
    )

    # Add <default class="collision"> settings
    collision_default = ET.SubElement(default, "default", {"class": "collision"})
    # Group 3's visualization is diabled by default
    ET.SubElement(collision_default, "geom", {"type": "sphere", "group": "3"})


def include_all_contacts(root: ET.Element):
    contact = root.find("contact")
    if contact is not None:
        root.remove(contact)


def exclude_all_contacts(root: ET.Element):
    contact = root.find("contact")
    if contact is not None:
        root.remove(contact)

    contact = ET.SubElement(root, "contact")

    collision_bodies: List[str] = []
    for body in root.findall(".//body"):
        body_name = body.get("name")
        if body_name and body.find("./geom[@class='collision']") is not None:
            collision_bodies.append(body_name)

    for body1_name in collision_bodies:
        for body2_name in collision_bodies:
            if body1_name != body2_name:
                ET.SubElement(contact, "exclude", body1=body1_name, body2=body2_name)


def add_contacts(root: ET.Element, collision_config: Dict[str, Dict[str, Any]]):
    # Ensure there is a <contact> element
    contact = root.find("contact")
    if contact is not None:
        root.remove(contact)

    contact = ET.SubElement(root, "contact")

    collision_bodies: Dict[str, ET.Element] = {}
    for body in root.findall(".//body"):
        body_name = body.get("name")
        geom = body.find("./geom[@class='collision']")
        if body_name and geom is not None:
            collision_bodies[body_name] = geom

    pairs: List[Tuple[str, str]] = []
    excludes: List[Tuple[str, str]] = []

    collision_body_names = list(collision_bodies.keys())
    for body_name in collision_body_names:
        if "floor" in collision_config[body_name]["contact_pairs"]:
            pairs.append((body_name, "floor"))

    for i in range(len(collision_bodies) - 1):
        for j in range(i + 1, len(collision_body_names)):
            body1_name = collision_body_names[i]
            body2_name = collision_body_names[j]

            paired_1 = body2_name in collision_config[body1_name]["contact_pairs"]
            paired_2 = body1_name in collision_config[body2_name]["contact_pairs"]
            if paired_1 and paired_2:
                pairs.append((body1_name, body2_name))
            else:
                excludes.append((body1_name, body2_name))

    # Add all <pair> elements first
    for body1_name, body2_name in pairs:
        geom1_name = collision_bodies[body1_name].get("name")
        if body2_name == "floor":
            geom2_name = "floor"
        else:
            geom2_name = collision_bodies[body2_name].get("name")

        if geom1_name is None or geom2_name is None:
            raise ValueError(
                f"Could not find geom name for {body1_name} or {body2_name}"
            )

        ET.SubElement(contact, "pair", geom1=geom1_name, geom2=geom2_name)

    # Add all <exclude> elements after
    for body1_name, body2_name in excludes:
        ET.SubElement(contact, "exclude", body1=body1_name, body2=body2_name)


def add_waist_constraints(root: ET.Element, offsets: Dict[str, float]):
    # Ensure there is an <equality> element
    tendon = root.find("tendon")
    if tendon is not None:
        root.remove(tendon)

    tendon = ET.SubElement(root, "tendon")

    waist_roll_coef = round_to_sig_digits(offsets["waist_roll_coef"], 6)
    waist_yaw_coef = round_to_sig_digits(offsets["waist_yaw_coef"], 6)

    # waist roll
    fixed_roll = ET.SubElement(
        tendon, "fixed", name="waist_roll_coupling", limited="true", range="0 0.001"
    )
    ET.SubElement(fixed_roll, "joint", joint="waist_act_1", coef=f"{waist_roll_coef}")
    ET.SubElement(fixed_roll, "joint", joint="waist_act_2", coef=f"{-waist_roll_coef}")
    ET.SubElement(fixed_roll, "joint", joint="waist_roll", coef="1")

    # waist roll
    fixed_yaw = ET.SubElement(
        tendon, "fixed", name="waist_yaw_coupling", limited="true", range="0 0.001"
    )
    ET.SubElement(fixed_yaw, "joint", joint="waist_act_1", coef=f"{waist_yaw_coef}")
    ET.SubElement(fixed_yaw, "joint", joint="waist_act_2", coef=f"{waist_yaw_coef}")
    ET.SubElement(fixed_yaw, "joint", joint="waist_yaw", coef="1")


def add_knee_constraints(root: ET.Element):
    # Ensure there is an <equality> element
    equality = root.find("./equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    body_pairs: List[Tuple[str, str]] = [
        ("knee_rod", "bearing_683"),
        ("knee_rod_2", "bearing_683_2"),
        ("knee_rod_3", "bearing_683_3"),
        ("knee_rod_4", "bearing_683_4"),
    ]

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


def add_ankle_constraints(root: ET.Element, offsets: Dict[str, float]):
    # Ensure there is an <equality> element
    equality = root.find("./equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    body_pairs: List[Tuple[str, str]] = [
        ("ank_motor_arm", "ank_motor_rod_long"),
        ("ank_motor_arm_2", "ank_motor_rod_short"),
        ("ank_motor_arm_3", "ank_motor_rod_long_2"),
        ("ank_motor_arm_4", "ank_motor_rod_short_2"),
    ]

    # Add equality constraints for each pair
    for body1, body2 in body_pairs:
        ET.SubElement(
            equality,
            "connect",
            body1=body1,
            body2=body2,
            solimp="0.9999 0.9999 0.001 0.5 2",
            solref="0.0001 1",
            anchor=f"{offsets['ank_act_arm_r']} 0 {offsets['ank_act_arm_y']}",
        )


def add_actuators_to_mjcf(root: ET.Element, joints_config: Dict[str, Any]):
    # Create <actuator> element if it doesn't exist
    actuator = root.find("./actuator")
    if actuator is not None:
        root.remove(actuator)

    actuator = ET.SubElement(root, "actuator")

    for joint in root.findall(".//joint"):
        joint_name = joint.get("name")
        if joint_name in joints_config and "spec" in joints_config[joint_name]:
            if "_drive" in joint_name:
                joint_driven_name = joint_name.replace("_drive", "_driven")
                joint_driven = root.find(f".//joint[@name='{joint_driven_name}']")
                if joint_driven is None:
                    raise ValueError(
                        f"The driven joint {joint_driven_name} is not found"
                    )

                position = ET.SubElement(
                    actuator,
                    "position",
                    name=joint_name,
                    joint=joint_driven_name,
                    kp=str(joints_config[joint_name]["kp_sim"]),
                    gear=str(
                        round_to_sig_digits(
                            1 / joints_config[joint_name]["gear_ratio"], 6
                        )
                    ),
                    ctrlrange=joint_driven.get("range", "-3.141592 3.141592"),
                )
            else:
                position = ET.SubElement(
                    actuator,
                    "position",
                    name=joint_name,
                    joint=joint_name,
                    kp=str(joints_config[joint_name]["kp_sim"]),
                    ctrlrange=joint.get("range", "-3.141592 3.141592"),
                )

            position.set("class", joints_config[joint_name]["spec"])


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


def add_body_link(root: ET.Element, urdf_path: str, offsets: Dict[str, float]):
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
        pos=f"0 0 {offsets['torso_z']}",
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


def create_scene_xml(mjcf_path: str, is_fixed: bool):
    robot_name = os.path.basename(mjcf_path).replace(".xml", "")

    # Create the root element
    mujoco = ET.Element("mujoco", attrib={"model": f"{robot_name}_scene"})

    # Include the robot model
    ET.SubElement(mujoco, "include", attrib={"file": os.path.basename(mjcf_path)})

    # Add statistic element
    center_z = -0.05 if is_fixed else 0.25
    ET.SubElement(
        mujoco, "statistic", attrib={"center": f"0 0 {center_z}", "extent": "0.6"}
    )

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

    worldbody = ET.SubElement(mujoco, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        attrib={"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"},
    )
    ET.SubElement(
        worldbody,
        "camera",
        attrib={
            "name": "side",
            "pos": "0 -1 1",
            "xyaxes": "1 0 0 0 1 2",
            "mode": "trackcom",
        },
    )

    if not is_fixed:
        # Worldbody settings
        ET.SubElement(
            worldbody,
            "geom",
            attrib={
                "name": "floor",
                "size": "0 0 0.05",
                "type": "plane",
                "material": "groundplane",
                "condim": "3",
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

    # Create a tree from the root element and write it to a file
    tree = ET.ElementTree(mujoco)
    tree.write(os.path.join(os.path.dirname(mjcf_path), f"{robot_name}_scene.xml"))


def process_mjcf_fixed_file(root: ET.Element, robot: Robot):
    update_compiler_settings(root)
    add_option_settings(root)

    if robot.config["general"]["use_torso_site"]:
        add_torso_site(root)

    if robot.config["general"]["has_imu"]:
        add_imu_sensor(root)

    update_joint_params(root, robot.config["joints"])
    update_geom_classes(root, ["contype", "conaffinity", "group", "density"])
    add_actuators_to_mjcf(root, robot.config["joints"])

    if robot.config["general"]["is_waist_closed_loop"]:
        add_waist_constraints(root, robot.config["general"]["offsets"])

    if robot.config["general"]["is_knee_closed_loop"]:
        add_knee_constraints(root)

    if robot.config["general"]["is_ankle_closed_loop"]:
        add_ankle_constraints(root, robot.config["general"]["offsets"])

    add_default_settings(root)
    # include_all_contacts(root)
    exclude_all_contacts(root)


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

    process_mjcf_fixed_file(xml_root, robot)
    xml_tree.write(mjcf_fixed_path)

    if robot.config["general"]["is_fixed"]:
        mjcf_path = mjcf_fixed_path
    else:
        create_scene_xml(mjcf_fixed_path, is_fixed=True)

        mjcf_path = os.path.join(robot_dir, robot_name + ".xml")
        add_body_link(xml_root, urdf_path, robot.config["general"]["offsets"])
        add_contacts(xml_root, robot.collision_config)
        xml_tree.write(mjcf_path)

    create_scene_xml(mjcf_path, robot.config["general"]["is_fixed"])


def main():
    parser = argparse.ArgumentParser(description="Process the MJCF.")
    parser.add_argument(
        "--robot",
        type=str,
        default="sysID_XC430",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    args = parser.parse_args()

    get_mjcf_files(args.robot)


if __name__ == "__main__":
    main()
