import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple
from xml.dom.minidom import parseString

from transforms3d.euler import euler2quat

from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import round_to_sig_digits

# This script contains utility functions for modifying and writing MJCF XML files.


def pretty_write_xml(root: ET.Element, file_path: str):
    """Formats an XML Element into a pretty-printed XML string and writes it to a specified file.

    Args:
        root (ET.Element): The root element of the XML tree to be formatted.
        file_path (str): The path to the file where the formatted XML will be written.
    """
    # Convert the Element or ElementTree to a string
    xml_str = ET.tostring(root, encoding="utf-8").decode("utf-8")

    # Parse and pretty-print the XML string
    dom = parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")

    # Remove blank lines
    pretty_xml = "\n".join([line for line in pretty_xml.splitlines() if line.strip()])

    # Write the pretty XML to the file
    with open(file_path, "w") as file:
        file.write(pretty_xml)


def find_root_link_name(root: ET.Element):
    """Finds the root link name in a URDF XML structure.

    Args:
        root (ET.Element): The root element of the URDF XML tree.

    Returns:
        str: The name of the root link.

    Raises:
        ValueError: If no root link can be identified in the URDF.
    """
    child_links = {joint.find("child").get("link") for joint in root.findall("joint")}  # type: ignore
    all_links = {link.get("name") for link in root.findall("link")}

    # The root link is the one not listed as a child
    root_link = all_links - child_links
    if root_link:
        return str(root_link.pop())
    else:
        raise ValueError("Could not find root link in URDF")


def replace_mesh_file(root: ET.Element, old_file: str, new_file: str):
    """Replaces occurrences of a specified mesh file name with a new file name in an XML structure.

    Args:
        root (ET.Element): The root element of the XML tree to search within.
        old_file (str): The file name to be replaced.
        new_file (str): The new file name to replace the old file name with.
    """
    # Find all mesh elements
    for mesh in root.findall(".//mesh"):
        # Check if the file attribute matches the old file name
        if mesh.get("file") == old_file:
            # Replace with the new file name
            mesh.set("file", new_file)


def update_compiler_settings(root: ET.Element):
    """Updates the compiler settings in the given XML element.

    This function searches for a 'compiler' element within the provided XML root element and updates its 'autolimits' attribute to 'true'. If no 'compiler' element is found, a ValueError is raised.

    Args:
        root (ET.Element): The root XML element containing the compiler settings.

    Raises:
        ValueError: If no 'compiler' element is found in the XML.
    """
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("No compiler element found in the XML.")

    compiler.set("autolimits", "true")


def add_option_settings(root: ET.Element):
    """Adds or updates the 'option' settings in the given XML element.

    This function searches for an 'option' subelement within the provided XML root element. If found, it removes the existing 'option' element. It then creates a new 'option' subelement with a 'flag' child element that has an attribute 'eulerdamp' set to 'disable'.

    Args:
        root (ET.Element): The root XML element to modify.
    """
    option = root.find("option")
    if option is not None:
        root.remove(option)

    option = ET.SubElement(root, "option")

    ET.SubElement(option, "flag", {"eulerdamp": "disable"})


def add_imu_sensor(root: ET.Element, general_config: Dict[str, Any]):
    """Adds an IMU sensor site to the worldbody element of an XML tree.

    This function inserts a new site element representing an IMU sensor into the
    worldbody of the provided XML tree. The position and orientation of the sensor
    are determined by the offsets specified in the general configuration.

    Args:
        root (ET.Element): The root element of the XML tree, expected to contain a
            worldbody element.
        general_config (Dict[str, Any]): A dictionary containing configuration
            details, including offsets for the IMU sensor's position and orientation.

    Raises:
        ValueError: If the worldbody element is not found in the XML tree.
    """
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("No worldbody element found in the XML.")

    offsets = general_config["offsets"]
    site_attributes = {
        "name": "imu",
        "type": "box",
        "size": "0.0128 0.0128 0.0008",
        "pos": f"{offsets['imu_x']} {offsets['imu_y']} {offsets['imu_z']}",
        "zaxis": offsets["imu_zaxis"],
    }
    site_element = ET.Element("site", site_attributes)
    worldbody.insert(0, site_element)

    # sensor = root.find("sensor")
    # if sensor is not None:
    #     root.remove(sensor)

    # sensor = ET.SubElement(root, "sensor")

    # # Adding framequat sub-element
    # ET.SubElement(
    #     sensor,
    #     "framequat",
    #     attrib={
    #         "name": "orientation",
    #         "objtype": "site",
    #         "noise": "0.001",
    #         "objname": "imu",
    #     },
    # )

    # # Adding framepos sub-element
    # ET.SubElement(
    #     sensor,
    #     "framepos",
    #     attrib={
    #         "name": "position",
    #         "objtype": "site",
    #         "noise": "0.001",
    #         "objname": "imu",
    #     },
    # )

    # # Adding gyro sub-element
    # ET.SubElement(
    #     sensor,
    #     "gyro",
    #     attrib={
    #         "name": "angular_velocity",
    #         "site": "imu",
    #         "noise": "0.005",
    #         "cutoff": "34.9",
    #     },
    # )

    # # Adding velocimeter sub-element
    # ET.SubElement(
    #     sensor,
    #     "velocimeter",
    #     attrib={
    #         "name": "linear_velocity",
    #         "site": "imu",
    #         "noise": "0.001",
    #         "cutoff": "30",
    #     },
    # )

    # # Adding accelerometer sub-element
    # ET.SubElement(
    #     sensor,
    #     "accelerometer",
    #     attrib={
    #         "name": "linear_acceleration",
    #         "site": "imu",
    #         "noise": "0.005",
    #         "cutoff": "157",
    #     },
    # )

    # # Adding magnetometer sub-element
    # ET.SubElement(
    #     sensor, "magnetometer", attrib={"name": "magnetometer", "site": "imu"}
    # )


def update_joint_params(root: ET.Element, joints_config: Dict[str, Any]):
    """Updates joint parameters in an XML structure based on a given configuration.

    Args:
        root (ET.Element): The root element of the XML structure containing joint elements.
        joints_config (Dict[str, Any]): A dictionary mapping joint names to their configuration attributes and values. Attributes can include 'damping', 'armature', and 'frictionloss'.
    """
    # Iterate over all joints in the XML
    for joint in root.findall(".//joint"):
        joint_name = joint.get("name")

        # Check if the "actuatorfrcrange" attribute exists
        if "actuatorfrcrange" in joint.attrib:
            # Remove the attribute using the `del` keyword
            del joint.attrib["actuatorfrcrange"]

        if joint_name in joints_config:
            for attr_name in joints_config[joint_name]:
                if attr_name in ["damping", "armature", "frictionloss"]:
                    attr_value = round_to_sig_digits(
                        joints_config[joint_name][attr_name], 6
                    )
                    joint.set(attr_name, str(attr_value))


def update_geom_classes(root: ET.Element, geom_keys: List[str]):
    """Updates the class attribute of geometry elements in an XML tree.

    Args:
        root (ET.Element): The root element of the XML tree containing geometry elements.
        geom_keys (List[str]): A list of attribute keys to be removed from each geometry element.

    Raises:
        ValueError: If a geometry element's name does not contain "visual" or "collision".
    """
    for geom in root.findall(".//geom"):
        name: str | None = geom.get("name")
        if name is None:
            name = geom.get("mesh")
            if name is None:
                continue

        # Determine the class based on the mesh name
        if "visual" in name:
            geom.set("class", "visual")
        elif "collision" in name:
            geom.set("class", "collision")
        else:
            raise ValueError(f"Not collision class for name: {name}")

        for attr in geom_keys:
            if attr in geom.attrib:
                del geom.attrib[attr]


def add_keyframes(root: ET.Element, robot: Robot, is_fixed: bool):
    """Adds keyframes to the given XML element for a robot's configuration.

    This function modifies the provided XML element by creating or updating a
    <keyframe> element based on the robot's configuration and whether the robot
    is fixed. It constructs a string representing the robot's joint positions
    (`qpos`) and assigns it to the keyframe.

    Args:
        root (ET.Element): The root XML element to which the keyframe will be added.
        robot (Robot): The robot object containing configuration and motor ordering.
        is_fixed (bool): A flag indicating whether the robot is fixed in place.
    """
    # Create or find the <default> element
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)

    keyframe = ET.SubElement(root, "keyframe")

    general_config = robot.config["general"]

    has_lower_body = "arms" not in robot.name
    has_upper_body = "legs" not in robot.name
    has_gripper = False
    for motor_name in robot.motor_ordering:
        if "gripper" in motor_name:
            has_gripper = True

    if is_fixed:
        qpos_str = ""
    else:
        qpos_str = f"0 0 {general_config['offsets']['default_torso_z']} 1 0 0 0 "

    if has_upper_body and has_lower_body:  # neck
        if "active" in robot.name:
            qpos_str += "0 0 "
        elif general_config["is_neck_closed_loop"]:
            qpos_str += "0 0 0 0 0 0 0 0 "
        else:
            qpos_str += "0 0 0 0 "

    if has_lower_body:  # waist and legs
        if general_config["is_waist_closed_loop"]:
            qpos_str += "0 0 0 0 "
        else:
            qpos_str += "0 0 "

        if "active" in robot.name:
            qpos_str += (
                "0.145689 0 0 -0.534732 -0.379457 0 "
                + "-0.145689 0 0 0.534732 0.379457 0 "
            )
        else:
            qpos_str += (
                "0.145689 0 0 0 -0.534732 -0.534757 -0.534707 -0.379457 0 -0.534732 -0.534757 -0.534707 "
                + "-0.145689 0 0 0 0.534732 0.534757 0.534707 0.379457 0 0.534732 0.534757 0.534707 "
            )

    if has_upper_body:  # arms
        if "active" in robot.name:
            qpos_str += "0.174533 -0.261799 -1.0472 0.523599 1.0472 1.309 0 "
        else:
            qpos_str += "0.174533 -0.261799 1.0472 -1.0472 0.523599 -1.0472 1.0472 1.309 -1.309 0 "

        if has_gripper:
            qpos_str += "0 0 0 "

        if "active" in robot.name:
            qpos_str += "-0.174533 -0.261799 1.0472 0.523599 -1.0472 -1.309 0 "
        else:
            qpos_str += "-0.174533 -0.261799 -1.0472 1.0472 0.523599 1.0472 -1.0472 -1.309 1.309 0"

        if has_gripper:
            qpos_str += " 0 0 0"

    ET.SubElement(keyframe, "key", {"name": "home", "qpos": qpos_str})


def add_default_settings(
    root: ET.Element,
    general_config: Dict[str, Any],
    joints_config: Dict[str, Any],
    actuator_type: str,
):
    """Adds default settings to an XML element tree for a simulation environment.

    This function modifies the provided XML root element by adding or updating
    a `<default>` element with specific settings for visual, collision, and motor
    configurations based on the provided general and joint configurations.

    Args:
        root (ET.Element): The root XML element to which default settings will be added.
        general_config (Dict[str, Any]): A dictionary containing general configuration
            parameters, such as solution reference values.
        joints_config (Dict[str, Any]): A dictionary containing joint-specific configuration
            parameters, including motor specifications.
        actuator_type (str): The type of actuator to configure, either 'motor' or another
            type, which determines the range settings applied to the actuators.
    """
    # Create or find the <default> element
    default = root.find("default")
    if default is not None:
        root.remove(default)

    default = ET.SubElement(root, "default")

    # ET.SubElement(
    #     default,
    #     "geom",
    #     {
    #         "type": "mesh",
    #         "solref": f"{general_config['solref'][0]} {general_config['solref'][1]}",
    #     },
    # )
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
    ET.SubElement(collision_default, "geom", {"group": "3"})

    for motor_name, torque_limit in zip(
        ["XM430", "XC430", "2XC430", "XL430", "2XL430", "XC330"], [3, 2, 2, 2, 2, 1]
    ):
        has_motor = False
        for joint_config in joints_config.values():
            if "spec" not in joint_config:
                continue

            if joint_config["spec"] == motor_name:
                has_motor = True
                break

        if has_motor:
            motor_default = ET.SubElement(default, "default", {"class": motor_name})
            if actuator_type == "motor":
                ET.SubElement(
                    motor_default,
                    actuator_type,
                    {"ctrlrange": f"-{torque_limit} {torque_limit}"},
                )
            else:
                ET.SubElement(
                    motor_default,
                    actuator_type,
                    {"forcerange": f"-{torque_limit} {torque_limit}"},
                )


def include_all_contacts(root: ET.Element):
    """Removes the first 'contact' element from the given XML root element if it exists.

    Args:
        root (ET.Element): The root element of an XML tree from which the 'contact' element will be removed.
    """
    contact = root.find("contact")
    if contact is not None:
        root.remove(contact)


def exclude_all_contacts(root: ET.Element):
    """Removes existing contact elements and creates new exclusions for all pairs of collision bodies in the XML tree.

    Args:
        root (ET.Element): The root element of the XML tree to modify.

    Modifies:
        The XML tree by removing the existing "contact" element and adding a new one with "exclude" sub-elements for each pair of bodies that have collision geometries.
    """
    contact = root.find("contact")
    if contact is not None:
        root.remove(contact)

    contact = ET.SubElement(root, "contact")

    collision_bodies: List[str] = []
    for body in root.findall(".//body"):
        body_name = body.get("name")
        if body_name and body.find("./geom[@class='collision']") is not None:
            collision_bodies.append(body_name)

    for i in range(len(collision_bodies) - 1):
        for j in range(i + 1, len(collision_bodies)):
            ET.SubElement(
                contact, "exclude", body1=collision_bodies[i], body2=collision_bodies[j]
            )


def add_contacts(root: ET.Element, collision_config: Dict[str, Dict[str, Any]]):
    """Adds contact and exclusion pairs to an XML element based on a collision configuration.

    Args:
        root (ET.Element): The root XML element to which contact pairs and exclusions will be added.
        collision_config (Dict[str, Dict[str, Any]]): A dictionary containing collision configuration for each body. Each entry specifies which other bodies it can contact with.

    Raises:
        ValueError: If a geometry name cannot be found for any of the specified body pairs.
    """
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
        geom2_name: str | None = None
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


def add_neck_constraints(root: ET.Element, general_config: Dict[str, Any]):
    """Adds neck constraints to an XML element by creating equality constraints between specified body pairs.

    Args:
        root (ET.Element): The root XML element to which the neck constraints will be added.
        general_config (Dict[str, Any]): A dictionary containing configuration parameters, specifically 'solref', which is used to set the solref attribute for the constraints.
    """
    # Ensure there is an <equality> element
    equality = root.find("./equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    body_pairs: List[Tuple[str, str]] = [
        ("neck_rod", "bearing_683"),
        ("neck_rod_2", "bearing_683_2"),
    ]

    # Add equality constraints for each pair
    for body1, body2 in body_pairs:
        ET.SubElement(
            equality,
            "weld",
            body1=body1,
            body2=body2,
            solimp="0.9999 0.9999 0.001 0.5 2",
            solref=f"{general_config['solref'][0]} {general_config['solref'][1]}",
        )


def add_waist_constraints(root: ET.Element, general_config: Dict[str, Any]):
    """Adds waist constraints to the given XML element by creating and configuring tendon elements for waist roll and yaw.

    This function modifies the provided XML element by removing any existing 'tendon' element and adding a new one with specific constraints for waist roll and yaw. The constraints are defined using coefficients and backlash values from the general configuration.

    Args:
        root (ET.Element): The root XML element to which the waist constraints will be added.
        general_config (Dict[str, Any]): A dictionary containing configuration values, including offsets and backlash for waist roll and yaw.
    """
    # Ensure there is an <equality> element
    tendon = root.find("tendon")
    if tendon is not None:
        root.remove(tendon)

    tendon = ET.SubElement(root, "tendon")

    offsets = general_config["offsets"]
    waist_roll_coef = round_to_sig_digits(offsets["waist_roll_coef"], 6)
    waist_yaw_coef = round_to_sig_digits(offsets["waist_yaw_coef"], 6)

    waist_roll_backlash = general_config["waist_roll_backlash"]
    waist_yaw_backlash = general_config["waist_yaw_backlash"]
    # waist roll
    fixed_roll = ET.SubElement(
        tendon,
        "fixed",
        name="waist_roll_coupling",
        limited="true",
        range=f"-{waist_roll_backlash} {waist_roll_backlash}",
    )
    ET.SubElement(fixed_roll, "joint", joint="waist_act_1", coef=f"{waist_roll_coef}")
    ET.SubElement(fixed_roll, "joint", joint="waist_act_2", coef=f"{-waist_roll_coef}")
    ET.SubElement(fixed_roll, "joint", joint="waist_roll", coef="1")

    # waist roll
    fixed_yaw = ET.SubElement(
        tendon,
        "fixed",
        name="waist_yaw_coupling",
        limited="true",
        range=f"-{waist_yaw_backlash} {waist_yaw_backlash}",
    )
    ET.SubElement(fixed_yaw, "joint", joint="waist_act_1", coef=f"{-waist_yaw_coef}")
    ET.SubElement(fixed_yaw, "joint", joint="waist_act_2", coef=f"{-waist_yaw_coef}")
    ET.SubElement(fixed_yaw, "joint", joint="waist_yaw", coef="1")


def add_knee_constraints(root: ET.Element, general_config: Dict[str, Any]):
    """Adds knee constraints to an XML structure by ensuring the presence of an `<equality>` element and appending `<weld>` elements for specified body pairs.

    Args:
        root (ET.Element): The root element of the XML structure to which constraints are added.
        general_config (Dict[str, Any]): Configuration dictionary containing parameters for the constraints, specifically the 'solref' values.
    """
    # Ensure there is an <equality> element
    equality = root.find("./equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    body_pairs: List[Tuple[str, str]] = [
        ("knee_rod", "bearing_683_3"),
        ("knee_rod_2", "bearing_683_4"),
        ("knee_rod_3", "bearing_683_5"),
        ("knee_rod_4", "bearing_683_6"),
    ]

    # Add equality constraints for each pair
    for body1, body2 in body_pairs:
        ET.SubElement(
            equality,
            "weld",
            body1=body1,
            body2=body2,
            solimp="0.9999 0.9999 0.001 0.5 2",
            solref=f"{general_config['solref'][0]} {general_config['solref'][1]}",
        )


def add_ankle_constraints(root: ET.Element, general_config: Dict[str, Any]):
    """Adds ankle constraints to an XML structure by creating or modifying an `<equality>` element with specified body pairs and configuration parameters.

    Args:
        root (ET.Element): The root element of the XML structure to which the ankle constraints will be added.
        general_config (Dict[str, Any]): A dictionary containing configuration parameters, including offsets and solver settings for the constraints.
    """
    # Ensure there is an <equality> element
    equality = root.find("./equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    offsets = general_config["offsets"]
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
            solimp=f"{general_config['ank_solimp_0']} 0.9999 0.001 0.5 2",
            solref=f"{general_config['ank_solref_0']} {general_config['solref'][1]}",
            anchor=f"{offsets['ank_act_arm_r']} 0 {offsets['ank_act_arm_y']}",
        )


def add_joint_constraints(
    root: ET.Element, general_config: Dict[str, Any], joints_config: Dict[str, Any]
):
    """Adds joint constraints to an XML element based on the provided configuration.

    Args:
        root (ET.Element): The root XML element to which joint constraints will be added.
        general_config (Dict[str, Any]): General configuration containing parameters like 'solref'.
        joints_config (Dict[str, Any]): Configuration for each joint, specifying details such as 'transmission' type and 'gear_ratio'.

    Raises:
        ValueError: If a required driven or pinion joint is not found in the XML structure.
    """
    equality = root.find("./equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    for joint_name, joint_config in joints_config.items():
        if "spec" not in joint_config:
            continue

        transmission = joint_config["transmission"]
        if transmission == "gear":
            joint_driven_name = joint_name.replace("_drive", "_driven")
            joint_driven: ET.Element | None = root.find(
                f".//joint[@name='{joint_driven_name}']"
            )
            if joint_driven is None:
                raise ValueError(f"The driven joint {joint_driven_name} is not found")

            gear_ratio = round_to_sig_digits(
                -joints_config[joint_name]["gear_ratio"], 6
            )
            ET.SubElement(
                equality,
                "joint",
                joint1=joint_driven_name,
                joint2=joint_name,
                polycoef=f"0 {gear_ratio} 0 0 0",
                solimp="0.9999 0.9999 0.001 0.5 2",
                solref=f"{general_config['solref'][0]} {general_config['solref'][1]}",
            )
        elif transmission == "rack_and_pinion":
            joint_pinion_1_name = joint_name.replace("_rack", "_pinion")
            joint_pinion_2_name = joint_name.replace("_rack", "_pinion_mirror")
            for joint_pinion_name in [joint_pinion_1_name, joint_pinion_2_name]:
                joint_pinion: ET.Element | None = root.find(
                    f".//joint[@name='{joint_pinion_name}']"
                )
                if joint_pinion is None:
                    raise ValueError(
                        f"The pinion joint {joint_pinion_name} is not found"
                    )

                gear_ratio = round_to_sig_digits(
                    -joints_config[joint_name]["gear_ratio"], 6
                )

                ET.SubElement(
                    equality,
                    "joint",
                    joint1=joint_pinion_name,
                    joint2=joint_name,
                    polycoef=f"0 {gear_ratio} 0 0 0",
                    solimp="0.9999 0.9999 0.001 0.5 2",
                    solref=f"{general_config['solref'][0]} {general_config['solref'][1]}",
                )


def add_position_actuators_to_mjcf(root: ET.Element, joints_config: Dict[str, Any]):
    """Adds position actuators to the MJCF model based on the provided joint configurations.

    This function modifies the given MJCF XML tree by adding or updating the `<actuator>` element
    with `<position>` actuators for each joint specified in the `joints_config` dictionary. Each
    actuator is configured with properties such as `kp`, `ctrlrange`, and `class` based on the
    joint's configuration.

    Args:
        root (ET.Element): The root element of the MJCF XML tree.
        joints_config (Dict[str, Any]): A dictionary where keys are joint names and values are
            dictionaries containing joint configuration details, including 'spec' and 'kp_sim'.

    Raises:
        ValueError: If a joint specified in `joints_config` is not found in the MJCF model.
    """
    # Create <actuator> element if it doesn't exist
    actuator = root.find("./actuator")
    if actuator is not None:
        root.remove(actuator)

    actuator = ET.SubElement(root, "actuator")

    for joint_name, joint_config in joints_config.items():
        if "spec" not in joint_config:
            continue

        joint: ET.Element | None = root.find(f".//joint[@name='{joint_name}']")
        if joint is None:
            raise ValueError(f"The joint {joint_name} is not found")

        position = ET.SubElement(
            actuator,
            "position",
            name=joint_name,
            joint=joint_name,
            kp=str(joints_config[joint_name]["kp_sim"]),
            # kv=str(joints_config[joint_name]["kd_sim"]),
            ctrlrange=joint.get("range", "-3.141592 3.141592"),
        )

        position.set("class", joints_config[joint_name]["spec"])


def add_motor_actuators_to_mjcf(root: ET.Element, joints_config: Dict[str, Any]):
    """Adds motor actuators to the MJCF XML structure based on the provided joint configurations.

    This function modifies the given MJCF XML root element by creating or updating an `<actuator>` element. It iterates over the provided joint configurations and adds a `<motor>` element for each joint that has a specified configuration. If a joint specified in the configuration is not found in the XML, a `ValueError` is raised.

    Args:
        root (ET.Element): The root element of the MJCF XML structure.
        joints_config (Dict[str, Any]): A dictionary containing joint names as keys and their configuration details as values. Each configuration must include a "spec" key to be considered valid.

    Raises:
        ValueError: If a joint specified in the `joints_config` is not found in the MJCF XML structure.
    """
    # Create <actuator> element if it doesn't exist
    actuator = root.find("./actuator")
    if actuator is not None:
        root.remove(actuator)

    actuator = ET.SubElement(root, "actuator")

    for joint_name, joint_config in joints_config.items():
        if "spec" not in joint_config:
            continue

        joint: ET.Element | None = root.find(f".//joint[@name='{joint_name}']")
        if joint is None:
            raise ValueError(f"The joint {joint_name} is not found")

        motor = ET.SubElement(actuator, "motor", name=joint_name, joint=joint_name)

        motor.set("class", joints_config[joint_name]["spec"])


def parse_urdf_body_link(root: ET.Element, root_link_name: str):
    """Parses the URDF body link to extract inertial properties.

    Args:
        root (ET.Element): The root element of the URDF XML structure.
        root_link_name (str): The name of the link to extract properties from.

    Returns:
        dict or None: A dictionary containing the position, quaternion, mass, and diagonal inertia of the link if found; otherwise, None.
    """
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
        quat = euler2quat(*[float(x) for x in origin["rpy"].split(" ")])
        diaginertia = [
            float(x) for x in [inertia["ixx"], inertia["iyy"], inertia["izz"]]
        ]
        properties = {
            "pos": " ".join([f"{round_to_sig_digits(x, 6)}" for x in pos]),
            "quat": " ".join([f"{round_to_sig_digits(x, 6)}" for x in quat]),
            "mass": f"{round_to_sig_digits(float(mass), 6)}",
            "diaginertia": " ".join(f"{x:.5e}" for x in diaginertia),
        }
        return properties


def add_body_link(root: ET.Element, urdf_path: str, offsets: Dict[str, float]):
    """Adds a body link to the XML tree based on URDF file specifications.

    Args:
        root (ET.Element): The root element of the XML tree to which the body link will be added.
        urdf_path (str): The file path to the URDF file containing the robot's description.
        offsets (Dict[str, float]): A dictionary containing offset values, specifically for the 'torso_z' position.

    Raises:
        ValueError: If no 'worldbody' element is found in the XML tree.

    Prints:
        A message if no inertial properties are found in the URDF file.
    """
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


def add_ee_sites(root: ET.Element, ee_name: str):
    """Adds end-effector (EE) sites to the XML structure based on the specified configuration.

    Args:
        root (ET.Element): The root element of the XML tree to which EE sites will be added.
        ee_name (str): The name of the end-effector to identify target geometries.

    The function identifies geometries in the XML that are relevant to the end-effector based on naming conventions and adds site elements to these geometries. The sites are positioned relative to the geometries and are configured with predefined specifications.
    """

    def is_ee_collision(geom_name: str) -> bool:
        if "gripper" in ee_name:
            return "rail_guide" in geom_name
        else:
            return ee_name in geom_name and "collision" in geom_name

    # Site specifications
    site_specifications = {"type": "sphere", "size": "0.005", "rgba": "0.9 0.1 0.1 0.8"}

    target_geoms: Dict[str, Tuple[ET.Element, ET.Element]] = {}
    for parent in root.iter():
        for geom in parent.findall("geom"):
            name = geom.attrib.get("name", "")
            if is_ee_collision(name):
                # Store both the parent and the geom in the dictionary
                target_geoms[name] = (parent, geom)

    for i, (name, (parent, target_geom)) in enumerate(target_geoms.items()):
        geom_pos = list(map(float, target_geom.attrib["pos"].split()))
        geom_size = list(map(float, target_geom.attrib["size"].split()))

        if "gripper" in ee_name:
            bottom_center_pos = [
                geom_pos[0],
                geom_pos[1] - 2 * geom_size[2]
                if i == 0
                else geom_pos[1] + 2 * geom_size[2],
                geom_pos[2],
            ]
        else:
            bottom_center_pos = [
                geom_pos[0],
                geom_pos[1] + geom_size[1]
                if geom_pos[1] > 0
                else geom_pos[1] - geom_size[1],
                geom_pos[2],
            ]

        ET.SubElement(
            parent,
            "site",
            {
                "name": "left_ee_center" if i == 0 else "right_ee_center",
                "pos": " ".join([str(x) for x in bottom_center_pos]),
                **site_specifications,
            },
        )


def add_foot_sites(root: ET.Element, foot_name: str):
    """Adds foot sites to the XML structure based on the specified foot name in the configuration.

    Args:
        root (ET.Element): The root element of the XML structure to which foot sites will be added.
        foot_name (str): The name of the foot to identify target geometries.

    The function searches for geometries in the XML structure that match the specified foot name and contain "collision" in their name. For each matching geometry, it calculates the position for a new site and adds it as a child element to the geometry's parent, with predefined specifications for type, size, and color.
    """
    # Site specifications
    site_specifications = {"type": "sphere", "size": "0.005", "rgba": "0.9 0.1 0.1 0.8"}

    target_geoms: Dict[str, Tuple[ET.Element, ET.Element]] = {}
    for parent in root.iter():
        for geom in parent.findall("geom"):
            name = geom.attrib.get("name", "")
            if foot_name in name and "collision" in name:
                # Store both the parent and the geom in the dictionary
                target_geoms[name] = (parent, geom)

    for i, (name, (parent, target_geom)) in enumerate(target_geoms.items()):
        geom_pos = list(map(float, target_geom.attrib["pos"].split()))
        geom_size = list(map(float, target_geom.attrib["size"].split()))

        bottom_center_pos = [geom_pos[0] - geom_size[0], geom_pos[1], geom_pos[2]]

        ET.SubElement(
            parent,
            "site",
            {
                "name": "left_foot_center" if i == 0 else "right_foot_center",
                "pos": " ".join([str(x) for x in bottom_center_pos]),
                **site_specifications,
            },
        )


def replace_box_collision(root: ET.Element, general_config: Dict[str, Any]):
    """Replaces box-shaped collision geometries with sphere-shaped ones in an XML structure.

    This function searches for geometries within an XML element that match a specified naming pattern, indicating they are foot-related collision boxes. It then replaces each box with four spheres positioned at the corners of the original box. The function also updates the contact pairs in the XML to reflect these changes.

    Args:
        root (ET.Element): The root element of the XML tree to be modified.
        general_config (Dict[str, Any]): Configuration dictionary containing:
            - "foot_name" (str): Substring to identify target geometries.
            - "is_ankle_closed_loop" (bool): Determines the positioning of spheres.

    Raises:
        ValueError: If no geometries matching the specified naming pattern are found.
    """
    # Search for the target geom using the substring condition
    foot_name = general_config["foot_name"]

    target_geoms: Dict[str, Tuple[ET.Element, ET.Element]] = {}
    for parent in root.iter():
        for geom in parent.findall("geom"):
            name = geom.attrib.get("name", "")
            if foot_name in name and "collision" in name:
                # Store both the parent and the geom in the dictionary
                target_geoms[name] = (parent, geom)

    if len(target_geoms) == 0:
        raise ValueError(f"Could not find geom with name containing '{foot_name}'")

    for name, (parent, target_geom) in target_geoms.items():
        pos = list(map(float, target_geom.attrib["pos"].split()))
        size = list(map(float, target_geom.attrib["size"].split()))

        # Compute the radius for the spheres based on the box
        sphere_radius = 0.004
        x_offset = size[0] - sphere_radius
        y_offset = size[1] - sphere_radius
        z_offset = size[2] - sphere_radius

        if general_config["is_ankle_closed_loop"]:
            # Positions for the four corner balls
            ball_positions = [
                [pos[0] - x_offset, pos[1] + y_offset, pos[2] - z_offset],
                [pos[0] + x_offset, pos[1] + y_offset, pos[2] - z_offset],
                [pos[0] - x_offset, pos[1] + y_offset, pos[2] + z_offset],
                [pos[0] + x_offset, pos[1] + y_offset, pos[2] + z_offset],
            ]
        else:
            # Positions for the four corner balls
            ball_positions = [
                [pos[0] - x_offset, pos[1] - y_offset, pos[2] - z_offset],
                [pos[0] - x_offset, pos[1] + y_offset, pos[2] - z_offset],
                [pos[0] - x_offset, pos[1] - y_offset, pos[2] + z_offset],
                [pos[0] - x_offset, pos[1] + y_offset, pos[2] + z_offset],
            ]

        # Create the new sphere elements at each corner
        for i, ball_pos in enumerate(ball_positions):
            ball_pos = [round_to_sig_digits(x, 6) for x in ball_pos]
            sphere = ET.Element(
                "geom",
                {
                    "name": f"{name}_ball_{i + 1}",
                    "type": "sphere",
                    "size": f"{sphere_radius}",
                    "pos": f"{ball_pos[0]} {ball_pos[1]} {ball_pos[2]}",
                    "rgba": target_geom.attrib["rgba"],
                    "class": target_geom.attrib["class"],
                },
            )
            parent.append(sphere)

        # Remove the original box geom
        parent.remove(target_geom)

    # Now update the contact section based on the replacement
    contact = root.find(".//contact")

    if contact is not None:
        target_names = list(target_geoms.keys())
        for pair in contact.findall("pair"):
            geom1 = pair.attrib.get("geom1")
            geom2 = pair.attrib.get("geom2")

            if geom1 is None or geom2 is None:
                continue

            # Check if any of the geoms match the one we are replacing
            if geom1 in target_names or geom2 in target_names:
                # Remove the old contact pair
                contact.remove(pair)

                # Add new contact pairs with the four balls
                for i in range(1, 5):
                    if geom1 in target_names:
                        contact.append(
                            ET.Element(
                                "pair", {"geom1": f"{geom1}_ball_{i}", "geom2": geom2}
                            )
                        )
                    if geom2 in target_names:
                        contact.append(
                            ET.Element(
                                "pair", {"geom1": geom1, "geom2": f"{geom2}_ball_{i}"}
                            )
                        )


def create_scene_xml(mjcf_path: str, is_fixed: bool):
    """Generates an XML scene file for a robot model based on the provided MJCF file path and configuration.

    Args:
        mjcf_path (str): The file path to the MJCF XML file of the robot model.
        is_fixed (bool): A flag indicating whether the robot is fixed in place. If True, adjusts camera positions and scene settings accordingly.

    Creates an XML scene file that includes the robot model, visual settings, and camera configurations. The scene is saved in the same directory as the input MJCF file with a modified filename.
    """
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

    camera_settings: Dict[str, Dict[str, List[float]]] = {
        "perspective": {"pos": [0.7, -0.7, 0.7], "xy_axes": [1, 1, 0, -1, 1, 3]},
        "side": {"pos": [0, -1, 0.6], "xy_axes": [1, 0, 0, 0, 1, 3]},
        "top": {"pos": [0, 0, 1], "xy_axes": [0, 1, 0, -1, 0, 0]},
        "front": {"pos": [1, 0, 0.6], "xy_axes": [0, 1, 0, -1, 0, 3]},
    }

    for camera, settings in camera_settings.items():
        pos_list = settings["pos"]
        if is_fixed:
            pos_list = [pos_list[0], pos_list[1], pos_list[2] - 0.35]

        pos_str = " ".join(map(str, pos_list))
        xy_axes_str = " ".join(map(str, settings["xy_axes"]))

        ET.SubElement(
            worldbody,
            "camera",
            attrib={
                "name": camera,
                "pos": pos_str,
                "xyaxes": xy_axes_str,
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
        # Define the path frame body and attributes
        # ET.SubElement(
        #     worldbody,
        #     "body",
        #     {"name": "path_frame", "mocap": "true"},
        # )

    # Create a tree from the root element and write it to a file
    tree = ET.ElementTree(mujoco)
    pretty_write_xml(
        tree.getroot(),
        os.path.join(os.path.dirname(mjcf_path), f"{robot_name}_scene.xml"),
    )


def process_mjcf_file(root: ET.Element, robot: Robot):
    """Processes an MJCF (MuJoCo XML) file by updating and adding various settings and constraints based on the robot's configuration.

    Args:
        root (ET.Element): The root element of the MJCF XML tree.
        robot (Robot): The robot object containing configuration details.
    """
    update_compiler_settings(root)
    add_option_settings(root)

    # if robot.config["general"]["has_imu"]:
    #     add_imu_sensor(root, robot.config["general"])

    update_joint_params(root, robot.config["joints"])
    update_geom_classes(root, ["contype", "conaffinity", "group", "density"])
    add_joint_constraints(root, robot.config["general"], robot.config["joints"])

    if robot.config["general"]["is_neck_closed_loop"]:
        add_neck_constraints(root, robot.config["general"])

    if robot.config["general"]["is_waist_closed_loop"]:
        add_waist_constraints(root, robot.config["general"])

    if robot.config["general"]["is_knee_closed_loop"]:
        add_knee_constraints(root, robot.config["general"])

    if robot.config["general"]["is_ankle_closed_loop"]:
        add_ankle_constraints(root, robot.config["general"])

    if "ee_name" in robot.config["general"]:
        add_ee_sites(root, robot.config["general"]["ee_name"])

    if "foot_name" in robot.config["general"]:
        add_foot_sites(root, robot.config["general"]["foot_name"])

    if "sysID" not in robot.name:
        add_keyframes(root, robot, True)

    exclude_all_contacts(root)


def get_mjcf_files(robot_name: str):
    """Generates and processes MJCF files for a specified robot.

    This function removes any existing cache file for the robot, parses the URDF file, and generates MJCF files with visual and fixed configurations. It processes the MJCF files by adding actuators, default settings, and other necessary elements. The function also handles the creation of scene XML files for both fixed and non-fixed configurations.

    Args:
        robot_name (str): The name of the robot for which to generate MJCF files.

    Raises:
        ValueError: If the source MJCF file 'mjmodel.xml' is not found in the current directory.
    """
    cache_file_path = os.path.join(
        "toddlerbot", "descriptions", robot_name, f"{robot_name}_data.pkl"
    )
    if os.path.exists(cache_file_path):
        os.remove(cache_file_path)

    robot = Robot(robot_name)

    robot_dir = os.path.join("toddlerbot", "descriptions", robot_name)
    urdf_path = os.path.join(robot_dir, robot_name + ".urdf")
    urdf_tree = ET.parse(urdf_path)
    urdf_root = urdf_tree.getroot()
    pretty_write_xml(urdf_root, urdf_path)

    source_mjcf_path = os.path.join("mjmodel.xml")
    mjcf_vis_path = os.path.join(robot_dir, robot_name + "_vis.xml")
    if os.path.exists(source_mjcf_path):
        shutil.move(source_mjcf_path, mjcf_vis_path)
    else:
        raise ValueError(
            "No MJCF file found. Remember to click the button save_xml to save the model to mjmodel.xml in the current directory."
        )

    xml_tree = ET.parse(mjcf_vis_path)
    xml_root = xml_tree.getroot()

    process_mjcf_file(xml_root, robot)
    add_position_actuators_to_mjcf(xml_root, robot.config["joints"])
    add_default_settings(
        xml_root, robot.config["general"], robot.config["joints"], "position"
    )
    pretty_write_xml(xml_root, mjcf_vis_path)
    create_scene_xml(mjcf_vis_path, is_fixed=True)

    mjcf_fixed_path = os.path.join(robot_dir, robot_name + "_fixed.xml")
    add_motor_actuators_to_mjcf(xml_root, robot.config["joints"])
    add_default_settings(
        xml_root, robot.config["general"], robot.config["joints"], "motor"
    )
    pretty_write_xml(xml_root, mjcf_fixed_path)
    create_scene_xml(mjcf_fixed_path, is_fixed=True)

    if not robot.config["general"]["is_fixed"]:
        mjcf_path = os.path.join(robot_dir, robot_name + ".xml")
        add_body_link(xml_root, urdf_path, robot.config["general"]["offsets"])

        add_keyframes(xml_root, robot, False)

        add_contacts(xml_root, robot.collision_config)
        # replace_box_collision(xml_root, robot.config["general"])

        add_motor_actuators_to_mjcf(xml_root, robot.config["joints"])
        add_default_settings(
            xml_root, robot.config["general"], robot.config["joints"], "motor"
        )
        pretty_write_xml(xml_root, mjcf_path)
        create_scene_xml(mjcf_path, is_fixed=False)


def main():
    """Parses command-line arguments to process MJCF files for a specified robot.

    This function sets up an argument parser to accept a robot name, which should match the name in the descriptions. It then calls the `get_mjcf_files` function with the provided robot name to process the corresponding MJCF files.
    """
    parser = argparse.ArgumentParser(description="Process the MJCF.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    args = parser.parse_args()

    get_mjcf_files(args.robot)


if __name__ == "__main__":
    main()
