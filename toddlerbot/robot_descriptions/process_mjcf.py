import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple

from transforms3d.euler import euler2quat

from toddlerbot.sim.robot import Robot
from toddlerbot.utils.math_utils import round_to_sig_digits


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


def add_imu_sensor(root: ET.Element, general_config: Dict[str, Any]):
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


def add_keyframes(
    root: ET.Element,
    default_ctrl: List[float],
    is_fixed: bool,
    has_lower_body: bool,
    has_upper_body: bool,
    has_gripper: bool,
):
    # Create or find the <default> element
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)

    keyframe = ET.SubElement(root, "keyframe")

    if is_fixed:
        qpos_str = ""
    else:
        qpos_str = "0 0 0.336 1 0 0 0 "

    ctrl_str = " ".join(map(str, default_ctrl))

    if has_upper_body and has_lower_body:  # neck
        qpos_str += "0 0 0 0 "

    if has_lower_body:  # waist and legs
        qpos_str += (
            "0 0 0 0 "
            + "0 0 0 -0.267268 0.523599 -0.523599 -0.523599 -0.25637 0 0 0.248043 0 -0.246445 -0.253132 0.256023 0.523599 -0.523599 -0.523599 "
            + "0 0 0 0.267268 -0.523599 0.523599 0.523599 -0.25637 0 0 -0.248043 0 0.246445 0.253132 -0.256023 -0.523599 0.523599 0.523599 "
        )

    if has_upper_body:  # arms
        qpos_str += (
            "0.174533 -0.261799 1.0472 -1.0472 0.523599 -1.0472 1.0472 1.309 -1.309 0 "
        )
        if has_gripper:
            qpos_str += "0 0 0 "

        qpos_str += (
            "-0.174533 -0.261799 -1.0472 1.0472 0.523599 1.0472 -1.0472 -1.309 1.309 0"
        )
        if has_gripper:
            qpos_str += " 0 0 0"

    ET.SubElement(keyframe, "key", {"name": "home", "qpos": qpos_str, "ctrl": ctrl_str})


def add_default_settings(root: ET.Element, general_config: Dict[str, Any]):
    # Create or find the <default> element
    default = root.find("default")
    if default is not None:
        root.remove(default)

    default = ET.SubElement(root, "default")

    ET.SubElement(
        default,
        "geom",
        {
            "type": "mesh",
            "solref": f"{general_config['solref'][0]} {general_config['solref'][1]}",
        },
    )

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


def add_contacts(
    root: ET.Element, collision_config: Dict[str, Dict[str, Any]], foot_name: str
):
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
        if (
            "floor" in collision_config[body_name]["contact_pairs"]
            and foot_name in body_name
        ):
            pairs.append((body_name, "floor"))

    # for i in range(len(collision_bodies) - 1):
    #     for j in range(i + 1, len(collision_body_names)):
    #         body1_name = collision_body_names[i]
    #         body2_name = collision_body_names[j]

    #         paired_1 = body2_name in collision_config[body1_name]["contact_pairs"]
    #         paired_2 = body1_name in collision_config[body2_name]["contact_pairs"]
    #         if (paired_1 and paired_2) and (
    #             foot_name in body1_name or foot_name in body2_name
    #         ):
    #             pairs.append((body1_name, body2_name))
    #         else:
    #             excludes.append((body1_name, body2_name))

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


def add_waist_constraints(root: ET.Element, general_config: Dict[str, Any]):
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
            solref=f"{general_config['solref'][0]} {general_config['solref'][1]}",
        )


def add_ankle_constraints(root: ET.Element, general_config: Dict[str, Any]):
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
            joint_pinion_1_name = joint_name.replace("_rack", "_pinion_1")
            joint_pinion_2_name = joint_name.replace("_rack", "_pinion_2")
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


def add_actuators_to_mjcf(root: ET.Element, joints_config: Dict[str, Any]):
    # Create <actuator> element if it doesn't exist
    actuator = root.find("./actuator")
    if actuator is not None:
        root.remove(actuator)

    actuator = ET.SubElement(root, "actuator")

    for joint_name, joint_config in joints_config.items():
        if "spec" not in joint_config:
            continue

        # transmission = joint_config["transmission"]
        # if transmission == "gear":
        #     joint_driven_name = joint_name.replace("_drive", "_driven")
        #     joint_driven: ET.Element | None = root.find(
        #         f".//joint[@name='{joint_driven_name}']"
        #     )
        #     if joint_driven is None:
        #         raise ValueError(f"The driven joint {joint_driven_name} is not found")

        #     position = ET.SubElement(
        #         actuator,
        #         "position",
        #         name=joint_name,
        #         joint=joint_driven_name,
        #         kp=str(joints_config[joint_name]["kp_sim"]),
        #         gear=str(
        #             round_to_sig_digits(1 / joints_config[joint_name]["gear_ratio"], 6)
        #         ),
        #         ctrlrange=joint_driven.get("range", "-3.141592 3.141592"),
        #     )
        # else:
        joint: ET.Element | None = root.find(f".//joint[@name='{joint_name}']")
        if joint is None:
            raise ValueError(f"The joint {joint_name} is not found")

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


def replace_box_collision(root: ET.Element, foot_name: str):
    # Search for the target geom using the substring condition
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

        # Positions for the four corner balls
        ball_positions = [
            [pos[0] - x_offset, pos[1] + y_offset, pos[2] - z_offset],  # Bottom-left
            [pos[0] + x_offset, pos[1] + y_offset, pos[2] - z_offset],  # Bottom-right
            [pos[0] - x_offset, pos[1] + y_offset, pos[2] + z_offset],  # Top-left
            [pos[0] + x_offset, pos[1] + y_offset, pos[2] + z_offset],  # Top-right
        ]

        # Create the new sphere elements at each corner
        for i, ball_pos in enumerate(ball_positions):
            ball_pos = [round_to_sig_digits(x, 6) for x in ball_pos]
            sphere = ET.Element(
                "geom",
                {
                    "name": f"{name}_ball_{i+1}",
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


def create_scene_xml(mjcf_path: str, general_config: Dict[str, Any], is_fixed: bool):
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
                # "solref": f"{general_config['solref'][0]} {general_config['solref'][1]}",
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

    # if robot.config["general"]["has_imu"]:
    #     add_imu_sensor(root, robot.config["general"])

    update_joint_params(root, robot.config["joints"])
    update_geom_classes(root, ["contype", "conaffinity", "group", "density"])
    add_actuators_to_mjcf(root, robot.config["joints"])
    add_joint_constraints(root, robot.config["general"], robot.config["joints"])

    if robot.config["general"]["is_waist_closed_loop"]:
        add_waist_constraints(root, robot.config["general"])

    if robot.config["general"]["is_knee_closed_loop"]:
        add_knee_constraints(root, robot.config["general"])

    if robot.config["general"]["is_ankle_closed_loop"]:
        add_ankle_constraints(root, robot.config["general"])

    if "sysID" not in robot.name:
        has_gripper = False
        for motor_name in robot.motor_ordering:
            if "gripper" in motor_name:
                has_gripper = True

        default_ctrl = robot.get_joint_attrs("is_passive", False, "default_pos")
        add_keyframes(
            root,
            default_ctrl,
            True,
            "arms" not in robot.name,
            "legs" not in robot.name,
            has_gripper,
        )

    add_default_settings(root, robot.config["general"])

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
        create_scene_xml(mjcf_fixed_path, robot.config["general"], is_fixed=True)

        mjcf_path = os.path.join(robot_dir, robot_name + ".xml")
        add_body_link(xml_root, urdf_path, robot.config["general"]["offsets"])

        has_gripper = False
        for motor_name in robot.motor_ordering:
            if "gripper" in motor_name:
                has_gripper = True

        default_ctrl = robot.get_joint_attrs("is_passive", False, "default_pos")
        add_keyframes(
            xml_root,
            default_ctrl,
            False,
            "arms" not in robot.name,
            "legs" not in robot.name,
            has_gripper,
        )
        add_contacts(
            xml_root, robot.collision_config, robot.config["general"]["foot_name"]
        )
        replace_box_collision(xml_root, robot.config["general"]["foot_name"])
        xml_tree.write(mjcf_path)

    create_scene_xml(
        mjcf_path, robot.config["general"], robot.config["general"]["is_fixed"]
    )


def main():
    parser = argparse.ArgumentParser(description="Process the MJCF.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    args = parser.parse_args()

    get_mjcf_files(args.robot)


if __name__ == "__main__":
    main()
