import argparse
import os
import xml.etree.ElementTree as ET

from toddlerbot.sim.robot import Robot


def merge_mjcf(robot_name, arm_name, target_body_parent_name="panda_joint1"):
    # Paths to MJCF files
    arm_dir = os.path.join("toddlerbot", "robot_descriptions", arm_name)
    arm_mjcf_path = os.path.join(arm_dir, "panda_nohand.xml")

    robot_dir = os.path.join("toddlerbot", "robot_descriptions", robot_name)
    robot_mjcf_path = os.path.join(robot_dir, f"{robot_name}_fixed.xml")

    # Parse both MJCF files
    tree1 = ET.parse(robot_mjcf_path)
    root1 = tree1.getroot()

    tree2 = ET.parse(arm_mjcf_path)
    root2 = tree2.getroot()

    # Elements that need to be merged (e.g., asset, default, actuator, etc.)
    merge_tags = ["asset", "default", "actuator", "compiler", "option"]

    # Dictionary to store merged elements per tag
    merged_elements = {tag: [] for tag in merge_tags}

    # Function to merge elements based on tag and attributes
    def merge_elements(root, tag_list):
        for tag in tag_list:
            for elem in root.findall(tag):
                merged_elements[tag].append(elem)

    # Merge elements from both files
    merge_elements(root1, merge_tags)
    merge_elements(root2, merge_tags)

    # Create a new MJCF tree for merged output
    merged_root = ET.Element("mujoco", attrib={"model": f"{robot_name}_{arm_name}"})

    # Append merged elements under their respective tags
    for tag in merge_tags:
        if merged_elements[tag]:
            # Create a new tag element and append all items under it
            parent = ET.Element(tag)
            for elem in merged_elements[tag]:
                for child in elem:  # Append only the children of `elem`
                    parent.append(child)
            merged_root.append(parent)

    # Append the <worldbody> from root1 to the new tree (without merging)
    worldbody1 = root1.find("worldbody")
    if worldbody1 is not None:
        merged_root.append(worldbody1)

    # Append the <worldbody> from root2 to the new tree (without merging)
    worldbody2 = root2.find("worldbody")
    if worldbody2 is not None:
        merged_root.append(worldbody2)

    # Write the merged MJCF to a new file
    merged_tree = ET.ElementTree(merged_root)
    merged_tree.write(f"{robot_name}_{arm_name}.xml")


def main():
    parser = argparse.ArgumentParser(description="Process the MJCF.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="franka_arm",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    args = parser.parse_args()

    # Call the function with the two XML files
    merge_mjcf(args.robot, args.arm)


if __name__ == "__main__":
    main()
