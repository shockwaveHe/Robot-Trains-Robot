import argparse
import os
import shutil
import xml.etree.ElementTree as ET


def process_urdf_and_stl_files(robot_dir):
    robot_name = os.path.basename(robot_dir)

    # Check for URDF file in the base directory
    urdf_file = "robot.urdf"
    urdf_path = os.path.join(robot_dir, urdf_file)
    if not os.path.exists(urdf_path):
        print("No URDF file found in the robot directory.")
        return

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

    # Save changes to the URDF
    tree.write(urdf_path)

    # Create 'meshes' directory if not exists
    meshes_dir = os.path.join(robot_dir, "meshes")
    if not os.path.exists(meshes_dir):
        os.makedirs(meshes_dir)

    # Move or delete STL and PART files based on whether they're referenced
    for subdir, _, files in os.walk(robot_dir):
        for file in files:
            if file.endswith((".stl", ".part")):
                file_path = os.path.join(subdir, file)
                if file in referenced_stls:
                    shutil.move(file_path, os.path.join(meshes_dir, file))
                else:
                    os.remove(file_path)

    # Add XML declaration if missing
    with open(urdf_path, "r+") as urdf_file:
        content = urdf_file.read()
        if not content.startswith('<?xml version="1.0" ?>'):
            content = '<?xml version="1.0" ?>\n' + content
            urdf_file.seek(0)
            urdf_file.write(content)
            urdf_file.truncate()

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
