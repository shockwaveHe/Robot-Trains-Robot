import argparse
import os
import shutil
import xml.etree.ElementTree as ET

import trimesh

from toddlerbot.utils.file_utils import find_description_path


def compute_bounding_box_mesh(urdf_path, mesh_filename):
    robot_dir = os.path.dirname(urdf_path)
    mesh = trimesh.load(os.path.join(robot_dir, mesh_filename))
    # Compute the oriented bounding box
    bounding_box = mesh.bounding_box_oriented
    # Create the full path for the bounding box file
    collision_mesh_file_path = os.path.join(
        robot_dir,
        "collisions",
        os.path.basename(mesh_filename).replace("visual", "collision"),
    )
    # Export the bounding box to a file
    bounding_box.export(collision_mesh_file_path)

    return os.path.relpath(collision_mesh_file_path, urdf_path)


def update_collisons(robot_name):
    urdf_path = find_description_path(robot_name)

    # Ensure the collision directory exists
    collision_dir = os.path.join(os.path.dirname(urdf_path), "collisions")
    if os.path.exists(collision_dir):
        shutil.rmtree(collision_dir)

    os.makedirs(collision_dir, exist_ok=True)

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    link_collision_list = [
        "body_link",
        "neck_link",
        "head_link",
        "hip_roll_link",
        "left_hip_pitch_link",
        "left_calf_link",
        "ank_roll_link",
        "hip_roll_link_2",
        "right_hip_pitch_link",
        "right_calf_link",
        "ank_roll_link_2",
        "sho_pitch_link",
        "sho_roll_link",
        "elb_link",
        "sho_pitch_link_2",
        "sho_roll_link_2",
        "elb_link_2",
    ]

    for link in root.findall("link"):
        link_name = link.get("name")
        # Find the visual element and its mesh filename
        visual = link.find("visual")
        if visual is not None:
            geometry = visual.find("geometry")
            mesh = geometry.find("mesh")
            mesh_filename = mesh.get("filename") if mesh is not None else None

            # Check if the link is in the list for updating collisions
            if link_name in link_collision_list and mesh_filename is not None:
                # Compute the bounding box and replace the collision mesh
                collision_bbox_file = compute_bounding_box_mesh(
                    urdf_path, mesh_filename
                )
                collision = link.find("collision")
                if collision is None:
                    # If no collision tag exists, create one
                    collision = ET.SubElement(link, "collision")
                    geometry = ET.SubElement(collision, "geometry")
                    mesh = ET.SubElement(geometry, "mesh")
                else:
                    geometry = collision.find("geometry")
                    mesh = geometry.find("mesh")

                mesh.set("filename", collision_bbox_file)
            else:
                # Remove the collision element if it exists
                collision = link.find("collision")
                if collision is not None:
                    link.remove(collision)

    # Save the modified URDF
    tree.write(
        os.path.join(
            os.path.dirname(urdf_path), "updated_" + os.path.basename(urdf_path)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update the collisions.")
    parser.add_argument(
        "--robot-name",
        type=str,
        default="base",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    args = parser.parse_args()

    update_collisons(args.robot_name)
