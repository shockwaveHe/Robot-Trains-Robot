import argparse
import os
import shutil
import xml.etree.ElementTree as ET

import trimesh

from toddlerbot.utils.file_utils import find_description_path

collision_link_list = [
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
    "sho_roll_link",
    "elb_link",
    "sho_roll_link_2",
    "elb_link_2",
]
collision_link_dict = {key: (1.0, 1.0, 1.0) for key in collision_link_list}


def compute_bounding_box_mesh(robot_dir, mesh_filename, scale_factors):
    mesh = trimesh.load(os.path.join(robot_dir, mesh_filename))
    # Compute the oriented bounding box
    bounding_box = mesh.bounding_box_oriented
    # Compute the centroid of the bounding box
    centroid = bounding_box.centroid

    bbox_mesh = trimesh.Trimesh(
        vertices=bounding_box.vertices, faces=bounding_box.faces
    )
    # Translate the mesh to the origin
    bbox_mesh.apply_translation(-centroid)
    # Apply non-uniform scaling
    bbox_mesh.apply_scale(scale_factors)
    # Translate the mesh back to its original centroid
    bbox_mesh.apply_translation(centroid)

    # Create the full path for the bounding box file
    collision_mesh_file_path = os.path.join(
        robot_dir,
        "collisions",
        os.path.basename(mesh_filename).replace("visual", "collision"),
    )
    # Export the bounding box to a file
    bbox_mesh.export(collision_mesh_file_path)

    return os.path.relpath(collision_mesh_file_path, robot_dir)


def update_collisons(robot_name):
    urdf_path = find_description_path(robot_name)

    # Ensure the collision directory exists
    collision_dir = os.path.join(os.path.dirname(urdf_path), "collisions")
    if os.path.exists(collision_dir):
        shutil.rmtree(collision_dir)

    os.makedirs(collision_dir, exist_ok=True)

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for link in root.findall("link"):
        link_name = link.get("name")
        # Find the visual element and its mesh filename
        visual = link.find("visual")
        if visual is not None:
            geometry = visual.find("geometry")
            mesh = geometry.find("mesh")
            mesh_filename = mesh.get("filename") if mesh is not None else None

            # Check if the link is in the list for updating collisions
            if link_name in collision_link_dict and mesh_filename is not None:
                # Compute the bounding box and replace the collision mesh
                collision_bbox_file = compute_bounding_box_mesh(
                    os.path.dirname(urdf_path),
                    mesh_filename,
                    collision_link_dict[link_name],
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
    tree.write(urdf_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update the collisions.")
    parser.add_argument(
        "--robot-name",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    args = parser.parse_args()

    update_collisons(args.robot_name)
