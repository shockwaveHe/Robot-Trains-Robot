import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from typing import Dict, Tuple

import trimesh  # type: ignore

from toddlerbot.utils.file_utils import find_robot_file_path

default_scale_factor = (1.0, 1.0, 1.0)
hip_scale_factor = (0.5, 1.0, 1.0)
leg_scale_factor = (1.0, 0.5, 1.0)
arm_scale_factor = (1.0, 0.5, 1.0)

collision_link_dict: Dict[str, Tuple[float, float, float]] = {
    "body_link": default_scale_factor,
    "neck_link": default_scale_factor,
    "head_link": default_scale_factor,
    "hip_roll_link": hip_scale_factor,
    "left_hip_pitch_link": leg_scale_factor,
    "left_calf_link": leg_scale_factor,
    # "ank_roll_link": default_scale_factor, # We need accurate contact for the feet
    "hip_roll_link_2": hip_scale_factor,
    "right_hip_pitch_link": leg_scale_factor,
    "right_calf_link": leg_scale_factor,
    # "ank_roll_link_2": default_scale_factor,
    "sho_roll_link": arm_scale_factor,
    "elb_link": arm_scale_factor,
    "sho_roll_link_2": arm_scale_factor,
    "elb_link_2": arm_scale_factor,
}


def compute_bounding_box_mesh(
    robot_dir: str, mesh_filename: str, scale_factor: Tuple[float, float, float]
):
    mesh: trimesh.Trimesh = trimesh.load(os.path.join(robot_dir, mesh_filename))  # type: ignore

    # Compute the centroid of the bounding box
    centroid = mesh.bounding_box.centroid

    bounding_mesh = trimesh.Trimesh(
        vertices=mesh.bounding_box.vertices, faces=mesh.bounding_box.faces
    )
    # Translate the mesh to the origin
    bounding_mesh.apply_translation(-centroid)  # type: ignore
    # Apply non-uniform scaling
    bounding_mesh.apply_scale(scale_factor)  # type: ignore
    # Translate the mesh back to its original centroid
    bounding_mesh.apply_translation(centroid)  # type: ignore

    # Create the full path for the bounding box file
    collision_mesh_file_path = os.path.join(
        robot_dir,
        "collisions",
        os.path.basename(mesh_filename).replace("visual", "collision"),
    )
    # Export the bounding box to a file
    bounding_mesh.export(collision_mesh_file_path)  # type: ignore

    return os.path.relpath(collision_mesh_file_path, robot_dir)


def update_collisons(robot_name: str):
    urdf_path = find_robot_file_path(robot_name)

    # Ensure the collision directory exists
    collision_dir = os.path.join(os.path.dirname(urdf_path), "collisions")
    if os.path.exists(collision_dir):
        shutil.rmtree(collision_dir)

    os.makedirs(collision_dir, exist_ok=True)

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for link in root.findall("link"):
        link_name = link.get("name")
        if link_name is None:
            continue

        # Find the visual element and its mesh filename
        visual = link.find("visual")
        if visual is not None:
            geometry = visual.find("geometry")
            mesh = geometry.find("mesh") if geometry is not None else None
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
                    mesh = geometry.find("mesh") if geometry is not None else None

                if mesh is not None:
                    mesh.set("filename", collision_bbox_file)

            elif "ank_roll_link" not in link_name:
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
