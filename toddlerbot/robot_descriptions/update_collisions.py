import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from typing import List

import trimesh  # type: ignore


def compute_bounding_box_mesh(robot_dir: str, mesh_filename: str, scale: List[float]):
    mesh: trimesh.Trimesh = trimesh.load(os.path.join(robot_dir, mesh_filename))  # type: ignore

    # Compute the centroid of the bounding box
    centroid = mesh.bounding_box.centroid

    bounding_mesh = trimesh.Trimesh(
        vertices=mesh.bounding_box.vertices, faces=mesh.bounding_box.faces
    )
    # Translate the mesh to the origin
    bounding_mesh.apply_translation(-centroid)  # type: ignore
    # Apply non-uniform scaling
    bounding_mesh.apply_scale(scale)  # type: ignore
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
    robot_dir = os.path.join("toddlerbot", "robot_descriptions", robot_name)
    collision_config_file_path = os.path.join(robot_dir, "config_collision.json")
    urdf_path = os.path.join(robot_dir, f"{robot_name}.urdf")

    # Ensure the collision directory exists
    collision_dir = os.path.join(os.path.dirname(urdf_path), "collisions")
    if os.path.exists(collision_dir):
        shutil.rmtree(collision_dir)

    os.makedirs(collision_dir, exist_ok=True)

    with open(collision_config_file_path, "r") as f:
        collision_config = json.load(f)

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for link in root.findall("link"):
        link_name = link.get("name")
        if link_name is None or link_name not in collision_config:
            continue

        if collision_config[link_name]["has_collision"]:
            if collision_config[link_name]["type"] == "box":
                # Find the visual element and its mesh filename
                visual = link.find("visual")
                geometry = visual.find("geometry") if visual is not None else None
                mesh = geometry.find("mesh") if geometry is not None else None
                mesh_filename = mesh.get("filename") if mesh is not None else None

                if mesh_filename is not None:
                    # Compute the bounding box and replace the collision mesh
                    collision_filename = compute_bounding_box_mesh(
                        os.path.dirname(urdf_path),
                        mesh_filename,
                        collision_config[link_name]["scale"],
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
                        mesh.set("filename", collision_filename)
        else:
            # Remove the collision element if it exists
            collision = link.find("collision")
            if collision is not None:
                link.remove(collision)

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
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    args = parser.parse_args()

    update_collisons(args.robot)
