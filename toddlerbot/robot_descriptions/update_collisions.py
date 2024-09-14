import argparse
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def compute_bounding_box(mesh: trimesh.Trimesh):
    # Compute the minimum and maximum bounds along each axis
    bounds_min = mesh.bounds[0]
    bounds_max = mesh.bounds[1]

    # Calculate the size (width, height, depth) of the bounding box
    size = bounds_max - bounds_min

    # The center of the bounding box
    center = (bounds_max + bounds_min) / 2

    return size, center


def compute_bounding_sphere(mesh: trimesh.Trimesh):
    # Compute the centroid of the mesh
    centroid = mesh.centroid

    # Compute the radius as the maximum distance from the centroid to any vertex
    distances = np.linalg.norm(mesh.vertices - centroid, axis=1)
    radius = np.max(distances)

    return radius, centroid


def compute_bounding_cylinder(mesh: trimesh.Trimesh):
    def bounding_cylinder_along_axis(axis: int):
        # Project the mesh vertices onto the plane perpendicular to the axis
        axes = [0, 1, 2]
        axes.remove(axis)
        projection = mesh.vertices[:, axes]

        # Compute the centroid in the plane of projection (XY, XZ, or YZ plane)
        centroid_in_plane = np.mean(projection, axis=0)

        # Compute the radius as the maximum distance from the centroid to any vertex in this plane
        distances = projection - centroid_in_plane
        radius = np.max(np.linalg.norm(distances, axis=1))

        # Compute the height as the range along the remaining axis
        height = mesh.bounds[:, axis].ptp()  # ptp() computes the range (max - min)

        # Compute the full centroid of the cylinder in 3D space
        centroid = np.zeros(3)
        centroid[axes] = centroid_in_plane
        centroid[axis] = np.mean(mesh.bounds[:, axis])

        # Determine the RPY angles based on the axis
        if axis == 0:  # X-axis
            rpy = (0.0, np.pi / 2, 0.0)  # 90 degrees rotation around Y-axis
        elif axis == 1:  # Y-axis
            rpy = (np.pi / 2, 0.0, 0.0)  # 90 degrees rotation around X-axis
        else:  # Z-axis (default)
            rpy = (0.0, 0.0, 0.0)  # No rotation needed

        # Return the radius, height, centroid, and volume of the cylinder
        return radius, height, centroid, rpy, np.pi * radius**2 * height

    # Calculate bounding cylinders for each principal axis
    cylinders = [
        bounding_cylinder_along_axis(0),  # X-axis
        bounding_cylinder_along_axis(1),  # Y-axis
        bounding_cylinder_along_axis(2),  # Z-axis
    ]

    # Select the cylinder with the smallest volume
    best_cylinder = min(cylinders, key=lambda c: c[-1])

    # Return the radius, height, and centroid of the smallest cylinder
    return best_cylinder[0], best_cylinder[1], best_cylinder[2], best_cylinder[3]


def update_collisons(robot_name: str):
    robot_dir = os.path.join("toddlerbot", "robot_descriptions", robot_name)
    collision_config_file_path = os.path.join(robot_dir, "config_collision.json")
    urdf_path = os.path.join(robot_dir, f"{robot_name}.urdf")

    # Ensure the collision directory exists
    # collision_dir = os.path.join(os.path.dirname(urdf_path), "collisions")
    # if os.path.exists(collision_dir):
    #     shutil.rmtree(collision_dir)

    # os.makedirs(collision_dir, exist_ok=True)

    with open(collision_config_file_path, "r") as f:
        collision_config = json.load(f)

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for link in root.findall("link"):
        link_name = link.get("name")
        if link_name is None or link_name not in collision_config:
            continue

        if collision_config[link_name]["has_collision"]:
            # Find the visual element and its mesh filename
            visual = link.find("visual")
            geometry = visual.find("geometry") if visual is not None else None
            mesh = geometry.find("mesh") if geometry is not None else None
            mesh_filename = mesh.get("filename") if mesh is not None else None

            if mesh_filename is not None:
                # Load the mesh and compute the bounding cylinder
                mesh = trimesh.load(
                    os.path.join(os.path.dirname(urdf_path), mesh_filename)
                )
                # Set or create the collision element
                collision = link.find("collision")
                if collision is None:
                    collision = ET.SubElement(
                        link, "collision", {"name": f"{link_name}_collision"}
                    )
                else:
                    collision.set("name", f"{link_name}_collision")

                geometry = collision.find("geometry")
                if geometry is not None:
                    # Remove the existing geometry
                    collision.remove(geometry)

                geometry = ET.SubElement(collision, "geometry")

                if collision_config[link_name]["type"] == "box":
                    size, center = compute_bounding_box(mesh)
                    rpy = [0, 0, 0]
                    size[0] *= collision_config[link_name]["scale"][0]
                    size[1] *= collision_config[link_name]["scale"][1]
                    size[2] *= collision_config[link_name]["scale"][2]
                    ET.SubElement(
                        geometry,
                        "box",
                        {"size": f"{size[0]} {size[1]} {size[2]}"},
                    )
                elif collision_config[link_name]["type"] == "sphere":
                    radius, center = compute_bounding_sphere(mesh)
                    radius *= collision_config[link_name]["scale"][0]
                    rpy = [0, 0, 0]
                    ET.SubElement(geometry, "sphere", {"radius": str(radius)})
                else:
                    radius, height, center, rpy = compute_bounding_cylinder(mesh)
                    radius *= collision_config[link_name]["scale"][0]
                    height *= collision_config[link_name]["scale"][1]
                    ET.SubElement(
                        geometry,
                        "cylinder",
                        {"radius": str(radius), "length": str(height)},
                    )

                xyz_str = f"{center[0]} {center[1]} {center[2]}"
                rpy_str = f"{rpy[0]} {rpy[1]} {rpy[2]}"
                origin = collision.find("origin")
                if origin is not None:
                    # Remove the existing geometry
                    origin.set("xyz", xyz_str)
                    origin.set("rpy", rpy_str)
                else:
                    ET.SubElement(collision, "origin", {"xyz": xyz_str, "rpy": rpy_str})

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
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in robot_descriptions.",
    )
    args = parser.parse_args()

    update_collisons(args.robot)
