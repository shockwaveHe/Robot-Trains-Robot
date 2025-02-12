import argparse
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

# This script updates the collision elements in a robot's URDF file based on a configuration file.


def compute_bounding_box(mesh: trimesh.Trimesh):
    """Computes the size and center of the bounding box for a given 3D mesh.

    Args:
        mesh (trimesh.Trimesh): The 3D mesh for which to compute the bounding box.

    Returns:
        tuple: A tuple containing the size (width, height, depth) of the bounding box and the center point of the bounding box.
    """
    # Compute the minimum and maximum bounds along each axis
    bounds_min = mesh.bounds[0]
    bounds_max = mesh.bounds[1]

    # Calculate the size (width, height, depth) of the bounding box
    size = bounds_max - bounds_min

    # The center of the bounding box
    center = (bounds_max + bounds_min) / 2

    return size, center


def compute_bounding_sphere(mesh: trimesh.Trimesh):
    """Computes the bounding sphere of a given 3D mesh.

    The bounding sphere is defined by its centroid and radius, where the radius is the maximum distance from the centroid to any vertex of the mesh.

    Args:
        mesh (trimesh.Trimesh): The 3D mesh for which to compute the bounding sphere.

    Returns:
        tuple: A tuple containing the radius (float) and the centroid (numpy.ndarray) of the bounding sphere.
    """
    # Compute the centroid of the mesh
    centroid = mesh.centroid

    # Compute the radius as the maximum distance from the centroid to any vertex
    distances = np.linalg.norm(mesh.vertices - centroid, axis=1)
    radius = np.max(distances)

    return radius, centroid


def compute_bounding_cylinder(mesh: trimesh.Trimesh):
    """Compute the bounding cylinder of a 3D mesh with the smallest volume.

    This function calculates the bounding cylinders of a given 3D mesh along each of the principal axes (X, Y, Z) and selects the one with the smallest volume. The bounding cylinder is defined by its radius, height, centroid, and orientation in terms of roll-pitch-yaw (RPY) angles.

    Args:
        mesh (trimesh.Trimesh): The 3D mesh for which the bounding cylinder is to be computed.

    Returns:
        tuple: A tuple containing the radius, height, centroid (as a 3D coordinate), and RPY angles of the bounding cylinder with the smallest volume.
    """

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


def compute_bounding_capsule(mesh: trimesh.Trimesh):
    """Compute the smallest bounding capsule for a given 3D mesh along its principal axes.

    This function calculates bounding capsules along the X, Y, and Z axes of the mesh and selects the one with the smallest volume. A bounding capsule is defined by its radius, height, centroid, and orientation in terms of roll, pitch, and yaw (RPY) angles.

    Args:
        mesh (trimesh.Trimesh): A 3D mesh object for which the bounding capsule is to be computed.

    Returns:
        tuple: A tuple containing the radius, height, centroid (as a 3D coordinate), and RPY orientation of the smallest bounding capsule.
    """
    hull = ConvexHull(mesh.vertices)
    hull_vertices = mesh.vertices[hull.vertices]

    def bounding_capsule_along_axis(axis: int):
        # Project the hull vertices onto the plane perpendicular to the axis
        axes = [0, 1, 2]
        axes.remove(axis)
        projection = hull_vertices[:, axes]

        # Compute the centroid in the plane of projection
        centroid_in_plane = np.mean(projection, axis=0)

        # Compute the radius as the maximum distance from the centroid to any vertex in this plane
        distances = projection - centroid_in_plane
        radius = np.max(np.linalg.norm(distances, axis=1))

        # Compute the height of the cylindrical part (subtract the hemispheres)
        total_height = hull_vertices[:, axis].ptp()  # ptp() computes range (max - min)
        height = max(0, total_height - 2 * radius)  # Ensure non-negative height

        # Compute the full centroid of the capsule in 3D space
        centroid = np.zeros(3)
        centroid[axes] = centroid_in_plane
        centroid[axis] = (mesh.bounds[0, axis] + mesh.bounds[1, axis]) / 2

        # Determine the RPY angles based on the axis
        if axis == 0:  # X-axis
            rpy = (0.0, np.pi / 2, 0.0)  # 90 degrees rotation around Y-axis
        elif axis == 1:  # Y-axis
            rpy = (np.pi / 2, 0.0, 0.0)  # 90 degrees rotation around X-axis
        else:  # Z-axis (default)
            rpy = (0.0, 0.0, 0.0)  # No rotation needed

        # Volume of the capsule = volume of cylinder + 2 * volume of hemispheres
        cylinder_volume = np.pi * radius**2 * height
        hemisphere_volume = 2 / 3 * np.pi * radius**3
        total_volume = cylinder_volume + 2 * hemisphere_volume

        return radius, height, centroid, rpy, total_volume

    # Calculate bounding capsules for each principal axis
    capsules = [
        bounding_capsule_along_axis(0),  # X-axis
        bounding_capsule_along_axis(1),  # Y-axis
        bounding_capsule_along_axis(2),  # Z-axis
    ]

    # Select the capsule with the smallest volume
    best_capsule = min(capsules, key=lambda c: c[-1])

    # Return the radius, height, centroid, and orientation of the smallest capsule
    return best_capsule[0], best_capsule[1], best_capsule[2], best_capsule[3]


def update_collisons(robot_name: str):
    """Updates the collision elements in a robot's URDF file based on a configuration file.

    This function reads a collision configuration from a JSON file and updates the
    collision elements in the specified robot's URDF file. It computes bounding
    geometries for each link that requires a collision element and modifies the URDF
    accordingly.

    Args:
        robot_name (str): The name of the robot whose collision elements are to be updated.
    """
    robot_dir = os.path.join("toddlerbot", "descriptions", robot_name)
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
                    size, center = compute_bounding_box(mesh)  # type: ignore
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
                    radius, center = compute_bounding_sphere(mesh)  # type: ignore
                    radius *= collision_config[link_name]["scale"][0]
                    rpy = [0, 0, 0]
                    ET.SubElement(geometry, "sphere", {"radius": str(radius)})
                elif collision_config[link_name]["type"] == "cylinder":
                    radius, height, center, rpy = compute_bounding_cylinder(mesh)  # type: ignore
                    radius *= collision_config[link_name]["scale"][0]
                    height *= collision_config[link_name]["scale"][1]
                    ET.SubElement(
                        geometry,
                        "cylinder",
                        {"radius": str(radius), "length": str(height)},
                    )
                elif collision_config[link_name]["type"] == "capsule":
                    radius, height, center, rpy = compute_bounding_capsule(mesh)  # type: ignore
                    radius *= collision_config[link_name]["scale"][0]
                    height *= collision_config[link_name]["scale"][1]
                    ET.SubElement(
                        geometry,
                        "capsule",
                        {"radius": str(radius), "length": str(height)},
                    )
                else:
                    ET.SubElement(geometry, "mesh", {"filename": mesh_filename})
                    continue

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
        help="The name of the robot. Need to match the name in descriptions.",
    )
    args = parser.parse_args()

    update_collisons(args.robot)
