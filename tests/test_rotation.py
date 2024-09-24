import numpy as np
from scipy.spatial.transform import Rotation as R


def rotate_z_then_y(z_angle_deg: float) -> np.ndarray:
    """
    Performs a rotation around the z-axis by z_angle_deg, followed by a
    -90 degree rotation around the y-axis, and returns the resulting Euler angles.

    Parameters:
    - z_angle_deg: The rotation around the z-axis in degrees.

    Returns:
    - The resulting Euler angles in radians (xyz order).
    """
    # Convert z rotation angle to radians
    theta_z = np.deg2rad(z_angle_deg)

    # Create the first rotation: z-axis rotation
    rotation_z = R.from_euler("z", theta_z)

    # Create the second rotation: -90 degrees around the y-axis
    rotation_y = R.from_euler("y", -90, degrees=True)

    # Combine the two rotations
    combined_rotation = rotation_y * rotation_z  # First z, then y

    # Convert the combined rotation back to Euler angles (xyz order)
    combined_euler = combined_rotation.as_euler("xyz")  # Assuming XYZ rotation order

    return combined_euler


# Example usage with a 45 degree rotation around z-axis
z_angle_deg = 30  # You can change this value
combined_euler = rotate_z_then_y(z_angle_deg)

print(f"Z rotation: {z_angle_deg} degrees")
print(f"Combined Euler angles (in radians): {combined_euler}")
print(f"Combined Euler angles (in degrees): {np.rad2deg(combined_euler)}")
