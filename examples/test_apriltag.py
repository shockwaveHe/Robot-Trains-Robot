import cv2
import matplotlib.pyplot as plt
import numpy as np

from toddlerbot.sensing.camera import AprilTagDetector, Camera

# This script is for visualizing the AprilTag detection in real-time using two cameras.


def visualize(ax, averaged_poses, T_left, T_right, frame_id):
    """
    Real-time visualization of the two cameras and AprilTag poses using Matplotlib.
    """
    ax.clear()

    # Define the camera and tag points
    camera_points = np.array(
        [
            T_left[:3, 3],  # Left camera
            T_right[:3, 3],  # Right camera
        ]
    )
    camera_axes = np.array([T_left[:3, :3], T_right[:3, :3]])

    # Plot the cameras as coordinate systems
    for i, (camera_point, axes) in enumerate(zip(camera_points, camera_axes)):
        ax.scatter(*camera_point, c="blue", label=f"Camera {i + 1}" if i == 0 else "")
        ax.text(*camera_point, f"Camera {i + 1}", color="blue")

        # Plot coordinate axes
        for axis, color in zip(axes.T, ["red", "green", "blue"]):  # x, y, z axes
            ax.quiver(*camera_point, *axis, length=0.05, color=color, normalize=True)

    # Plot the AprilTags as coordinate systems
    for tag_id, T_tag in averaged_poses.items():
        position = T_tag[:3, 3]  # Translation vector
        orientation = T_tag[:3, :3]  # Rotation matrix

        ax.scatter(*position, c="red", label=f"Tag {tag_id}")
        ax.text(*position, f"Tag {tag_id}", color="red")

        # Plot coordinate axes
        for axis, color in zip(orientation.T, ["red", "green", "blue"]):  # x, y, z axes
            ax.quiver(*position, *axis, length=0.05, color=color, normalize=True)

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    # Set limits to be centered and have equal range
    ax.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
    ax.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
    ax.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])

    # Labels and grid
    ax.set_title(f"Real-Time AprilTag Visualization (Frame {frame_id})")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.legend()
    ax.grid(True)


if __name__ == "__main__":
    left_eye = Camera("left")
    right_eye = Camera("right")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1.0, 1.0, 1.0])  # Equal aspect ratio

    frame_id = 0

    april_detector = AprilTagDetector()
    try:
        while True:
            left_tag_poses = left_eye.detect_tags()
            right_tag_poses = right_eye.detect_tags()

            # Average the poses for each tag
            averaged_poses = {}
            for tag_id, poses in left_tag_poses.items():
                avg_pose = np.mean([poses, right_tag_poses[tag_id]], axis=0)
                averaged_poses[tag_id] = avg_pose

            print(averaged_poses)
            # Update the Matplotlib plot
            visualize(
                ax,
                averaged_poses,
                left_eye.eye_transform,
                right_eye.eye_transform,
                frame_id,
            )
            plt.pause(0.001)  # Pause to allow real-time updates

            frame_id += 1

    except KeyboardInterrupt:
        left_eye.close()
        right_eye.close()
        cv2.destroyAllWindows()
        plt.close()
