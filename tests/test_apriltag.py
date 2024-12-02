import os
import pickle
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pupil_apriltags as apriltag

from toddlerbot.sensing.camera import Camera


class AprilTagDetector:
    def __init__(self, families: str = "tag36h11") -> None:
        self.detector = apriltag.Detector(
            families=families, quad_decimate=1.0, decode_sharpening=0.25
        )

    def detect(
        self,
        img: npt.NDArray[np.uint8],
        intrinsics: Dict[str, float] | npt.NDArray[np.float32],
        tag_size: float,
    ) -> List[apriltag.Detection]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if isinstance(intrinsics, dict):
            camera_params = [
                intrinsics["fx"],
                intrinsics["fy"],
                intrinsics["cx"],
                intrinsics["cy"],
            ]
        else:
            camera_params = [
                intrinsics[0, 0],
                intrinsics[1, 1],
                intrinsics[0, 2],
                intrinsics[1, 2],
            ]

        results = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=tag_size,
        )
        return results


def compute_average_pose(left_results, right_results, T_right_to_left):
    """
    Compute the average pose of AprilTags detected in both left and right images,
    and apply a 180-degree rotation to align the normal vector of the tag.
    """
    # 180-degree rotation matrix around the x-axis
    R_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    tag_poses = {}
    for left_detection in left_results:
        tag_id = left_detection.tag_id
        if tag_id not in tag_poses:
            tag_poses[tag_id] = []

        # Pose in left camera
        R_left, t_left = left_detection.pose_R, left_detection.pose_t
        T_left = np.eye(4)
        T_left[:3, :3] = R_left @ R_180  # Apply 180-degree rotation
        T_left[:3, 3] = t_left.flatten()

        tag_poses[tag_id].append(T_left)

    for right_detection in right_results:
        tag_id = right_detection.tag_id
        if tag_id not in tag_poses:
            continue  # Only average tags seen in both views

        # Pose in right camera, transformed to the left camera
        R_right, t_right = right_detection.pose_R, right_detection.pose_t
        T_right = np.eye(4)
        T_right[:3, :3] = R_right @ R_180  # Apply 180-degree rotation
        T_right[:3, 3] = t_right.flatten()

        T_left_to_right = np.eye(4)
        T_left_to_right[:3, 3] = T_right_to_left
        T_right_in_left = T_left_to_right @ T_right

        tag_poses[tag_id].append(T_right_in_left)

    # Average the poses for each tag
    averaged_poses = {}
    for tag_id, poses in tag_poses.items():
        if len(poses) > 1:
            avg_pose = np.mean(poses, axis=0)  # Element-wise mean
            averaged_poses[tag_id] = avg_pose

    return averaged_poses


def visualize(ax, averaged_poses, T_right_to_left, frame_id):
    """
    Real-time visualization of the two cameras and AprilTag poses using Matplotlib.
    """
    ax.clear()

    # Define the camera and tag points
    camera_points = np.array(
        [
            [0, 0, 0],  # Left camera
            T_right_to_left,  # Right camera
        ]
    )
    tag_points = np.array([T_tag[:3, 3] for T_tag in averaged_poses.values()])

    # Set axis limits for clarity
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.1, 0.1)
    ax.set_zlim(-0.1, 0.2)

    # Plot left and right cameras
    ax.scatter(
        camera_points[:, 0],
        camera_points[:, 1],
        camera_points[:, 2],
        c="blue",
        label="Cameras",
    )
    ax.text(
        camera_points[0, 0],
        camera_points[0, 1],
        camera_points[0, 2],
        "Left Camera",
        color="blue",
    )
    ax.text(
        camera_points[1, 0],
        camera_points[1, 1],
        camera_points[1, 2],
        "Right Camera",
        color="blue",
    )

    # Plot AprilTags
    if len(tag_points) > 0:
        ax.scatter(
            tag_points[:, 0],
            tag_points[:, 1],
            tag_points[:, 2],
            c="red",
            label="AprilTags",
        )
        for i, (x, y, z) in enumerate(tag_points):
            ax.text(x, y, z, f"Tag {list(averaged_poses.keys())[i]}", color="red")

    # Labels and grid
    ax.set_title(f"Real-Time AprilTag Visualization (Frame {frame_id})")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.legend()
    ax.grid(True)


if __name__ == "__main__":
    left_eye = Camera(0)
    right_eye = Camera(4)

    calib_params_path = os.path.join("toddlerbot", "sensing", "calibration_params.pkl")
    with open(calib_params_path, "rb") as f:
        calib_params = pickle.load(f)

    K1, D1, K2, D2, R, T = (
        calib_params["K1"],
        calib_params["D1"],
        calib_params["K2"],
        calib_params["D2"],
        calib_params["R"],
        calib_params["T"],
    )

    # Transformation from right camera to left camera
    T_right_to_left = np.array([-0.03, 0, 0])  # Translation along -x axis

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

    frame_id = 0

    try:
        while True:
            left_image = left_eye.get_frame()
            right_image = right_eye.get_frame()

            combined = np.hstack((left_image, right_image))
            cv2.imshow("AprilTag Detection", combined)

            april_detector = AprilTagDetector()
            left_results = april_detector.detect(
                left_image, intrinsics=K1, tag_size=0.03
            )
            right_results = april_detector.detect(
                right_image, intrinsics=K2, tag_size=0.03
            )

            averaged_poses = compute_average_pose(
                left_results, right_results, T_right_to_left
            )
            # Update the Matplotlib plot
            visualize(ax, averaged_poses, T_right_to_left, frame_id)
            plt.pause(0.001)  # Pause to allow real-time updates

            frame_id += 1

    except KeyboardInterrupt:
        left_eye.close()
        right_eye.close()
        cv2.destroyAllWindows()
        plt.close()
