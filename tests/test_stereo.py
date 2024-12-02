import argparse
import glob
import os
import pickle
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from toddlerbot.sensing.camera import Camera

# --------------------------
# Utility Functions
# --------------------------


def collect_images(camera_left_id, camera_right_id, save_path, num_images):
    """
    Collect calibration images from stereo cameras.
    """
    cam_left = Camera(camera_left_id)
    cam_right = Camera(camera_right_id)

    print("Press SPACE to capture an image pair. Press ESC to exit.")
    count = 0

    while count < num_images:
        frame_left = cam_left.get_frame()
        frame_right = cam_right.get_frame()

        combined = np.hstack((frame_left, frame_right))
        cv2.imshow("Stereo Cameras", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break
        elif key == 32:  # SPACE to capture
            cv2.imwrite(f"{save_path}/left_{count:02d}.png", frame_left)
            cv2.imwrite(f"{save_path}/right_{count:02d}.png", frame_right)
            print(f"Saved image pair {count + 1}")
            count += 1

    cam_left.close()
    cam_right.close()


def collect_test_images(camera_left_id, camera_right_id, save_path):
    """
    Collect calibration images from stereo cameras.
    """
    cam_left = Camera(camera_left_id)
    cam_right = Camera(camera_right_id)

    print("Press SPACE to capture an image pair. Press ESC to exit.")
    count = 0

    while True:
        frame_left = cam_left.get_frame()
        frame_right = cam_right.get_frame()

        combined = np.hstack((frame_left, frame_right))
        cv2.imshow("Stereo Cameras", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break
        elif key == 32:  # SPACE to capture
            cv2.imwrite(f"{save_path}/left_test_{count:02d}.png", frame_left)
            cv2.imwrite(f"{save_path}/right_test_{count:02d}.png", frame_right)
            print(f"Saved image pair {count + 1}")
            count += 1

    cam_left.close()
    cam_right.close()


def fisheye_calibration(image_paths, checkerboard_size):
    """
    Calibrate a fisheye camera and visualize chessboard corners.
    """
    termination_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    objp = np.zeros((1, checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[
        0 : checkerboard_size[0], 0 : checkerboard_size[1]
    ].T.reshape(-1, 2)
    objp *= checkerboard_size[2]

    objpoints = []  # 3D points
    imgpoints = []  # 2D points

    for image_path in tqdm(image_paths, desc="Calibrating"):
        img = cv2.imread(image_path)
        # cv2.imshow("Image", img)
        # cv2.waitKey(200)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Gray", gray)
        # cv2.waitKey(200)

        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size[:2], None)

        if ret:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), termination_criteria)
            imgpoints.append(corners)

            # Debug: Show chessboard corners
            img_with_corners = cv2.drawChessboardCorners(
                img, checkerboard_size[:2], corners, ret
            )
            cv2.imshow("Chessboard Corners", img_with_corners)
            cv2.waitKey(200)

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs, tvecs = [], []
    ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
    )

    cv2.destroyAllWindows()
    return K, D


def stereo_calibration(
    image_paths_left, image_paths_right, checkerboard_size, K1, D1, K2, D2
):
    """
    Calibrate stereo cameras and compute rectification transforms.
    """
    # Create object points for the checkerboard
    objp = np.zeros((1, checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[
        0 : checkerboard_size[0], 0 : checkerboard_size[1]
    ].T.reshape(-1, 2)
    objp *= checkerboard_size[2]

    objpoints, imgpoints_left, imgpoints_right = [], [], []

    for img_left, img_right in zip(image_paths_left, image_paths_right):
        gray_left = cv2.cvtColor(cv2.imread(img_left), cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(cv2.imread(img_right), cv2.COLOR_BGR2GRAY)

        ret_left, corners_left = cv2.findChessboardCorners(
            gray_left, checkerboard_size[:2], None
        )
        ret_right, corners_right = cv2.findChessboardCorners(
            gray_right, checkerboard_size[:2], None
        )

        if ret_left and ret_right:
            objpoints.append(objp)  # Append directly
            imgpoints_left.append(corners_left.reshape(1, -1, 2))  # Flatten to (N, 2)
            imgpoints_right.append(corners_right.reshape(1, -1, 2))  # Flatten to (N, 2)
        else:
            print(f"Skipping image pair: {img_left}, {img_right}")

    # Validate input consistency
    assert (
        len(objpoints) == len(imgpoints_left) == len(imgpoints_right)
    ), "Mismatch in the number of object and image points."

    # Debugging: Print data consistency
    print(f"Number of valid image pairs: {len(objpoints)}")
    print(
        f"Example objpoints[0] shape: {objpoints[0].shape}, dtype: {objpoints[0].dtype}"
    )
    print(
        f"Example imgpoints_left[0] shape: {imgpoints_left[0].shape}, dtype: {imgpoints_left[0].dtype}"
    )
    print(
        f"Example imgpoints_right[0] shape: {imgpoints_right[0].shape}, dtype: {imgpoints_right[0].dtype}"
    )

    # Stereo calibration
    ret, K1, D1, K2, D2, R, T, E, F = cv2.fisheye.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        K1,
        D1,
        K2,
        D2,
        gray_left.shape[::-1],
        flags=cv2.fisheye.CALIB_FIX_INTRINSIC,
    )

    # Debugging: Baseline validation
    baseline = np.linalg.norm(T)
    print(f"Baseline (distance between cameras): {baseline:.4f} meters")

    return R, T


def rectify_stereo(K1, D1, K2, D2, R, T, image_size, left_image, right_image):
    """
    Rectify stereo images, compute Q matrix, and visualize rectified images.
    """
    # Perform rectification
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY
    )

    # Compute rectification maps
    map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(
        K1, D1, R1, P1, image_size, cv2.CV_16SC2
    )
    map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(
        K2, D2, R2, P2, image_size, cv2.CV_16SC2
    )

    # Apply rectification maps to the input images
    rectified_left = cv2.remap(
        left_image, map1_left, map2_left, interpolation=cv2.INTER_LINEAR
    )
    rectified_right = cv2.remap(
        right_image, map1_right, map2_right, interpolation=cv2.INTER_LINEAR
    )

    # Stack images for comparison (2x2 grid)
    top_row = np.hstack((left_image, right_image))  # Original images
    bottom_row = np.hstack((rectified_left, rectified_right))  # Rectified images
    grid = np.vstack((top_row, bottom_row))

    # Add labels for better clarity
    grid = cv2.putText(
        grid,
        "Before Rectification",
        (50, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    grid = cv2.putText(
        grid,
        "After Rectification",
        (50, top_row.shape[0] + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    # Visualization using OpenCV
    cv2.imshow("Before and After Rectification (2x2 Grid)", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return R1, R2, P1, P2, Q


def compute_disparity_and_point_cloud(
    K1, D1, K2, D2, R1, R2, P1, P2, Q, image_size, left_image, right_image
):
    """
    Compute disparity map and generate point cloud.
    """
    # Compute rectification maps
    map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(
        K1, D1, R1, P1, image_size, cv2.CV_16SC2
    )
    map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(
        K2, D2, R2, P2, image_size, cv2.CV_16SC2
    )

    # Apply rectification maps to the input images
    rectified_left = cv2.remap(
        left_image, map1_left, map2_left, interpolation=cv2.INTER_LINEAR
    )
    rectified_right = cv2.remap(
        right_image, map1_right, map2_right, interpolation=cv2.INTER_LINEAR
    )

    left_gray = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

    # # Create a copy of the images for visualization
    # left_vis = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
    # right_vis = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)

    # # Draw horizontal lines at regular intervals
    # num_lines = 10
    # height, width = left_gray.shape
    # step = height // num_lines

    # for y in range(0, height, step):
    #     color = (0, 255, 0)  # Green
    #     cv2.line(left_vis, (0, y), (width, y), color, 1)
    #     cv2.line(right_vis, (0, y), (width, y), color, 1)

    # # Combine images side by side for visualization
    # combined = np.hstack((left_vis, right_vis))

    # # Display the result
    # cv2.imshow("Epipolar Alignment (Rectified Images)", combined)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # Stack images for comparison (2x2 grid)
    # top_row = np.hstack((left_image, right_image))  # Original images
    # bottom_row = np.hstack((rectified_left, rectified_right))  # Rectified images
    # grid = np.vstack((top_row, bottom_row))

    # # Add labels for better clarity
    # grid = cv2.putText(
    #     grid,
    #     "Before Rectification",
    #     (50, 30),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (255, 0, 0),
    #     2,
    #     cv2.LINE_AA,
    # )
    # grid = cv2.putText(
    #     grid,
    #     "After Rectification",
    #     (50, top_row.shape[0] + 30),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (255, 0, 0),
    #     2,
    #     cv2.LINE_AA,
    # )

    # # Visualization using OpenCV
    # cv2.imshow("Before and After Rectification (2x2 Grid)", grid)
    # cv2.waitKey(0)

    # combined = np.hstack((left_gray, right_gray))
    # cv2.imshow("Rectified Stereo Images", combined)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()

    numDisparities = 16 * 4  # Adjust for your scene
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=numDisparities,
        blockSize=7,
        P1=8 * 3 * 7**2,
        P2=32 * 3 * 7**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,  # Increase for more unique matches
        speckleWindowSize=100,
        speckleRange=2,
    )
    # stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=13)

    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disparity[disparity < 0] = 0  # Replace invalid values
    disparity[disparity > numDisparities] = numDisparities  # Clip large values

    plt.figure(figsize=(18, 6))

    # Left image
    plt.subplot(1, 3, 1)
    plt.imshow(left_gray, cmap="gray")
    plt.title("Left Gray Image")
    plt.axis("off")

    # Right image
    plt.subplot(1, 3, 2)
    plt.imshow(right_gray, cmap="gray")
    plt.title("Right Gray Image")
    plt.axis("off")

    # Disparity map
    plt.subplot(1, 3, 3)
    plt.imshow(disparity, cmap="gray")
    plt.colorbar(label="Disparity Value")
    plt.title("Disparity Map")
    plt.axis("off")

    # Display the plots
    plt.tight_layout()
    plt.show()

    # Reproject disparity to 3D space
    points_3d = cv2.reprojectImageTo3D(disparity, Q)

    # Create mask to filter out invalid points
    mask = (disparity > 0) & (disparity < 64)
    points = points_3d[mask]
    colors = left_image[mask]

    # Plot point cloud using Matplotlib
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Downsample points for faster visualization
    sampled_points = points[::100]  # Take every 10th point
    sampled_colors = colors[::100] / 255.0  # Normalize colors to [0, 1]

    ax.scatter(
        sampled_points[:, 0],
        sampled_points[:, 1],
        sampled_points[:, 2],
        c=sampled_colors,
        s=0.5,
        marker="o",
    )

    ax.set_title("3D Point Cloud")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


# --------------------------
# Main Script
# --------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fisheye Stereo Calibration and Point Cloud Generation"
    )
    parser.add_argument(
        "--collect", action="store_true", help="Collect calibration images"
    )
    parser.add_argument("--test", action="store_true", help="Collect test images")
    parser.add_argument(
        "--calibrate", action="store_true", help="Perform camera calibration"
    )
    parser.add_argument(
        "--rectify", action="store_true", help="Perform stereo rectification"
    )
    parser.add_argument(
        "--pointcloud", action="store_true", help="Generate point cloud"
    )
    parser.add_argument(
        "--time-str",
        type=str,
        default="20241201_142227",
        help="Time string for the experiment",
    )
    args = parser.parse_args()

    checkerboard_size = (9, 6, 0.06)
    image_size = (640, 480)

    if len(args.time_str) > 0:
        time_str = args.time_str
    else:
        time_str = time.strftime("%Y%m%d_%H%M%S")

    exp_name = "stereo_calibration"
    exp_folder_path = f"results/{exp_name}_{time_str}"
    os.makedirs(exp_folder_path, exist_ok=True)

    calib_params_path = os.path.join("toddlerbot", "sensing", "calibration_params.pkl")
    rec_params_path = os.path.join("toddlerbot", "sensing", "rectification_params.npz")

    if args.collect:
        collect_images(0, 4, exp_folder_path, 20)

    if args.test:
        collect_test_images(0, 4, exp_folder_path)

    if args.calibrate:
        left_images = sorted(glob.glob(os.path.join(exp_folder_path, "left_*.png")))
        right_images = sorted(glob.glob(os.path.join(exp_folder_path, "right_*.png")))

        K1, D1 = fisheye_calibration(left_images, checkerboard_size)
        K2, D2 = fisheye_calibration(right_images, checkerboard_size)

        R, T = stereo_calibration(
            left_images, right_images, checkerboard_size, K1, D1, K2, D2
        )
        calib_params = {
            "K1": K1,
            "D1": D1,
            "K2": K2,
            "D2": D2,
            "R": R,
            "T": T,
        }

        with open(calib_params_path, "wb") as f:
            pickle.dump(calib_params, f)

    # if args.rectify:
    #     with open(calib_params_path, "rb") as f:
    #         calib_params = pickle.load(f)

    #     left_image = cv2.imread(os.path.join(exp_folder_path, "left_00.png"))
    #     right_image = cv2.imread(os.path.join(exp_folder_path, "right_00.png"))
    #     K1, D1, K2, D2, R, T = (
    #         calib_params["K1"],
    #         calib_params["D1"],
    #         calib_params["K2"],
    #         calib_params["D2"],
    #         calib_params["R"],
    #         calib_params["T"],
    #     )
    #     R1, R2, P1, P2, Q = rectify_stereo(
    #         K1, D1, K2, D2, R, T, image_size, left_image, right_image
    #     )

    #     rec_params = {
    #         "R1": R1,
    #         "R2": R2,
    #         "P1": P1,
    #         "P2": P2,
    #         "Q": Q,
    #     }

    #     with open(rec_params_path, "wb") as f:
    #         pickle.dump(rec_params, f)

    # if args.pointcloud:
    #     with open(calib_params_path, "rb") as f:
    #         calib_params = pickle.load(f)

    #     with open(rec_params_path, "rb") as f:
    #         rec_params = pickle.load(f)

    #     K1, D1, K2, D2, R, T = (
    #         calib_params["K1"],
    #         calib_params["D1"],
    #         calib_params["K2"],
    #         calib_params["D2"],
    #         calib_params["R"],
    #         calib_params["T"],
    #     )

    #     R1 = rec_params["R1"]
    #     R2 = rec_params["R2"]
    #     P1 = rec_params["P1"]
    #     P2 = rec_params["P2"]
    #     Q = rec_params["Q"]

    #     left_image = cv2.imread(os.path.join(exp_folder_path, "left_test_00.png"))
    #     right_image = cv2.imread(os.path.join(exp_folder_path, "right_test_00.png"))
    #     compute_disparity_and_point_cloud(
    #         K1, D1, K2, D2, R1, R2, P1, P2, Q, image_size, left_image, right_image
    #     )
