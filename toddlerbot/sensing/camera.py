import os
import pickle
import subprocess
from typing import Dict, List

import cv2
import numpy as np
import numpy.typing as npt
import pupil_apriltags as apriltag


class AprilTagDetector:
    """AprilTag detector class."""

    def __init__(self, families: str = "tag36h11") -> None:
        """Initializes an instance of the class with an AprilTag detector.

        Args:
            families (str): The tag family to be used by the detector. Defaults to "tag36h11".
        """
        self.detector = apriltag.Detector(
            families=families, quad_decimate=1.0, decode_sharpening=0.25
        )

    def detect(
        self,
        img: npt.NDArray[np.uint8],
        intrinsics: Dict[str, float] | npt.NDArray[np.float32],
        tag_size: float,
    ) -> List[apriltag.Detection]:
        """Detect AprilTags in an image and estimate their poses.

        Args:
            img (npt.NDArray[np.uint8]): The input image in BGR format.
            intrinsics (Dict[str, float] | npt.NDArray[np.float32]): Camera intrinsics, either as a dictionary with keys 'fx', 'fy', 'cx', 'cy' or as a 3x3 intrinsic matrix.
            tag_size (float): The size of the AprilTag in meters.

        Returns:
            List[apriltag.Detection]: A list of detected AprilTags with pose estimates.
        """
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

    def vis_tag(
        self, img: npt.NDArray[np.uint8], results: List[apriltag.Detection]
    ) -> npt.NDArray[np.uint8]:
        """Visualizes detected AprilTags on an image by drawing bounding boxes, centers, and labels.

        Args:
            img (npt.NDArray[np.uint8]): The input image on which to draw the detections.
            results (List[apriltag.Detection]): A list of AprilTag detection results, each containing corner points, center, tag family, and tag ID.

        Returns:
            npt.NDArray[np.uint8]: The image with visualized AprilTag detections.
        """
        for detection in results:
            ptA, ptB, ptC, ptD = [
                tuple(map(int, corner)) for corner in detection.corners
            ]

            cv2.line(img, ptA, ptB, (255, 0, 0), 5)
            cv2.line(img, ptB, ptC, (255, 0, 0), 5)
            cv2.line(img, ptC, ptD, (255, 0, 0), 5)
            cv2.line(img, ptD, ptA, (255, 0, 0), 5)

            cX, cY = tuple(map(int, detection.center))
            cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)

            tagFamily = detection.tag_family.decode("utf-8")
            tagID = detection.tag_id
            cv2.putText(
                img,
                f"{tagFamily} {tagID}",
                (ptA[0], ptA[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2,
            )

        return img


class Camera:
    """Camera class for capturing images and detecting AprilTags."""

    def __init__(self, side, width=640, height=480):
        """Initializes the camera setup for either the left or right side, configuring video capture settings and setting up an AprilTag detector.

        Args:
            side (str): Specifies the side of the camera, either 'left' or 'right'.
            width (int, optional): The width of the video capture frame. Defaults to 640.
            height (int, optional): The height of the video capture frame. Defaults to 480.

        Raises:
            Exception: If the camera cannot be opened.
        """
        self.side = side

        video_devices = [
            int(dev[5:]) for dev in os.listdir("/dev") if dev.startswith("video")
        ]
        video_devices = sorted(video_devices)  # Sort numerically
        self.camera_id = video_devices[4] if side == "left" else video_devices[0]

        self.width = width
        self.height = height

        # Run the command
        subprocess.run(
            f"v4l2-ctl --device=/dev/video{self.camera_id} --set-ctrl=auto_exposure=1,exposure_time_absolute=30",
            shell=True,
            text=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        # print(result.stdout.strip())

        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.detector = AprilTagDetector()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        calib_params_path = os.path.join(script_dir, "calibration_params.pkl")
        with open(calib_params_path, "rb") as f:
            calib_params = pickle.load(f)
            self.intrinsics = (
                calib_params["K1"] if side == "left" else calib_params["K2"]
            )

        # Transformation from right camera to left camera
        self.eye_transform = (
            np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            if side == "left"
            else np.array(
                [[0, 0, 1, 0], [-1, 0, 0, -0.033], [0, -1, 0, 0], [0, 0, 0, 1]]
            )
        )
        if not self.cap.isOpened():
            raise Exception("Error: Could not open camera.")

    def get_frame(self):
        """Captures and returns a single frame from the video stream.

        Raises:
            Exception: If the frame could not be captured.

        Returns:
            numpy.ndarray: The captured video frame.
        """
        ret, frame = self.cap.read()

        if not ret:
            raise Exception("Error: Failed to capture frame.")

        return frame

    def get_jpeg(self):
        """Converts the current video frame to JPEG format and returns it along with the RGB frame.

        Returns:
            tuple: A tuple containing:
                - jpeg (numpy.ndarray): The encoded JPEG image.
                - frame_rgb (numpy.ndarray): The RGB representation of the current video frame.
        """
        frame = self.get_frame()
        frame_rgb = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        # Encode the frame as a JPEG with quality of 90
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        _, jpeg = cv2.imencode(".jpg", frame_rgb, encode_param)
        return jpeg, frame_rgb

    def detect_tags(self, tag_size: float = 0.03):
        """Detects AprilTags in the current frame and computes their poses.

        This method captures a frame from the camera, detects AprilTags using the specified tag size, and calculates their poses relative to the camera. The poses are returned as a dictionary mapping tag IDs to their transformation matrices.

        Args:
            tag_size (float): The size of the AprilTags in meters. Default is 0.03.

        Returns:
            Dict[int, npt.NDArray[np.float32]]: A dictionary where keys are tag IDs and values are 4x4 transformation matrices representing the pose of each detected tag in the camera frame.
        """
        frame = self.get_frame()
        results = self.detector.detect(
            frame, intrinsics=self.intrinsics, tag_size=tag_size
        )

        # frame_vis = self.detector.vis_tag(frame.copy(), results)
        # cv2.imshow("AprilTag Detection", frame_vis)
        # cv2.waitKey(1)

        tag_poses: Dict[int, npt.NDArray[np.float32]] = {}
        for detection in results:
            tag_id = detection.tag_id

            # Pose in left camera
            tag_transform = np.eye(4, dtype=np.float32)
            tag_transform[:3, :3] = detection.pose_R
            tag_transform[:3, 3] = detection.pose_t.flatten()

            # Transform into the new left camera frame
            tag_transform = self.eye_transform @ tag_transform

            tag_poses[tag_id] = tag_transform

        return tag_poses

    def close(self):
        """Releases the video capture object and closes all OpenCV windows.

        This method should be called to properly release the resources associated with the video capture and to close any OpenCV windows that were opened during the process.
        """
        self.cap.release()
        cv2.destroyAllWindows()
