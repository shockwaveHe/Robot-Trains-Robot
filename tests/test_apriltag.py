import argparse
import os
from typing import Dict, List

import cv2
import numpy as np
import numpy.typing as npt
import pupil_apriltags as apriltag


class AprilTagDetector:
    def __init__(self, families: str = "tag36h11") -> None:
        self.detector = apriltag.Detector(
            families=families, quad_decimate=1.0, decode_sharpening=0.25
        )

    def detect(
        self,
        img: npt.NDArray[np.uint8],
        intrinsics: Dict[str, float],
        tag_size: float,
    ) -> List[apriltag.Detection]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = self.detector.detect(
            gray,
            # estimate_tag_pose=True,
            # camera_params=[
            #     intrinsics["fx"],
            #     intrinsics["fy"],
            #     intrinsics["cx"],
            #     intrinsics["cy"],
            # ],
            # tag_size=tag_size,
        )
        return results

    def vis_tag(
        self, img: npt.NDArray[np.uint8], results: List[apriltag.Detection]
    ) -> npt.NDArray[np.uint8]:
        for detection in results:
            ptA, ptB, ptC, ptD = [
                tuple(map(int, corner)) for corner in detection.corners
            ]

            cv2.line(img, ptA, ptB, (0, 255, 0), 2)
            cv2.line(img, ptB, ptC, (0, 255, 0), 2)
            cv2.line(img, ptC, ptD, (0, 255, 0), 2)
            cv2.line(img, ptD, ptA, (0, 255, 0), 2)

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
                (0, 255, 0),
                2,
            )

        return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect AprilTags in an image.")
    parser.add_argument("--image", type=str, help="Path to the input image file")
    args = parser.parse_args()

    # Read and process the image
    image = cv2.imread(os.path.join("tests", "images", f"{args.image}.jpg")).astype(
        np.uint8
    )

    intrinsics_color_dict = {
        "fx": 850.0,  # Adjust based on the camera and your experience
        "fy": 850.0,  # Similar to fx for square pixels
        "cx": image.shape[1] / 2,
        "cy": image.shape[0] / 2,
    }
    april_detector = AprilTagDetector()
    results = april_detector.detect(
        image, intrinsics=intrinsics_color_dict, tag_size=0.05
    )
    print("Detected AprilTags:", results)

    image_vis = april_detector.vis_tag(image, results)

    # Display the annotated image
    cv2.imshow("AprilTag", image_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
