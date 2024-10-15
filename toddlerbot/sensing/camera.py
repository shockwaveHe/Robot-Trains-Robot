import time

import cv2
import numpy as np


class Camera:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        if not self.cap.isOpened():
            raise Exception("Error: Could not open camera.")

    def get_state(self):
        ret, frame = self.cap.read()

        if not ret:
            raise Exception("Error: Failed to capture frame.")

        frame = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        return frame

    def get_jpeg(self):
        frame = self.get_state()
        # Encode the frame as a JPEG with quality of 90
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        _, img_encoded_jpeg = cv2.imencode(".jpg", frame, encode_param)
        return img_encoded_jpeg, frame

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
