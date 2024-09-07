import time

import cv2
import numpy as np


class Camera:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(self.camera_id)
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

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
