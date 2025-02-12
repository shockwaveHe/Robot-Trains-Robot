import time

import cv2
import numpy as np

from toddlerbot.utils.comm_utils import ZMQNode

# This script is used to receive JPEG images from a ZMQ connection and display them in real-time.
# You can run this on the remote controller to see what the robot sees.


def main():
    """Continuously receives and processes image data from a ZMQNode receiver.

    This function initializes a ZMQNode as a receiver and enters an infinite loop
    to receive messages containing image data. The image data is decoded and
    converted to BGR format for display. The function also calculates and prints
    the latency of the received message.

    Note:
        The function runs indefinitely and requires a proper setup of a ZMQNode
        to function correctly. It also requires OpenCV and NumPy for image
        processing and display.
    """

    receiver = ZMQNode(type="receiver")

    # Receive data continuously
    while True:
        msg = receiver.get_msg()

        if msg is None:
            # time.sleep(0.3)
            # print("No data received")
            continue

        msg_time = msg.time

        frame = msg.camera_frame
        frame = np.frombuffer(frame, np.uint8)
        frame = cv2.cvtColor(cv2.imdecode(frame, cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)

        print(frame.shape)

        latency = time.time() - msg_time
        print(f"Latency: {latency * 1000:.2f} ms")

        cv2.imshow("Received Image", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
