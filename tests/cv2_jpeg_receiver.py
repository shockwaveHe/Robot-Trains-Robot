import time

import cv2
import numpy as np

from toddlerbot.utils.comm_utils import ZMQNode

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
    print(f"Latency: {latency*1000:.2f} ms")

    cv2.imshow("Received Image", frame)
    cv2.waitKey(1)
