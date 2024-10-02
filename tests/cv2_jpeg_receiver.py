import pickle
import time

import cv2
import numpy as np

from toddlerbot.utils.comm_utils import ZMQNode

receiver = ZMQNode(type="receiver")

# Receive data continuously
while True:
    data_dict = receiver.get_msg()

    if data_dict is None:
        # time.sleep(0.3)
        # print("No data received")
        continue

    msg_time = data_dict["time"]

    frame = data_dict["camera_frame"]
    frame = np.frombuffer(frame, np.uint8)
    frame = cv2.cvtColor(cv2.imdecode(frame, cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)

    print(frame.shape)

    latency = time.time() - msg_time
    print(f"Latency: {latency*1000:.2f} ms")

    cv2.imshow("Received Image", frame)
    cv2.waitKey(1)

    print(data_dict.keys())
