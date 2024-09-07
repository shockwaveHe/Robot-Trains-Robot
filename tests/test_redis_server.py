import pickle
import time

import cv2
import numpy as np
import zmq

# Set up ZeroMQ context and socket for receiving data
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://0.0.0.0:5555")  # Listen on all interfaces

# Receive data continuously
while True:
    # Receive the serialized numpy array
    serialized_array = socket.recv()

    # Deserialize the array
    send_dict = pickle.loads(serialized_array)

    tm = send_dict["time"]
    nparr = np.frombuffer(send_dict["image"], np.uint8)
    img = np.array(cv2.imdecode(nparr, cv2.IMREAD_COLOR))

    t_rcv = time.time()
    print(f"Received array:\n{tm}, timenow: {t_rcv}")
    print(img.shape)
    t_diff = t_rcv - tm
    print(f"Time difference: {t_diff:.4f} seconds")
