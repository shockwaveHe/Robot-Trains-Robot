import pickle
import time

import cv2
import numpy as np
import zmq

# Set up ZeroMQ context and socket for sending data
context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.connect("tcp://10.5.6.190:5555")  # Connect to the server's IP address and port

# Run the loop to send data at 30Hz
while True:
    # Example numpy array (you can dynamically update this array)
    array = np.random.rand(640, 480, 3) * 255
    array = np.array(array, dtype=np.uint8)
    ret, buffer = cv2.imencode(".png", array)
    send_dict = {"image": buffer.tobytes(), "time": time.time()}

    # Serialize the numpy array using pickle
    serialized_array = pickle.dumps(send_dict)

    # Send the serialized data
    socket.send(serialized_array)

    # Maintain 30Hz
    time.sleep(1 / 30)
