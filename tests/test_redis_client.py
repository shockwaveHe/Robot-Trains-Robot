import pickle
import time

import cv2
import numpy as np
import zmq

# Set up ZeroMQ context and socket for sending data
context = zmq.Context()
socket = context.socket(zmq.PUSH)
# Set high water mark and enable non-blocking send
socket.setsockopt(zmq.SNDHWM, 1)  # Limit queue to 10 messages
# socket.setsockopt(zmq.IMMEDIATE, 1)  # Prevent blocking if receiver is not available
socket.connect("tcp://10.5.6.212:5555")  # Connect to the server's IP address and port

# Run the loop to send data at 30Hz
while True:
    # Example numpy array (you can dynamically update this array)
    array = np.random.rand(640, 480, 3) * 255
    array = np.array(array, dtype=np.uint8)
    ret, buffer = cv2.imencode(".png", array)
    send_dict = {"image": buffer.tobytes(), "time": time.time()}

    # Serialize the numpy array using pickle
    serialized_array = pickle.dumps(send_dict)

    try:
        # Send the serialized data with non-blocking to avoid hanging if the queue is full
        socket.send(serialized_array, zmq.NOBLOCK)
        print("Message sent!")
    except zmq.Again:
        # Queue is full, drop the oldest message
        print("Message queue full, dropping message.")

    # Maintain 30Hz
    time.sleep(1 / 50)
