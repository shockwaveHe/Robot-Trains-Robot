import pickle
import time

import cv2
import numpy as np
import zmq

from toddlerbot.utils.comm_utils import ZMQNode

# class ZMQNode:
#     def __init__(self, type="sender", ip=None, queue_len=1):
#         self.type = type
#         if type not in ["sender", "receiver"]:
#             raise ValueError("ZMQ type must be either 'sender' or 'receiver'")

#         self.queue_len = queue_len
#         self.ip = ip
#         self.start_zmq()

#     def start_zmq(self):
#         # Set up ZeroMQ context and socket for data exchange
#         self.zmq_context = zmq.Context()
#         if self.type == "sender":
#             self.socket = self.zmq_context.socket(zmq.PUSH)
#             # Set high water mark and enable non-blocking send
#             self.socket.setsockopt(
#                 zmq.SNDHWM, self.queue_len
#             )  # Limit queue to 10 messages
#             self.socket.setsockopt(
#                 zmq.IMMEDIATE, 1
#             )  # Prevent blocking if receiver is not available
#             self.socket.connect("tcp://" + self.ip + ":5555")
#         elif self.type == "receiver":
#             self.socket = self.zmq_context.socket(zmq.PULL)
#             self.socket.bind("tcp://0.0.0.0:5555")

#     def send_msg(self, send_dict):
#         if self.type != "sender":
#             raise ValueError("ZMQ type must be 'sender' to send messages")

#         # Serialize the numpy array using pickle
#         serialized_array = pickle.dumps(send_dict)
#         # Send the serialized data
#         try:
#             # Send the serialized data with non-blocking to avoid hanging if the queue is full
#             self.socket.send(serialized_array, zmq.NOBLOCK)
#             # print("Message sent!")
#         except zmq.Again:
#             pass

#     def get_msg(self):
#         if self.type != "receiver":
#             raise ValueError("ZMQ type must be 'receiver' to receive messages")

#         try:
#             # Non-blocking receive
#             serialized_array = self.socket.recv(zmq.NOBLOCK)
#             send_dict = pickle.loads(serialized_array)
#             return send_dict
#         except zmq.Again:
#             # No data is available
#             print("No message available right now")
#             return None


class ZMQSender:
    def __init__(self, ip, queue_len=1):
        self.queue_len = queue_len
        self.ip = ip
        self.start_zmq()

    def start_zmq(self):
        # Set up ZeroMQ context and socket for receiving data
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.PUSH)
        # Set high water mark and enable non-blocking send
        self.socket.setsockopt(zmq.SNDHWM, self.queue_len)  # Limit queue to 10 messages
        self.socket.setsockopt(
            zmq.IMMEDIATE, 1
        )  # Prevent blocking if receiver is not available
        self.socket.connect("tcp://" + self.ip + ":5555")

    def send_msg(self, send_dict):
        # Serialize the numpy array using pickle
        serialized_array = pickle.dumps(send_dict)
        # Send the serialized data
        try:
            # Send the serialized data with non-blocking to avoid hanging if the queue is full
            self.socket.send(serialized_array, zmq.NOBLOCK)
            # print("Message sent!")
        except zmq.Again:
            pass


class ZMQReceiver:
    def __init__(self):
        self.start_zmq()

    def start_zmq(self):
        # Set up ZeroMQ context and socket for receiving data
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.PULL)
        self.socket.bind("tcp://0.0.0.0:5555")

    def get_zmq_data(self):
        try:
            # Non-blocking receive
            serialized_array = self.socket.recv(zmq.NOBLOCK)
            send_dict = pickle.loads(serialized_array)
            return send_dict
        except zmq.Again:
            # No data is available
            print("No message available right now")
            return None


send_to_remote = True
if send_to_remote:
    sender = ZMQNode(type="sender", ip="192.168.0.46")

# Open the camera (0 is the default camera)
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
# Set the resolution to 640x360
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# Get the fourcc code of the video format
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

print(f"Video format (FOURCC): {fourcc_str}")

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Capture frames in a loop
while True:
    # Capture frame-by-frame
    t1 = time.time()
    ret, frame = cap.read()
    orig_frame = frame.copy()

    # Encode the frame as a JPEG with quality of 90
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, img_encoded_jpeg = cv2.imencode(".jpg", frame, encode_param)

    # # resize frame to 171x96
    # frame = cv2.resize(frame, (171, 96))
    # # crop center part of frame to 96x96
    # frame = frame[0:96, 38:134]

    # If frame was not captured successfully, break the loop
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # print(f"Frame size: {frame.shape[1]}x{frame.shape[0]} (width x height)")
    # Display the resulting frame
    cv2.imshow("Camera Frame", frame)
    # # set window size
    # cv2.resizeWindow("Camera Frame", 640, 360)

    if send_to_remote:
        send_dict = {"image": np.array(orig_frame), "time": time.time()}
        sender.send_msg(send_dict)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    t2 = time.time()
    print(f"Capture frame: {t2 - t1:.2f} s")

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
