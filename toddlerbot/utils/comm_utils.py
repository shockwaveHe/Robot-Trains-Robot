import pickle
from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt
import zmq


@dataclass
class ZMQMessage:
    time: float
    is_logging: bool
    control_inputs: dict
    action: npt.NDArray[np.float32]
    fsr: npt.NDArray[np.float32]
    trial: int


"""
Usage:

    - Sender:   ZMQNode(type='sender', ip='10.5.6.171')
    - Receiver: ZMQNode(type='receiver')
"""


class ZMQNode:
    def __init__(self, type: str = "sender", ip: str = "", queue_len: int = 1):
        self.type = type
        if type not in ["sender", "receiver"]:
            raise ValueError("ZMQ type must be either 'sender' or 'receiver'")

        self.queue_len = queue_len
        self.ip = ip
        self.start_zmq()

    def start_zmq(self):
        # Set up ZeroMQ context and socket for data exchange
        self.zmq_context = zmq.Context()
        if self.type == "sender":
            self.socket = self.zmq_context.socket(zmq.PUSH)
            # Set high water mark and enable non-blocking send
            self.socket.setsockopt(
                zmq.SNDHWM, self.queue_len
            )  # Limit queue to 10 messages
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.SNDBUF, 1024)  # Smaller send buffer
            # self.socket.setsockopt(
            #     zmq.IMMEDIATE, 1
            # )  # Prevent blocking if receiver is not available
            self.socket.connect("tcp://" + self.ip + ":5555")

        elif self.type == "receiver":
            self.socket = self.zmq_context.socket(zmq.PULL)
            self.socket.bind("tcp://0.0.0.0:5555")  # Listen on all interfaces
            self.socket.setsockopt(zmq.RCVHWM, 1)  # Limit receiver's queue to 1 message
            self.socket.setsockopt(zmq.CONFLATE, 1)  # Only keep the latest message
            self.socket.setsockopt(zmq.RCVBUF, 1024)

    def send_msg(self, msg: ZMQMessage):
        if self.type != "sender":
            raise ValueError("ZMQ type must be 'sender' to send messages")

        # Serialize the numpy array using pickle
        serialized_array = pickle.dumps(msg)
        # Send the serialized data
        try:
            # Send the serialized data with non-blocking to avoid hanging if the queue is full
            self.socket.send(serialized_array, zmq.NOBLOCK)
            # print("Message sent!")
        except zmq.Again:
            pass

    # For some reason a simple get is not working. buffer will blow up when read speed is too slow
    # So we will read all the way until the buffer if empty to bypass this problem
    def get_msg(self, return_last: bool = True):
        if self.type != "receiver":
            raise ValueError("ZMQ type must be 'receiver' to receive messages")

        messages: List[ZMQMessage] = []
        while True:
            try:
                # Non-blocking receive
                serialized_array = self.socket.recv(zmq.NOBLOCK)
                msg = pickle.loads(serialized_array)
                messages.append(msg)

            except zmq.Again:
                # No more data is available
                break

        if return_last:
            # for message in messages:
            #     print(message["test"], message["time"], time.time())
            return messages[-1] if messages else None
        else:
            return messages if messages else None
