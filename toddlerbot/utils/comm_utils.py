import pickle
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
import zmq


@dataclass
class ZMQMessage:
    """Data class for ZMQ messages."""

    time: float
    control_inputs: Optional[Dict[str, float]] = None
    action: Optional[npt.NDArray[np.float32]] = None
    fsr: Optional[npt.NDArray[np.float32]] = None
    camera_frame: Optional[npt.NDArray[np.uint8]] = None
    arm_force: Optional[npt.NDArray[np.float32]] = None
    arm_torque: Optional[npt.NDArray[np.float32]] = None
    arm_ee_pos: Optional[npt.NDArray[np.float32]] = None
    arm_ee_vel: Optional[npt.NDArray[np.float32]] = None
    ee_pos_target: Optional[npt.NDArray[np.float32]] = None
    lin_vel: Optional[npt.NDArray[np.float32]] = None
    is_done: Optional[bool] = False
    is_stopped: Optional[bool] = False
    is_paused: Optional[bool] = False
    total_steps: Optional[int] = 0
    external_guidance_stage: Optional[str] = "enhance"
    torso_roll: Optional[float] = None
    torso_pitch: Optional[float] = None


def sync_time(ip: str):
    assert len(ip) > 0, "IP address must be provided!"
    try:
        result = subprocess.run(
            f"sudo ntpdate -u {ip}",
            shell=True,
            text=True,
            check=True,
            stdout=subprocess.PIPE,
        )
        print(result.stdout.strip())
    except Exception:
        print("Failed to sync time with the follower!")


def sync_time(ip: str):
    assert len(ip) > 0, "IP address must be provided!"
    try:
        result = subprocess.run(
            f"sudo ntpdate -u {ip}",
            shell=True,
            text=True,
            check=True,
            stdout=subprocess.PIPE,
        )
        print(result.stdout.strip())
    except Exception:
        print("Failed to sync time with the follower!")


def sync_time(ip: str):
    """Synchronizes the system time with a network time server.

    This function connects to a network time server specified by the given IP address and adjusts the system clock to match the server's time.

    Args:
        ip (str): The IP address of the network time server to synchronize with.
    """
    assert len(ip) > 0, "IP address must be provided!"
    try:
        result = subprocess.run(
            f"sudo ntpdate -u {ip}",
            shell=True,
            text=True,
            check=True,
            stdout=subprocess.PIPE,
        )
        print(result.stdout.strip())
    except Exception:
        print("Failed to sync time with the follower!")


class ZMQNode:
    """A class for handling ZMQ communication between sender and receiver nodes."""

    def __init__(self, type: str = "sender", ip: str = "", queue_len: int = 1):
        """Initializes a ZMQ connection with specified type, IP, and queue length.

        Args:
            type (str): The type of ZMQ connection, either 'sender' or 'receiver'. Defaults to 'sender'.
            ip (str): The IP address for the connection. Defaults to an empty string, which is replaced by '127.0.0.1'.
            queue_len (int): The length of the message queue. Defaults to 1.

        Raises:
            ValueError: If the type is not 'sender' or 'receiver'.
        """
        self.type = type
        if type not in ["sender", "receiver"]:
            raise ValueError("ZMQ type must be either 'sender' or 'receiver'")

        self.queue_len = queue_len
        self.ip = ip if len(ip) > 0 else "127.0.0.1"
        self.start_zmq()

    def start_zmq(self):
        """Initialize a ZeroMQ context and socket for data exchange based on the specified type.

        Sets up a ZeroMQ context and configures a socket as either a sender or receiver. For a sender, it connects to a specified IP and port, setting options to manage message queue length and non-blocking behavior. For a receiver, it binds to a port and configures options to manage message queue length and ensure only the latest message is kept.
        """
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
        print(f"ZMQ {self.type} started at {self.ip}")

    def send_msg(self, msg: ZMQMessage):
        """Sends a serialized ZMQMessage if the instance type is 'sender'.

        Args:
            msg (ZMQMessage): The message to be sent, which will be serialized.

        Raises:
            ValueError: If the instance type is not 'sender'.
        """
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

    def get_msg(self, return_last: bool = True):
        """Retrieves messages from a ZMQ socket buffer until it is empty.

        This method is designed to handle cases where reading from the buffer is too slow, causing issues with simple get operations. It reads all available messages from the buffer and returns either the last message or all messages, depending on the `return_last` parameter.

        Args:
            return_last (bool): If True, returns only the last message received. If False, returns all messages. Defaults to True.

        Returns:
            ZMQMessage or List[ZMQMessage] or None: The last message if `return_last` is True, a list of all messages if `return_last` is False, or None if no messages are available.

        Raises:
            ValueError: If the ZMQ socket type is not 'receiver'.
        """

        # For some reason a simple get is not working. buffer will blow up when read speed is too slow
        # So we will read all the way until the buffer if empty to bypass this problem
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

    def close(self):
        self.socket.close()
        self.zmq_context.term()
        print(f"ZMQ {self.type} closed")
