from multiprocessing import shared_memory
import struct
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
import serial

from toddlerbot.policies import BasePolicy
from toddlerbot.sensing.FSR import FSR
from toddlerbot.sim import Obs
from toddlerbot.sim.robot import Robot
from toddlerbot.tools.keyboard import Keyboard
from toddlerbot.utils.comm_utils import ZMQMessage, ZMQNode
from toddlerbot.utils.math_utils import interpolate_action


class ArmTreadmillLeaderPolicy(BasePolicy, policy_name="arm_treadmill_leader"):
    def __init__(
        self,
        name: str,
        robot: Robot,
        init_motor_pos: npt.NDArray[np.float32],
        keyboard: Optional[Keyboard] = None,
        ip: str = "127.0.0.1",
    ):
        super().__init__(name, robot, init_motor_pos)

        self.zmq = ZMQNode(type="sender", ip=ip)


        self.is_running = False
        self.toggle_motor = True
        self.is_button_pressed = False

        if keyboard is None:
            self.keyboard = Keyboard()
        else:
            self.keyboard = keyboard

        self.reset_duration = 5.0
        self.reset_end_time = 1.0
        self.reset_time = None

        self.speed = 5.0
        self.serial_thread = threading.Thread(target=self.serial_thread_func)
        self.serial_thread.start()

        self.walk_x = 0.0
        self.walk_y = 0.0

        self.stopped = False
        self.force = 10.0
        shm_name = 'force_shm'
        try:
            self.force_shm = shared_memory.SharedMemory(name=shm_name, create=True, size=8)
        except FileExistsError:
            self.force_shm = shared_memory.SharedMemory(name=shm_name, create=False, size=8)
        self.force_shm.buf[:8] = struct.pack('d', self.force)

    # note: calibrate zero at: toddlerbot/tools/calibration/calibrate_zero.py --robot toddlerbot_arms
    # note: zero points can be accessed in config_motors.json

    def step(
        self, obs: Obs, is_real: bool = False
    ) -> Tuple[Dict[str, float], npt.NDArray[np.float32]]:
        if obs.time < self.prep_duration:
            action = np.asarray(
                interpolate_action(obs.time, self.prep_time, self.prep_action)
            )
            return {}, action
        keyboard_inputs = self.keyboard.get_keyboard_input()
        self.walk_x += keyboard_inputs["walk_x_delta"]
        self.walk_y += keyboard_inputs["walk_y_delta"]
        control_inputs = {"walk_x": self.walk_x, "walk_y": self.walk_y}
        self.stopped = keyboard_inputs["stop"]
        self.speed += keyboard_inputs["speed_delta"]
        self.force += keyboard_inputs["force_delta"]
        self.force_shm.buf[:8] = struct.pack('d', self.force)

        if self.stopped:
            self.force = 0.0
            self.force_shm.buf[:8] = struct.pack('d', self.force)
            self.keyboard.close()
            control_inputs = {"walk_x": 0.0, "walk_y": 0.0}
            print("Stopping the system")

        # compile data to send to follower
        msg = ZMQMessage(
            time=time.time(),
            control_inputs=control_inputs,
        )
        # print(f"Sending: {msg}")
        print(f"Speed: {self.speed}, Force: {self.force}, Walk: ({self.walk_x}, {self.walk_y})")
        self.zmq.send_msg(msg)

        if self.stopped:
            self.serial_thread.join()
            self.force_shm.close()
            self.force_shm.unlink()
        return control_inputs, action

    def serial_thread_func(self):
        # Configure the serial connection
        ser = serial.Serial(
            port='/dev/ttyUSB0',     # Replace with your port name on Linux
            baudrate=9600,           # Baud rate
            parity=serial.PARITY_NONE,    # Parity (None)
            stopbits=serial.STOPBITS_ONE,  # Stop bits (1)
            bytesize=serial.EIGHTBITS,     # Data bits (8)
            timeout=0.1              # Set a timeout to prevent blocking
        )

        # Check if the port is open
        if ser.is_open:
            print(f"Connected to {ser.name}")

        try:
            while not self.stopped:
                # Get the current speed
                current_speed = self.speed

                # Prepare the data to send
                data_to_send = f"{current_speed}\n".encode('utf-8')
                ser.write(data_to_send)  # Send the data

                # Optionally read a response (if the device sends back data)
                response = ser.readline()  # Read a line from the device
                if response:
                    print(f"Send: {data_to_send}, Received: {response}")

                # Sleep to prevent excessive CPU usage
                time.sleep(0.01)
        except Exception as e:
            print(f"Error in serial thread: {e}")
        finally:
            # Send zero speed before closing
            ser.write(b"0\n")
            ser.close()
            print("Serial connection closed.")