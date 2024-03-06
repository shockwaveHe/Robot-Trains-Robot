"""
Ben Katz
Motor Module Python API
Assumes the serial device is a nucleo running the firmware at:
Corresponding STM32F446 Firmware here:
https://os.mbed.com/users/benkatz/code/CanMaster/
"""

import struct
import time
from dataclasses import dataclass
from typing import Dict

import serial

from toddlerbot.actuation.base_controller import BaseController

BAUDRATE = 115200


@dataclass
class SunnySkyCommand:
    id: int = 1
    p_des: float = 0
    v_des: float = 0
    kp: int = 40
    kd: int = 8
    i_ff: float = 0


class SunnySkyController(BaseController):
    def __init__(self, port):
        self.tx_data_prefix = ">tx_data:"
        self.tx_timeout = 1.0
        try:
            self.ser = serial.Serial(port, baudrate=BAUDRATE, timeout=0.05)
            time.sleep(1)
            print("Connected to motor module controller.")
        except:
            raise ValueError("Failed to connect to motor module controller!")

    def send_command(self, command: SunnySkyCommand):
        """
        send_command(desired position, desired velocity, position gain, velocity gain, feed-forward current)

        Sends data over CAN, reads response, and populates rx_data with the response.
        """
        b = struct.pack(
            "<B2f2Hf",
            command.id,
            command.p_des,
            command.v_des,
            command.kp,
            command.kd,
            command.i_ff,
        )
        self.serial_write(b)

        self.ser.reset_input_buffer()

        start_time = time.time()
        while time.time() - start_time < self.tx_timeout:
            line = self.ser.readline()
            decoded_line = line.decode().strip()
            if decoded_line.startswith(self.tx_data_prefix):
                _, data_str = decoded_line.split(self.tx_data_prefix, 1)
                id, p, v, t, vb = map(float, data_str.split(","))
                return int(id), p, v, t, vb

        return None

    def enable_motor(self, id):
        """
        Puts motor with CAN ID "id" into torque-control mode. 2nd red LED will turn on
        """
        b = b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFC"
        b = b + bytes(bytearray([id]))
        self.serial_write(b)
        # time.sleep(.1)
        # self.ser.flushInput()

    def disable_motor(self, id):
        """
        Removes motor with CAN ID "id" from torque-control mode. 2nd red LED will turn off
        """
        b = b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFD"
        b = b + bytes(bytearray([id]))
        self.serial_write(b)

    def zero_motor(self, id):
        """
        Zero Position Sensor. Sets the mechanical position to zero.
        """
        b = b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFE"
        b = b + bytes(bytearray([id]))
        self.serial_write(b)

    def serial_write(self, data):
        data = b"<" + data + b">"
        self.ser.write(data)


if __name__ == "__main__":
    # Example usage
    controller = SunnySkyController("/dev/tty.usbmodem1101")

    id = 1
    p_des = 1.57
    p_tol = 0.01
    p_error = float("inf")

    print("Initializing motor controller...")

    try:
        # Enable motor
        print(f"Enabling motor with ID {id}...")
        controller.enable_motor(id)

        while p_error > p_tol:
            start_time = time.time()

            # Adjust kd based on the position error
            kd = 50 if p_error > p_tol else 8

            # Create command with dynamic kd and fixed kp, i_ff
            cmd = SunnySkyCommand(id=id, p_des=p_des, v_des=0.0, kp=40, kd=kd, i_ff=0.0)

            # Send command and receive output
            out = controller.send_command(cmd)

            # Calculate position error
            p_error = abs(p_des - out[1])

            # Print command output and performance metrics
            print(
                f"Command Output: ID={out[0]}, Position={out[1]}, Velocity={out[2]}, Current={out[3]}, Voltage={out[4]}"
            )
            print(f"Position Error: {p_error}")
            print(f"Control Frequency: {1 / (time.time() - start_time):.2f} Hz")

    finally:
        # Disable motor
        print(f"Disabling motor with ID {id}...")
        controller.disable_motor(id)

    print("Process completed successfully.")
