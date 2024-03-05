"""
Ben Katz
Motor Module Python API
Assumes the serial device is a nucleo running the firmware at:
Corresponding STM32F446 Firmware here:
https://os.mbed.com/users/benkatz/code/CanMaster/
"""

import time
from struct import *

import serial


class MotorModuleController:
    def __init__(self, port):
        try:
            self.ser = serial.Serial(port, timeout=0.05)
            self.ser.baudrate = 115200
            self.rx_data = [0, 0, 0, 0, 0, 0]
            self.rx_values = [0, 0, 0, 0]
            self.tx_data = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            print("connected to motor module controller")
        except:
            print("failed to connect to motor module controller")
            pass

    def send_data(self):
        pass

    def send_command(self, id, p_des, v_des, kp, kd, i_ff):
        """
        send_command(desired position, desired velocity, position gain, velocity gain, feed-forward current)

        Sends data over CAN, reads response, and populates rx_data with the response.
        """
        id = int(id)
        b = (
            bytes(bytearray([id]))
            + pack("f", p_des)
            + pack("f", v_des)
            + pack("f", kp)
            + pack("f", kd)
            + pack("f", i_ff)
        )
        # print(int.from_bytes(b, byteorder='big'))
        self.ser.write(b)

        b_rx = self.ser.read(13)
        self.rx_values[0] = b_rx[0]
        # ID
        self.rx_values[1] = unpack("f", b_rx[1:5])  # Position
        self.rx_values[2] = unpack("f", b_rx[5:9])  # Velocity
        self.rx_values[3] = unpack("f", b_rx[9:13])  # Current

        # print(self.rx_values)

    def enable_motor(self, id):
        """
        Puts motor with CAN ID "id" into torque-control mode.  2nd red LED will turn on
        """
        b = b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFC"
        b = b + bytes(bytearray([id]))
        self.ser.write(b)
        # time.sleep(.1)
        # self.ser.flushInput()

    def disable_motor(self, id):
        """
        Removes motor with CAN ID "id" from torque-control mode.  2nd red LED will turn off
        """
        b = b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFD"
        b = b + bytes(bytearray([id]))
        self.ser.write(b)

    def zero_motor(self, id):
        pass
