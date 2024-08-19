import time

import numpy as np
import serial

from toddlerbot.actuation.dynamixel.dynamixel_control import (
    DynamixelConfig,
    DynamixelController,
)

# Create a controller object
config = DynamixelConfig(
    port="/dev/ttyUSB0",
    baudrate=3000000,
    control_mode=["extended_position"],
    kP=[2400],
    kI=[0.0],
    kD=[2400],
    kFF2=[0.0],
    kFF1=[0.0],
    init_pos=[-3.14 / 3],
    pos_max=-1.0,
    pos_min=-400 / 360 * 2 * np.pi,
    default_vel=10.0,
    interp_method="linear",
)
controller = DynamixelController(config, motor_ids=[17])

# Configure the serial connection
serial_port = "/dev/ttyACM0"
baud_rate = 115200  # Baud rate for the serial communication

# Open the serial port
try:
    ser = serial.Serial(serial_port, baud_rate, timeout=1)
    print(f"Connected to {serial_port} at {baud_rate} baud.")
except serial.SerialException as e:
    print(f"Failed to connect to {serial_port}: {e}")
    exit()

# Read data from the serial port
try:
    while True:
        if ser.in_waiting > 0:
            data = ser.readline().decode("utf-8").rstrip()
            print(f"Received: {data}")
            # convert to float
            pos = float(data)
            controller.set_pos(pos=[-2 * pos - 1])
            # time.sleep(0.01)
except KeyboardInterrupt:
    print("Stopping the serial reading.")

# Close the serial port
ser.close()
print(f"Closed connection to {serial_port}.")


# for i in range(1000):
#     controller.set_pos(pos=[np.sin(i*0.05)+1.1])
#     # state_dict = controller.get_motor_state()
#     print(np.sin(i*0.05)+1.1)
#     # print(state_dict)
#     time.sleep(0.001)

# time.sleep(1)
# controller.set_pos(pos=[0.5])
# state_dict = controller.get_motor_state()
# print(state_dict)
# print('At 0.1')

# time.sleep(1)
# controller.set_pos(pos=[0.0])
# state_dict = controller.get_motor_state()
# print(state_dict)
# print('At 0.0')
# time.sleep(1)

# state_dict = controller.get_motor_state()
# print(state_dict)
# controller.close_motors()
