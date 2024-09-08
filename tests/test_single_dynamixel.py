import time

import numpy as np
import serial

from toddlerbot.actuation.dynamixel.dynamixel_control import (
    DynamixelConfig,
    DynamixelController,
)

# Create a controller object
config = DynamixelConfig(
    port="/dev/tty.usbserial-FT8ISUJY",
    baudrate=3000000,
    control_mode=["extended_position"],
    kP=[2400],
    kI=[0.0],
    kD=[2400],
    kFF2=[0.0],
    kFF1=[0.0],
    init_pos=[0.0],
)
controller = DynamixelController(config, motor_ids=[1, 2])  # 22,29

# Configure the serial connection
serial_port = "/dev/tty.usbmodem21301"
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
            # convert to two flot in format "0.0000,0.0000"
            # pos = float(data)
            pos0 = float(data.split(",")[0])
            pos1 = float(data.split(",")[1])
            pos0 = np.clip(pos0, 0.0, 2.0)
            pos1 = np.clip(pos1, 0.0, 2.0)
            print("Received:", pos0, pos1)
            controller.set_pos(pos=[pos0 * 0.75, pos1 * 0.75])
            # time.sleep(0.01)
except KeyboardInterrupt:
    print("Stopping the serial reading.")

# Close the serial port
ser.close()
print(f"Closed connection to {serial_port}.")


# for i in range(1000):
#     # t1 = time.time()
#     controller.set_pos(pos=[np.sin(i*0.05)])
#     # time.sleep(0.1)
#     state_dict = controller.get_motor_state()
#     # print(i, np.sin(i*0.05)+1.1)
#     # print(state_dict)
#     # t2 = time.time()
#     # print(1/(t2-t1))
#     time.sleep(0.005)


def state_dict_to_np(state_dict):
    np_array = np.zeros((len(state_dict), 4))
    for i, key in enumerate(state_dict.keys()):
        np_array[i, :] = np.array(
            [key, state_dict[key].time, state_dict[key].pos, state_dict[key].vel]
        )
    return np_array


# for i in range(1000):
#     t1 = time.time()
#     # controller.set_pos(pos=[np.sin(i*0.05)])
#     # time.sleep(0.1)
#     state_dict = controller.get_motor_state()
#     # print(i, np.sin(i*0.05)+1.1)
#     print(state_dict_to_np(state_dict))
#     t2 = time.time()
#     print(1 / (t2 - t1))
#     # time.sleep(0.005)
