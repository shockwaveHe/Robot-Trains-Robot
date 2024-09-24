import glob
import time

import numpy as np
import serial
import platform

class FSR:
    def __init__(self, port_pattern="/dev/tty.usbmodem*", baud_rate=115200):
        # If on ubuntu
        os_type = platform.system()
        if os_type == "Linux":
            port_pattern = "/dev/ttyACM*"
        # Automatically detect the correct serial port
        matching_ports = glob.glob(port_pattern)
        if not matching_ports:
            raise Exception(f"No ports found matching pattern {port_pattern}")
        else:
            print(f"Found FSR interface ports: {matching_ports}")
            print(f"Using FSR interface port: {matching_ports[0]}")

        # Configure the serial connection
        self.serial_port = matching_ports[0]
        self.baud_rate = baud_rate

        # Open the serial port
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            print(f"Connected to {self.serial_port} at {self.baud_rate} baud.")
        except serial.SerialException as e:
            print(f"Failed to connect to {self.serial_port}: {e}")
            raise Exception("Error: Could not open FSR interface.")

    def get_state(self):
        """
        Reads the most recent FSR values from the serial port.
        The FSR values are percentage in the range from 0 to 100.
        Returns (0, 0) if no valid data is available.
        Retries up to two times if an error occurs.
        """
        for attempt in range(3):  # Try up to 3 times (initial attempt + 2 retries)
            try:
                # Flush the input buffer to discard old data
                self.ser.flushInput()

                # Read all available lines in the buffer
                data = self.ser.readline()

                if data:
                    if len(data) == 15:
                        # Decode and use the line of data
                        latest_data = data.decode("utf-8").rstrip()
                        # print(latest_data)
                        posR = float(latest_data.split(",")[0])
                        posL = float(latest_data.split(",")[1])
                        posR = np.clip(posR, 0.0, 2.0) / 2.0 * 100
                        posL = np.clip(posL, 0.0, 2.0) / 2.0 * 100
                        # print(f"Received: posL={posL}, posR={posR}")
                        return posL, posR
                else:
                    return None, None

            except Exception as e:
                if attempt >= 2:  # Only wait and retry if there are retries left
                    print(
                        f"Error reading FSR data after {attempt + 1} attempts. Because {e}"
                    )
                    return 0.0, 0.0

    def close(self):
        self.ser.close()
        print("Closed connection to FSR interface.")
