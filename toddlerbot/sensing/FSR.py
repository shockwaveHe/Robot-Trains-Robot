import platform

import numpy as np
import serial
from toddlerbot.utils.file_utils import find_ports


class FSR:
    """A class for interfacing with the FSR sensors on the robot."""

    def __init__(self, baud_rate=115200):
        """Initializes the FSR interface with a specified baud rate for serial communication.

        Args:
            baud_rate (int, optional): The baud rate for the serial connection. Defaults to 115200.

        Reads the most recent FSR values from the serial port, which are percentages ranging from 0 to 100. Returns (0, 0) if no valid data is available and retries up to two times if an error occurs.
        """
        os_type = platform.system()
        if os_type == "Linux":
            description = "/dev/ttyACM*"
        elif os_type == "Windows":
            description = "USB Serial Device"

        fsr_ports = find_ports(description)

        # Configure the serial connection
        self.serial_port = fsr_ports[0]
        self.baud_rate = baud_rate

        # Open the serial port
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            print(f"Connected to {self.serial_port} at {self.baud_rate} baud.")
        except serial.SerialException as e:
            print(f"Failed to connect to {self.serial_port}: {e}")
            raise Exception("Error: Could not open FSR interface.")

    def get_state(self):
        """Reads the most recent Force Sensitive Resistor (FSR) values from the serial port and returns them as percentages.

        The function attempts to read the FSR values, which are expected to be in the range of 0 to 100 percent. If no valid data is available, it returns (0, 0). The function retries up to two additional times if an error occurs during the reading process.

        Returns:
            tuple: A tuple containing two float values representing the FSR values as percentages. Returns (0, 0) if no valid data is available or if an error persists after retries.
        """
        # Reads the most recent FSR values from the serial port.
        # The FSR values are percentage in the range from 0 to 100.
        # Returns (0, 0) if no valid data is available.
        # Retries up to two times if an error occurs.
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
        """Closes the connection to the FSR interface.

        This method closes the serial connection to the FSR (Force Sensing Resistor) interface and prints a confirmation message indicating that the connection has been successfully closed.
        """
        self.ser.close()
        print("Closed connection to FSR interface.")
