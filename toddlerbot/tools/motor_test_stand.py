import time
import numpy as np
from tqdm import tqdm
from toddlerbot.actuation.dynamixel.dynamixel_control import (
    DynamixelConfig,
    DynamixelController,
)
import serial
import matplotlib.pyplot as plt
import joblib

class ArduinoController:
    def __init__(self, port='/dev/ttyUSB0', baud_rate=115200, timeout=1):
        """
        Initialize the serial connection to the Arduino.
        
        :param port: Serial port where Arduino is connected (e.g., 'COM3' or '/dev/ttyUSB0')
        :param baud_rate: Baud rate for serial communication
        :param timeout: Timeout for serial read operations
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial_connection = serial.Serial(port, baud_rate, timeout=timeout)
        time.sleep(3)  # Allow time for Arduino to reset

    def set_brake_value(self, value):
        """
        Send a brake value to the Arduino to control the brake (0-255).
        
        :param value: Brake value between 1 and 255
        """
        if 1 <= value <= 255:
            command = f"{value}\n"  # Newline indicates end of input
            self.serial_connection.write(command.encode('utf-8'))
            print(f"Sent brake value: {value}")
        else:
            raise ValueError("Brake value must be between 1 and 255.")

    def read_sensor(self):
        """
        Read the weight value from the Arduino. 
        
        :return: The weight value read from the Arduino
        """
        factor = 461300.0/0.7725375
        if self.serial_connection.in_waiting > 0:
            line = self.serial_connection.readline().decode('utf-8').strip()
            try:
                raw_torque = float(line)
                return raw_torque/factor
            except ValueError:
                print("Error: Could not parse sensor value.")
                return None
        return None

    def close_connection(self):
        """
        Close the serial connection.
        """
        # Set brake value to 0
        self.set_brake_value(1)
        # Close the serial connection
        if self.serial_connection.is_open:
            self.serial_connection.close()
            print("Serial connection closed.")

class DynController:
    def __init__(self):
        # Initial values for the PID controller
        self.kP = 0.0
        self.kI = 0.0
        self.kD = 0.0
        self.kFF1 = 0.0
        self.kFF2 = 0.0
        self.kVP = 0.0
        self.kVI = 0.0

        # Dynamixel motor ID
        self.id = 1

        self.connect_dynamixel()

    def connect_dynamixel(self):
        # Create a controller object
        config = DynamixelConfig(
            port="/dev/ttyUSB1",  # FT8ISUJY
            baudrate=2e6,
            control_mode=["current"],
            kP=[self.kP],
            kI=[self.kI],
            kD=[self.kD],
            kFF2=[self.kFF2],
            kFF1=[self.kFF1],
            init_pos=[0.0],
        )
        self.controller = DynamixelController(config, motor_ids=[self.id])
        _, self.v_in = self.controller.client.read_vin()
        self.controller.enable_motors()

    def close_connection(self):
        self.set_cur(0)
        self.controller.disable_motors()

    def get_state(self):
        """
        State: [position, velocity, current]
        """
        state_dict = self.controller.get_motor_state()

        return [state_dict[self.id].pos, state_dict[self.id].vel, state_dict[self.id].tor]
    
    def set_cur(self, cur):
        self.controller.set_cur([cur])

def interpolate(t1, t2, t, y1, y2):
    return y1 + (y2-y1)/(t2-t1)*(t-t1)

# Standard boilerplate to run
"""
Data: nt * [time, torque_reading, brake value, position, velocity, current]
"""
if __name__ == "__main__":
    # Initialize Arduino and Dynamixel controllers
    ard_controller = ArduinoController()
    dyn_controller = DynController()

    # Initialize data dictionary
    data_dict = []
    figure_last_time = time.time()
    last_brake_value = 1

    # Initialize the plot
    plt.figure(0)

    # test profile [t, set current, set brake]
    idle_time = 1.5
    test_time = 9
    profile = np.array([[0.0, 910, 1],
                        [idle_time, 910, 1],
                        [idle_time+test_time,  910, 220],
                        [idle_time+test_time+0.5,  0, 1]])
    start_t = None

    while True:
        try:
            # Read weight from Arduino
            raw_torque_reading = ard_controller.read_sensor()
            if raw_torque_reading is not None:
                curr_motor_state = dyn_controller.get_state()
                print(curr_motor_state)
                print(f"Current torque: {raw_torque_reading:.3f}")
                data_dict.append([time.time(), raw_torque_reading, last_brake_value] + curr_motor_state)

                # ========================== Test ==========================        
                # # dyn_controller.set_cur(min(np.sin(time.time()*2)*500+500,910))
                # dyn_controller.set_cur(910)
                # # Set brake value to 100
                # ard_controller.set_brake_value(255)

                # ========================== profile ==========================
                if start_t is None:
                    start_t = time.time()
                t = time.time()-start_t
                if t > profile[-1, 0]:
                    print("Test finished.")
                    time.sleep(10)

                for i in range(profile.shape[0]-1):
                    if profile[i, 0] <= t < profile[i+1, 0]:
                        # Blend the current and brake commands
                        cur_cmd = interpolate(profile[i, 0], profile[i+1, 0], t, profile[i, 1], profile[i+1, 1])
                        brake_cmd = interpolate(profile[i, 0], profile[i+1, 0], t, profile[i, 2], profile[i+1, 2])

                        # set the current and brake values
                        dyn_controller.set_cur(cur_cmd)
                        ard_controller.set_brake_value(brake_cmd)
                        last_brake_value = brake_cmd



            if time.time()-figure_last_time > 0.2:
                t1 = time.time()
                figure_last_time = time.time()
                temp_data_dict = np.array(data_dict)[-1000:]
                plt.clf()
                plt.scatter(temp_data_dict[:, 4]/6.28*60.0, temp_data_dict[:, 1])
                plt.grid()
                plt.pause(1e-3)
                t2 = time.time()
                print(f"Time for plotting: {t2-t1}")
        
        except Exception as e:
            print(f"An error occurred: {e}")

        # at keyboard interrupt, close the serial connection
        except KeyboardInterrupt:
            print(data_dict)
            output_dict = {"time": np.array(data_dict)[:, 0], 
                           "torque": np.array(data_dict)[:, 1], 
                           "brake": np.array(data_dict)[:, 2], 
                           "position": np.array(data_dict)[:, 3], 
                           "velocity": np.array(data_dict)[:, 4], 
                           "current": np.array(data_dict)[:, 5]}
            joblib.dump(output_dict, "./datasets/xc330_test.lz4", compress="lz4")
            print("Data saved.")
            ard_controller.close_connection()
            dyn_controller.close_connection()
            break
    