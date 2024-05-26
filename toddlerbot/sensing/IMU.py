import threading
import time
from collections import deque

import board
import busio
import numpy as np
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    # BNO_REPORT_MAGNETOMETER,
    BNO_REPORT_ROTATION_VECTOR,
)
from adafruit_bno08x.i2c import BNO08X_I2C
from scipy.spatial.transform import Rotation as R

from toddlerbot.utils.math_utils import quaternion_to_euler_array
from toddlerbot.utils.misc_utils import profile


class IMU:
    def __init__(self, window_size=50, frequency=200):
        # Initialize the I2C bus and sensor
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = BNO08X_I2C(self.i2c)

        # Enable the gyroscope and rotation vector features
        self.sensor.enable_feature(BNO_REPORT_ACCELEROMETER)
        self.sensor.enable_feature(BNO_REPORT_GYROSCOPE)
        self.sensor.enable_feature(BNO_REPORT_ROTATION_VECTOR)

        time.sleep(1.0)

        self.default_pose = None
        self.default_pose_inv = None

        # Initialize history buffers for moving average
        self.window_size = window_size
        self.euler_history = deque(maxlen=window_size)
        self.angular_velocity_history = deque(maxlen=window_size)

        # Set the frequency and start the data acquisition thread
        self.frequency = frequency
        self.stop_event = threading.Event()
        self.data_thread = threading.Thread(target=self._update_buffers)
        self.data_thread.start()

        time.sleep(1.0)

    def set_default_pose(self):
        quat = self.sensor.quaternion
        self.default_pose = R.from_quat(quat)
        self.default_pose_inv = self.default_pose.inv()

    # @profile()
    def get_state(self):
        if self.default_pose is None:
            self.set_default_pose()

        # Compute moving averages
        avg_euler = np.mean(self.euler_history, axis=0)
        rotation_relative = self.default_pose_inv * R.from_euler("xyz", avg_euler)
        euler_relative = rotation_relative.as_euler("xyz")

        avg_angular_velocity = np.mean(self.angular_velocity_history, axis=0)
        ang_vel_relative = self.default_pose_inv.apply(avg_angular_velocity)

        state = {"euler": euler_relative, "angular_velocity": ang_vel_relative}

        return state

    def _update_buffers(self):
        while not self.stop_event.is_set():
            self.euler_history.append(
                quaternion_to_euler_array(self.sensor.quaternion, order="xyzw")
            )
            self.angular_velocity_history.append(np.array(self.sensor.gyro))
            time.sleep(1 / self.frequency)

    def close(self):
        self.stop_event.set()
        self.data_thread.join()


if __name__ == "__main__":
    # import copy

    imu = IMU()
    imu.set_default_pose()

    # last_state = None
    step = 0
    while step < 100:  # True:
        step_start = time.time()
        # acceleration = imu.get_acceleration()
        state = imu.get_state()

        print(f"euler: {state['euler']}, omega: {state['angular_velocity']}")

        # if last_state:
        #     euler_angles_delta = state["euler_angles"] - last_state["euler_angles"]
        #     time_delta = state["time"] - last_state["time"]
        #     ang_vel_fd = euler_angles_delta / time_delta
        #     print(f"omega_fd: {ang_vel_fd}")

        # last_state = copy.deepcopy(state)

        step_time = time.time() - step_start
        print(f"Step time: {step_time * 1000:.3f} ms")

        step += 1

    imu.close()
