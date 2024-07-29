import threading
import time
from collections import deque

import board  # type: ignore
import busio  # type: ignore
import numpy as np
from adafruit_bno08x import (  # type: ignore
    BNO_REPORT_ACCELEROMETER,  # type: ignore
    BNO_REPORT_GYROSCOPE,  # type: ignore
    # BNO_REPORT_MAGNETOMETER,
    BNO_REPORT_ROTATION_VECTOR,  # type: ignore
)
from adafruit_bno08x.i2c import BNO08X_I2C  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore

from toddlerbot.utils.math_utils import quaternion_to_euler_array

# from toddlerbot.utils.misc_utils import profile


class IMU:
    def __init__(self, window_size: int = 50, frequency: int = 200):
        # Initialize the I2C bus and sensor
        self.i2c = busio.I2C(board.SCL, board.SDA)  # type: ignore
        self.sensor = BNO08X_I2C(self.i2c)

        # Enable the gyroscope and rotation vector features
        self.sensor.enable_feature(BNO_REPORT_ACCELEROMETER)  # type: ignore
        self.sensor.enable_feature(BNO_REPORT_GYROSCOPE)  # type: ignore
        self.sensor.enable_feature(BNO_REPORT_ROTATION_VECTOR)  # type: ignore

        time.sleep(1.0)

        self.zero_pose = None
        self.zero_pose_inv = None

        # Initialize history buffers for moving average
        self.euler_history = deque(maxlen=window_size)  # type: ignore
        self.angular_velocity_history = deque(maxlen=window_size)  # type: ignore

        # Set the frequency and start the data acquisition thread
        self.frequency = frequency
        self.stop_event = threading.Event()
        self.data_thread = threading.Thread(target=self._update_buffers)
        self.data_thread.start()

        time.sleep(1.0)

    def set_zero_pose(self):
        self.zero_pose = R.from_quat(np.array(self.sensor.quaternion))
        self.zero_pose_inv = self.zero_pose.inv()

    # @profile()
    def get_state(self):
        if self.zero_pose is None:
            self.set_zero_pose()

        # Compute moving averages
        avg_euler = np.mean(np.array(self.euler_history), axis=0)  # type: ignore
        rotation_relative = self.zero_pose_inv * R.from_euler("xyz", avg_euler)  # type: ignore
        quat_relative = list(rotation_relative.as_quat())  # type: ignore
        euler_relative = quaternion_to_euler_array(quat_relative, order="xyzw")  # type: ignore

        avg_angular_velocity = np.mean(np.array(self.angular_velocity_history), axis=0)  # type: ignore
        ang_vel_relative = np.array(
            self.zero_pose.apply(avg_angular_velocity),  # type: ignore
            dtype=np.float32,
        )

        state = {
            "imu_time": np.array(time.time(), dtype=np.float32),
            "imu_euler": euler_relative,
            "imu_ang_vel": ang_vel_relative,
        }

        return state

    def _update_buffers(self):
        while not self.stop_event.is_set():
            self.euler_history.append(  # type: ignore
                quaternion_to_euler_array(
                    np.array(self.sensor.quaternion), order="xyzw"
                )
            )
            self.angular_velocity_history.append(np.array(self.sensor.gyro))  # type: ignore
            time.sleep(1 / self.frequency)

    def close(self):
        self.stop_event.set()
        self.data_thread.join()


if __name__ == "__main__":
    # import copy

    imu = IMU()
    imu.set_zero_pose()

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
