import time

import board
import busio
import numpy as np
from adafruit_bno08x import (
    # BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    # BNO_REPORT_MAGNETOMETER,
    BNO_REPORT_ROTATION_VECTOR,
)
from adafruit_bno08x.i2c import BNO08X_I2C
from scipy.spatial.transform import Rotation as R


class IMU:
    def __init__(self):
        # Initialize the I2C bus and sensor
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = BNO08X_I2C(self.i2c)

        # Enable the accelerometer, gyroscope, and rotation vector features
        # self.sensor.enable_feature(BNO_REPORT_ACCELEROMETER)
        self.sensor.enable_feature(BNO_REPORT_GYROSCOPE)
        self.sensor.enable_feature(BNO_REPORT_ROTATION_VECTOR)

        # Warm up the sensor
        for _ in range(10):
            _ = self.sensor.quaternion
            _ = self.sensor.gyro
            time.sleep(0.1)

        self.default_pose = None
        self.default_pose_inv = None

    def set_default_pose(self):
        quat = self.sensor.quaternion
        self.default_pose = R.from_quat(quat)
        self.default_pose_inv = self.default_pose.inv()

    def get_acceleration(self):
        """Returns the accelerometer data as a tuple (x, y, z)."""
        accel = np.array(self.sensor.acceleration)
        # Transform acceleration to the default pose frame
        accel_relative = self.default_pose.apply(accel)
        return accel_relative

    def get_angular_velocity(self):
        """Returns the gyroscope data as a tuple (x, y, z)."""
        omega = np.array(self.sensor.gyro)
        # Transform angular velocity to the default pose frame
        omega_relative = self.default_pose.apply(omega)
        return omega_relative

    def get_quaternion(self):
        """Returns the quaternion data as a tuple (w, x, y, z)."""
        rotation_relative = self.default_pose_inv * R.from_quat(self.sensor.quaternion)
        quat_relative_xyzw = rotation_relative.as_quat()
        quat_relative_wxyz = np.array([quat_relative_xyzw[3], *quat_relative_xyzw[:3]])
        # Ensure the quaternion is in canonical form
        if quat_relative_wxyz[0] < 0:
            quat_relative_wxyz = -quat_relative_wxyz

        return quat_relative_wxyz

    def get_state(self):
        if self.default_pose is None:
            self.set_default_pose()

        state = {}

        # state["time"] = time.time()
        state["quaternion"] = self.get_quaternion()
        state["angular_velocity"] = self.get_angular_velocity()

        return state


if __name__ == "__main__":
    # import copy

    imu = IMU()
    time.sleep(1.0)

    # last_state = None
    while True:
        step_start = time.time()
        # acceleration = imu.get_acceleration()
        state = imu.get_state()

        print(f"omega: {state['angular_velocity']}, quat: {state['quaternion']}")

        # if last_state:
        #     euler_angles_delta = state["euler_angles"] - last_state["euler_angles"]
        #     time_delta = state["time"] - last_state["time"]
        #     ang_vel_fd = euler_angles_delta / time_delta
        #     print(f"omega_fd: {ang_vel_fd}")

        # last_state = copy.deepcopy(state)

        step_time = time.time() - step_start
        print(f"Step time: {step_time * 1000:.3f} ms")
        time.sleep(max(0.01 - step_time, 0))
