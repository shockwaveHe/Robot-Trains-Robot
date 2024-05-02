import time

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
from transforms3d.quaternions import qinverse, qmult


class IMU:
    def __init__(self, default_pose=(1, 0, 0, 0)):
        # Initialize the I2C bus and sensor
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = BNO08X_I2C(self.i2c)

        # Enable the accelerometer, gyroscope, and rotation vector features
        self.sensor.enable_feature(BNO_REPORT_ACCELEROMETER)
        self.sensor.enable_feature(BNO_REPORT_GYROSCOPE)
        self.sensor.enable_feature(BNO_REPORT_ROTATION_VECTOR)

        self.default_pose = np.array(default_pose)
        self.default_pose_inv = qinverse(self.default_pose)

    def get_acceleration(self):
        """Returns the accelerometer data as a tuple (x, y, z)."""
        return np.array(self.sensor.acceleration)

    def get_angular_velocity(self):
        """Returns the gyroscope data as a tuple (x, y, z)."""
        return np.array(self.sensor.gyro)

    def get_quaternion(self):
        """Returns the quaternion data as a tuple (i, j, k, real)."""
        quat = self.sensor.quaternion
        quat_curr = np.array([quat[3], quat[0], quat[1], quat[2]])
        quat_relative = qmult(self.default_pose_inv, quat_curr)
        return quat_relative


if __name__ == "__main__":
    imu = IMU()
    while True:
        acceleration = imu.get_acceleration()
        angular_velocity = imu.get_angular_velocity()
        quaternion = imu.get_quaternion()
        print(
            f"Acceleration: {acceleration}, Angular Velocity: {angular_velocity}, Quaternion: {quaternion}"
        )
        time.sleep(0.1)
