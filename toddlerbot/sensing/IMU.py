import time

import board  # type: ignore
import busio  # type: ignore
import numpy as np
from adafruit_bno08x import (  # type: ignore
    BNO_REPORT_GYROSCOPE,  # type: ignore
    BNO_REPORT_LINEAR_ACCELERATION,  # type: ignore
    BNO_REPORT_ROTATION_VECTOR,  # type: ignore
)
from adafruit_bno08x.i2c import BNO08X_I2C  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore

from toddlerbot.utils.math_utils import exponential_moving_average


class IMU:
    def __init__(self, alpha: float = 0.1):
        # Initialize the I2C bus and sensor
        self.i2c = busio.I2C(board.SCL, board.SDA)  # type: ignore
        self.sensor = BNO08X_I2C(self.i2c)

        # Enable the gyroscope and rotation vector features
        self.sensor.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)  # type: ignore
        self.sensor.enable_feature(BNO_REPORT_GYROSCOPE)  # type: ignore
        self.sensor.enable_feature(BNO_REPORT_ROTATION_VECTOR)  # type: ignore

        time.sleep(0.2)

        self.zero_pose = None
        self.zero_pose_inv = None

        self.alpha = alpha

        # Initialize previous Euler angle for smoothing
        self.time_last = time.time()
        self.euler_prev = np.zeros(3, dtype=np.float32)
        self.ang_vel_prev = np.zeros(3, dtype=np.float32)

    def set_zero_pose(self):
        self.zero_pose = R.from_quat(np.array(self.sensor.quaternion))
        self.zero_pose_inv = self.zero_pose.inv()

    def get_state(self):
        if self.zero_pose is None:
            self.set_zero_pose()

        assert self.zero_pose_inv is not None

        time_curr = time.time()
        lin_acc = np.array(self.sensor.acceleration)
        lin_acc_relative = self.zero_pose.apply(lin_acc).astype(np.float32)  # type: ignore
        lin_vel_relative = lin_acc_relative * (time_curr - self.time_last)
        self.time_last = time_curr
        filtered_lin_vel = exponential_moving_average(
            self.alpha, lin_vel_relative, self.lin_vel_prev
        )
        self.lin_vel_prev = filtered_lin_vel

        ang_vel = np.array(self.sensor.gyro)
        ang_vel_relative = self.zero_pose.apply(ang_vel).astype(np.float32)  # type: ignore
        filtered_ang_vel = exponential_moving_average(
            self.alpha, ang_vel_relative, self.ang_vel_prev
        )
        self.ang_vel_prev = filtered_ang_vel

        # Compute relative rotation based on zero pose
        rotation_relative = (
            R.from_quat(np.array(self.sensor.quaternion)) * self.zero_pose_inv
        )
        euler_relative = rotation_relative.as_euler("xyz").astype(np.float32)  # type: ignore
        # Ensure the transition is smooth by adjusting for any discontinuities
        delta = euler_relative - self.euler_prev
        delta = np.where(delta > np.pi, delta - 2 * np.pi, delta)  # type: ignore
        delta = np.where(delta < -np.pi, delta + 2 * np.pi, delta)  # type: ignore
        euler_relative = self.euler_prev + delta

        filtered_euler = exponential_moving_average(
            self.alpha, euler_relative, self.euler_prev
        )
        self.euler_prev = filtered_euler

        state = {
            "lin_vel": filtered_lin_vel,
            "ang_vel": filtered_ang_vel,
            "euler": filtered_euler,
        }

        return state

    def close(self):
        pass


if __name__ == "__main__":
    # import copy

    imu = IMU()

    step = 0
    while step < 1000000:  # True:
        step_start = time.time()
        # acceleration = imu.get_acceleration()
        state = imu.get_state()
        print(f"euler: {state['euler']}, ang_vel: {state['ang_vel']}")

        step_time = time.time() - step_start
        print(f"Step time: {step_time * 1000:.3f} ms")

        time.sleep(0.01)

        step += 1

    imu.close()
