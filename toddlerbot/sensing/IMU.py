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

from toddlerbot.utils.math_utils import (
    exponential_moving_average,
    quat2euler,
    quat_inv,
    quat_mult,
    rotate_vec,
)


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

        self.zero_quat = None

        self.alpha = alpha

        # Initialize previous Euler angle for smoothing
        self.time_last = time.time()
        self.lin_vel_prev = np.zeros(3, dtype=np.float32)
        self.ang_vel_prev = np.zeros(3, dtype=np.float32)
        self.euler_prev = np.zeros(3, dtype=np.float32)

    def set_zero_pose(self):
        self.zero_quat = np.array(self.sensor.quaternion, dtype=np.float32, copy=True)  # type: ignore

    def get_state(self):
        if self.zero_quat is None:
            self.set_zero_pose()

        assert self.zero_quat is not None

        quat_raw = np.array(self.sensor.quaternion, dtype=np.float32, copy=True)  # type: ignore
        # Compute relative rotation based on zero pose
        quat = quat_mult(quat_raw, quat_inv(self.zero_quat))
        euler = np.asarray(quat2euler(quat))  # type: ignore
        # Ensure the transition is smooth by adjusting for any discontinuities
        delta = euler - self.euler_prev
        delta = np.where(delta > np.pi, delta - 2 * np.pi, delta)  # type: ignore
        delta = np.where(delta < -np.pi, delta + 2 * np.pi, delta)  # type: ignore
        euler = self.euler_prev + delta

        filtered_euler = np.asarray(
            exponential_moving_average(self.alpha, euler, self.euler_prev),
            dtype=np.float32,
        )
        self.euler_prev = filtered_euler

        time_curr = time.time()
        lin_acc_raw = np.array(
            self.sensor.linear_acceleration, dtype=np.float32, copy=True
        )
        lin_acc_global = np.asarray(rotate_vec(lin_acc_raw, self.zero_quat))
        lin_acc = np.asarray(rotate_vec(lin_acc_global, quat_inv(quat)))
        lin_vel = self.lin_vel_prev + lin_acc * (time_curr - self.time_last)
        self.time_last = time_curr
        self.lin_vel_prev = lin_vel

        ang_vel_raw = np.array(self.sensor.gyro, dtype=np.float32, copy=True)
        ang_vel_global = np.asarray(rotate_vec(ang_vel_raw, self.zero_quat))
        ang_vel = np.asarray(rotate_vec(ang_vel_global, quat_inv(quat)))
        filtered_ang_vel = np.asarray(
            exponential_moving_average(self.alpha, ang_vel, self.ang_vel_prev),
            dtype=np.float32,
        )
        self.ang_vel_prev = filtered_ang_vel

        state = {
            "lin_vel": lin_vel,
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
