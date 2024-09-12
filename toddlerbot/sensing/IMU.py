import time
from typing import Dict

import board
import busio
import numpy as np
import numpy.typing as npt
from adafruit_bno08x import (
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_LINEAR_ACCELERATION,
    BNO_REPORT_ROTATION_VECTOR,
)
from adafruit_bno08x.i2c import BNO08X_I2C

from toddlerbot.utils.math_utils import (
    exponential_moving_average,
    quat2euler,
    quat_inv,
    quat_mult,
    rotate_vec,
)


class IMU:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

        # Initialize the I2C bus and sensor
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = BNO08X_I2C(self.i2c)

        # Enable the gyroscope and rotation vector features
        self.sensor.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)
        self.sensor.enable_feature(BNO_REPORT_GYROSCOPE)
        self.sensor.enable_feature(BNO_REPORT_ROTATION_VECTOR)

        time.sleep(0.2)

        self.zero_quat = None

        # Initialize previous Euler angle for smoothing
        # self.time_last = time.time()
        # self.lin_vel_prev = np.zeros(3, dtype=np.float32)
        self.ang_vel_prev = np.zeros(3, dtype=np.float32)
        self.euler_prev = np.zeros(3, dtype=np.float32)

    def set_zero_pose(self):
        self.zero_quat = np.array(self.sensor.quaternion, dtype=np.float32, copy=True)
        self.zero_quat_inv = np.asarray(quat_inv(self.zero_quat))

    def get_state(self) -> Dict[str, npt.NDArray[np.float32]]:
        if self.zero_quat is None:
            self.set_zero_pose()

        assert self.zero_quat is not None
        assert self.zero_quat_inv is not None

        quat_raw = np.array(self.sensor.quaternion, dtype=np.float32, copy=True)
        # Compute relative rotation based on zero pose
        quat = quat_mult(quat_raw, self.zero_quat_inv)
        euler = np.asarray(quat2euler(quat))
        # Ensure the transition is smooth by adjusting for any discontinuities
        euler_delta = euler - self.euler_prev
        euler_delta = (euler_delta + np.pi) % (2 * np.pi) - np.pi
        euler = self.euler_prev + euler_delta

        filtered_euler = np.asarray(
            exponential_moving_average(self.alpha, euler, self.euler_prev),
            dtype=np.float32,
        )
        self.euler_prev = filtered_euler

        # time_curr = time.time()
        # lin_acc_raw = np.array(
        #     self.sensor.linear_acceleration, dtype=np.float32, copy=True
        # )
        # lin_acc_global = np.asarray(rotate_vec(lin_acc_raw, self.zero_quat_inv))
        # lin_acc = np.asarray(rotate_vec(lin_acc_global, quat_inv(quat)))
        # lin_vel = self.lin_vel_prev + lin_acc * (time_curr - self.time_last)
        # self.lin_vel_prev = lin_vel
        # self.time_last = time_curr

        ang_vel_raw = np.array(self.sensor.gyro, dtype=np.float32, copy=True)
        ang_vel_global = np.asarray(rotate_vec(ang_vel_raw, self.zero_quat_inv))
        ang_vel = np.asarray(rotate_vec(ang_vel_global, quat_inv(quat)))
        filtered_ang_vel = np.asarray(
            exponential_moving_average(self.alpha, ang_vel, self.ang_vel_prev),
            dtype=np.float32,
        )
        self.ang_vel_prev = filtered_ang_vel

        state = {
            # "lin_vel": lin_acc,
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
        print(f"ang_vel: {state['ang_vel']}, euler: {state['euler']} ")

        step_time = time.time() - step_start
        print(f"Step time: {step_time * 1000:.3f} ms")

        time.sleep(0.01)

        step += 1

    imu.close()
