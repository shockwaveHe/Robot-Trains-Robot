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
    euler2quat,
    exponential_moving_average,
    quat2euler,
    quat_inv,
    quat_mult,
    rotate_vec,
)


class IMU:
    def __init__(self, euler_alpha: float = 0.6, ang_vel_alpha: float = 0.6):
        self.euler_alpha = euler_alpha
        self.ang_vel_alpha = ang_vel_alpha

        # # Set up the reset pin
        # reset_pin = DigitalInOut(board.D4)
        # reset_pin.direction = Direction.OUTPUT
        # reset_pin.value = True

        # time.sleep(1)

        # Initialize the I2C bus and sensor
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = BNO08X_I2C(self.i2c)

        # Enable the gyroscope and rotation vector features
        self.sensor.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)
        self.sensor.enable_feature(BNO_REPORT_GYROSCOPE)
        self.sensor.enable_feature(BNO_REPORT_ROTATION_VECTOR)

        time.sleep(0.2)

        quat_raw = np.array(
            [self.sensor.quaternion[3], *self.sensor.quaternion[:3]],
            dtype=np.float32,
            copy=True,
        )
        euler_raw = np.asarray(quat2euler(quat_raw))
        zero_euler = np.round(euler_raw / (np.pi / 2)) * (np.pi / 2)
        self.zero_quat = np.asarray(euler2quat(zero_euler))
        self.zero_quat_inv = np.asarray(quat_inv(self.zero_quat))

        # print(f"euler_raw: {euler_raw}, quat_raw: {quat_raw}")
        # print(f"zero_euler: {zero_euler}, zero_quat: {zero_quat}")

        self.convert_matrix = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
        )
        # Initialize previous Euler angle for smoothing
        # self.time_last = time.time()
        # self.lin_vel_prev = np.zeros(3, dtype=np.float32)
        self.ang_vel_prev: npt.NDArray[np.float32] | None = None
        self.euler_prev: npt.NDArray[np.float32] | None = None

    def get_state(self) -> Dict[str, npt.NDArray[np.float32]]:
        quat_raw = np.array(
            [self.sensor.quaternion[3], *self.sensor.quaternion[:3]],
            dtype=np.float32,
            copy=True,
        )
        # Compute relative rotation based on zero pose
        quat = quat_mult(quat_raw, self.zero_quat_inv)
        euler = np.asarray(quat2euler(quat))
        if self.euler_prev is not None:
            euler_delta = (euler - self.euler_prev + np.pi) % (2 * np.pi) - np.pi
            euler = self.euler_prev + euler_delta

        filtered_euler = np.asarray(
            exponential_moving_average(self.euler_alpha, euler, self.euler_prev),
            dtype=np.float32,
        )
        self.euler_prev = filtered_euler

        ang_vel_raw = np.array(self.sensor.gyro, dtype=np.float32, copy=True)
        ang_vel_torso = np.asarray(rotate_vec(ang_vel_raw, self.zero_quat))
        ang_vel = np.asarray(rotate_vec(ang_vel_torso, quat_inv(quat)))

        # print(
        #     f"ang_vel_raw: {ang_vel_raw}, ang_vel_torso: {ang_vel_torso}, ang_vel: {ang_vel}, euler: {euler}"
        # )

        filtered_ang_vel = np.asarray(
            exponential_moving_average(self.ang_vel_alpha, ang_vel, self.ang_vel_prev),
            dtype=np.float32,
        )
        self.ang_vel_prev = filtered_ang_vel

        obs_euler = self.convert_matrix @ filtered_euler
        obs_ang_vel = self.convert_matrix @ filtered_ang_vel

        state = {"euler": obs_euler, "ang_vel": obs_ang_vel}

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
        # print(
        #     np.array(
        #         [imu.sensor.quaternion[3], *imu.sensor.quaternion[:3]],
        #         dtype=np.float32,
        #         copy=True,
        #     )
        # )
        print(f"ang_vel: {state['ang_vel']}, euler: {state['euler']} ")

        step_time = time.time() - step_start
        print(f"Step time: {step_time * 1000:.3f} ms")

        time.sleep(0.02)

        step += 1

    imu.close()
