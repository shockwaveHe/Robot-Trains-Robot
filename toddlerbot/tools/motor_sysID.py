import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import serial
from sklearn.linear_model import LinearRegression, RANSACRegressor

from toddlerbot.actuation.dynamixel_control import (
    DynamixelConfig,
    DynamixelController,
)
from toddlerbot.utils.file_utils import find_ports
# from toddlerbot.visualization.vis_plot import plot_loop_time


@dataclass
class SysIDData:
    time: float
    torque: float
    brake: int
    pos: float
    vel: float
    cur: float


@dataclass
class SysIDProfile:
    time: float
    current: float
    brake: int


def fit_tor_cur(ax, x, y, color, line_label="", cur_min=50):
    # Ensure x is 2D (n_samples, 1) for scikit-learn.
    x = np.array(x)
    y = np.array(y)

    # Filter out data points with current values below the threshold.
    mask = x > cur_min
    x = x[mask]
    y = y[mask]

    x = x.reshape(-1, 1)

    # Use RANSAC for robust linear regression.
    model = RANSACRegressor(LinearRegression())
    model.fit(x, y)

    # Retrieve the underlying estimator's parameters.
    slope = round(model.estimator_.coef_[0].item(), 6)
    intercept = round(model.estimator_.intercept_.item(), 6)

    # Determine the endpoints for plotting the regression line.
    x_min, x_max = np.min(x), np.max(x)
    y_start = slope * x_min + intercept
    y_end = slope * x_max + intercept

    ax.plot(
        [x_min, x_max], [y_start, y_end], color=color, linewidth=2, label=line_label
    )
    return slope, intercept


def fit_tor_vel(ax, x, y, color, line_label="", degree=3):
    # Define a generalized piecewise function.
    def piecewise_poly(x, x_break, y_offset, *coeffs):
        # For x >= x_break, evaluate the polynomial:
        #   poly(x - x_break) = coeffs[0]*(x-x_break)^1 + coeffs[1]*(x-x_break)^2 + ...
        poly_val = np.zeros_like(x, dtype=float)
        for i, c in enumerate(coeffs):
            poly_val += c * (x - x_break) ** (i + 1)
        # Return constant for x < x_break, and the polynomial shift for x >= x_break.
        return np.where(x < x_break, y_offset, y_offset + poly_val)

    x = np.array(x)
    y = np.array(y)

    mask = np.logical_and(y > y[np.argmax(x)].item(), x > 0)
    x = x[mask]
    y = y[mask]

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Determine data range.
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    p0 = [x_min, y_max, (y_min - y_max) / (x_max - x_min)] + [0] * (degree - 1)

    lower_bounds = [x_min, y_min] + [-1.0] * degree
    upper_bounds = [x_max, y_max] + [1.0] * degree
    bounds = (lower_bounds, upper_bounds)

    # Fit the piecewise polynomial model.
    params, params_covariance = curve_fit(piecewise_poly, x, y, p0=p0, bounds=bounds)
    # Extract fitted parameters.
    x_break_fit = round(params[0], 6)
    y_offset_fit = round(params[1], 6)

    coeffs = [round(p, 6) for p in params[2:] if round(p, 6) != 0]
    # Plot the fitted function.
    y_fit = piecewise_poly(x, x_break_fit, y_offset_fit, *coeffs)
    ax.plot(x, y_fit, color=color, linewidth=2, label=line_label)

    # Define the endpoint as the prediction at x_max.
    end_y = piecewise_poly(np.array([x_max]), x_break_fit, y_offset_fit, *coeffs)[0]
    end_point = [round(x_max, 6), round(end_y, 6)]

    return (x_break_fit, y_offset_fit, end_point, coeffs)


def plot_data(
    data_list: List[SysIDData],
    save_path: str = "",
    color="blue",
    label="",
    axes=None,
):
    # Create a 3x3 grid of subplots
    if axes is None:
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.delaxes(axes[2, 2])
        # Create a new 3D subplot in the same grid location
        axes[2, 2] = fig.add_subplot(3, 3, 9, projection="3d")
    else:
        fig = plt.gcf()

    time_arr = np.array([item.time for item in data_list])
    torque_arr = np.array([item.torque for item in data_list])
    brake_arr = np.array([item.brake for item in data_list])
    pos_arr = np.array([item.pos for item in data_list])
    vel_arr = np.array([item.vel for item in data_list])
    cur_arr = np.array([item.cur for item in data_list])

    # Plot 1: Time vs. Torque
    ax = axes[0, 0]
    ax.scatter(
        time_arr, torque_arr, s=30, color=color, alpha=0.7, edgecolors="k", label=label
    )
    ax.set_title("Time vs. Torque")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque")
    ax.grid(True)

    # Plot 2: Time vs. Brake
    ax = axes[0, 1]
    ax.scatter(
        time_arr, brake_arr, s=30, color=color, alpha=0.7, edgecolors="k", label=label
    )
    ax.set_title("Time vs. Brake")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Brake")
    ax.grid(True)

    # Plot 3: Time vs. Position
    ax = axes[0, 2]
    ax.scatter(
        time_arr, pos_arr, s=30, color=color, alpha=0.7, edgecolors="k", label=label
    )
    ax.set_title("Time vs. Position")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position")
    ax.grid(True)

    # Plot 4: Time vs. Velocity
    ax = axes[1, 0]
    ax.scatter(
        time_arr, vel_arr, s=30, color=color, alpha=0.7, edgecolors="k", label=label
    )
    ax.set_title("Time vs. Velocity")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity")
    ax.grid(True)

    # Plot 5: Time vs. Current
    ax = axes[1, 1]
    ax.scatter(
        time_arr, cur_arr, s=30, color=color, alpha=0.7, edgecolors="k", label=label
    )
    ax.set_title("Time vs. Current")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current")
    ax.grid(True)

    # Plot 6: Torque vs. Velocity
    ax = axes[1, 2]
    ax.scatter(
        vel_arr, torque_arr, s=30, color=color, alpha=0.7, edgecolors="k", label=label
    )
    ax.set_title("Torque vs. Velocity")
    ax.set_xlabel("Velocity")
    ax.set_ylabel("Torque")
    ax.grid(True)
    ax.grid(True)

    # Plot 7: Velocity vs. Current
    ax = axes[2, 0]
    ax.scatter(
        cur_arr, vel_arr, s=30, color=color, alpha=0.7, edgecolors="k", label=label
    )
    ax.set_title("Velocity vs. Current")
    ax.set_xlabel("Current")
    ax.set_ylabel("Velocity")
    ax.grid(True)

    # Plot 8: Torque vs. Current
    ax = axes[2, 1]
    ax.scatter(
        cur_arr, torque_arr, s=30, color=color, alpha=0.7, edgecolors="k", label=label
    )
    ax.set_title("Torque vs. Current")
    ax.set_xlabel("Current")
    ax.set_ylabel("Torque")
    ax.grid(True)

    # Create the 3D scatter plot using torque_arr, vel_arr, and cur_arr
    ax = axes[2, 2]
    ax.scatter(
        torque_arr,
        vel_arr,
        cur_arr,
        s=30,
        color=color,
        alpha=0.7,
        edgecolors="k",
        label=label,
    )
    ax.set_title("3D Scatter: Torque vs. Velocity vs. Current")
    ax.set_xlabel("Torque")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("Current")

    # Adjust layout so titles and labels do not overlap.
    fig.suptitle("Comprehensive Data Plots", fontsize=16)
    fig.tight_layout()

    # Save the figure.
    if len(save_path) > 0:
        plt.savefig(os.path.join(save_path, "combined_plots.png"))
        plt.close(fig)


class ArduinoController:
    def __init__(self, baud_rate=115200, timeout=1):
        """
        Initialize the serial connection to the Arduino.

        :param port: Serial port where Arduino is connected (e.g., 'COM3' or '/dev/ttyUSB0')
        :param baud_rate: Baud rate for serial communication
        :param timeout: Timeout for serial read operations
        """
        self.port = find_ports("USB Serial")[0]
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial_connection = serial.Serial(self.port, baud_rate, timeout=timeout)
        time.sleep(2)  # Allow time for Arduino to reset
        self.serial_connection.reset_input_buffer()

    def set_brake_value(self, value):
        """
        Send a brake value to the Arduino to control the brake (0-255).

        :param value: Brake value between 1 and 255
        """
        if 1 <= value <= 255:
            command = f"{value}\n"  # Newline indicates end of input
            self.serial_connection.write(command.encode("utf-8"))
            self.serial_connection.flush()
            # print(f"Sent brake value: {value}")
        else:
            raise ValueError("Brake value must be between 1 and 255.")

    def read_sensor(self):
        """
        Read the weight value from the Arduino and return the most recent value.

        :return: The most recent weight value read from the Arduino, or None if no valid value is available.
        """
        factor = 461300.0 / 0.7725375
        latest_value = None

        # Continue reading while there is data available.
        while self.serial_connection.in_waiting > 0:
            line = self.serial_connection.readline().decode("utf-8").strip()
            try:
                latest_value = float(line)
            except ValueError:
                print(f"Error: Could not parse sensor value from '{line}'.")
                # Continue reading to try and get a valid value.

        if latest_value is not None:
            return latest_value / factor

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


def run(
    motor_name,
    motor_id,
    ard_controller,
    dyn_controller,
    command_max,
    save_path,
    control_dt=0.05,
    idle_time=2.0,
    test_time=15.0,
    cur_ratio=1.0,
):
    time_str = time.strftime("%Y%m%d_%H%M%S")

    _, v_in_arr = dyn_controller.client.read_vin()
    _, temp_arr = dyn_controller.client.read_temp()
    v_in = round(v_in_arr.item(), 1)
    temp = temp_arr.item()
    print(f"Voltage: {v_in:.2f} V, Temperature: {temp:.2f} C")

    with open(os.path.join(save_path, "stats.json"), "w") as f:
        json.dump(
            {
                "time": time_str,
                "motor": motor_name,
                "idle_time": idle_time,
                "test_time": test_time,
                "cur_ratio": cur_ratio,
                "voltage": v_in,
                "temperature": temp,
            },
            f,
        )

    cur_max, brake_min, brake_max = command_max
    cur_command = cur_max * cur_ratio
    brake_command = (brake_max - brake_min) * cur_ratio + brake_min
    profile_list = [
        SysIDProfile(0.0, cur_command, 1),
        SysIDProfile(idle_time, cur_command, 1),
        SysIDProfile(idle_time + test_time, cur_command, brake_command),
        SysIDProfile(idle_time + test_time + 0.5, 0, 1),
    ]

    start_time = time.time()

    loop_time_list = []
    data_list = []
    step_idx = 0
    last_brake_value = 1
    try:
        while True:
            # Read weight from Arduino
            step_start = time.time()

            if step_start - start_time > profile_list[-1].time:
                print("Test finished.")
                break

            raw_torque_reading = ard_controller.read_sensor()
            t_sensor = time.time()
            if raw_torque_reading is None:
                print("Error: Could not read torque sensor value.")
                continue

            print(f"Current torque: {raw_torque_reading:.3f}")

            motor_state_curr = dyn_controller.get_motor_state()
            t_dyn = time.time()

            # print(motor_state_curr)
            data_list.append(
                SysIDData(
                    t_dyn - start_time,
                    raw_torque_reading,
                    last_brake_value,
                    motor_state_curr[motor_id].pos,
                    motor_state_curr[motor_id].vel,
                    motor_state_curr[motor_id].cur,
                )
            )

            t = time.time() - start_time
            for i in range(len(profile_list) - 1):
                if profile_list[i].time <= t < profile_list[i + 1].time:
                    # Calculate interpolation factor
                    factor = (t - profile_list[i].time) / (
                        profile_list[i + 1].time - profile_list[i].time
                    )

                    # Compute the interpolated commands directly
                    cur_cmd = profile_list[i].current + factor * (
                        profile_list[i + 1].current - profile_list[i].current
                    )
                    brake_cmd = profile_list[i].brake + factor * (
                        profile_list[i + 1].brake - profile_list[i].brake
                    )

                    # Set the current and brake values
                    dyn_controller.set_cur([cur_cmd])
                    ard_controller.set_brake_value(brake_cmd)
                    last_brake_value = brake_cmd

                    break

            step_idx += 1
            step_end = time.time()

            loop_time_list.append(
                {
                    "sensor_time": t_sensor - step_start,
                    "dyn_time": t_dyn - t_sensor,
                    "cmd_time": step_end - t_dyn,
                }
            )

            time_until_next_step = start_time + control_dt * step_idx - step_end
            # print(f"time_until_next_step: {time_until_next_step * 1000:.2f} ms")
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    except KeyboardInterrupt:
        pass

    finally:
        # loop_time_dict = {
        #     "sensor_time": [item["sensor_time"] * 1000 for item in loop_time_list],
        #     "dyn_time": [item["dyn_time"] * 1000 for item in loop_time_list],
        #     "cmd_time": [item["cmd_time"] * 1000 for item in loop_time_list],
        # }
        # plot_loop_time(loop_time_dict, save_path)
        plot_data(data_list, save_path)

        joblib.dump(data_list, os.path.join(save_path, "data.lz4"), compress="lz4")


def collect_data(motor_name, command_max, result_path, motor_id=1):
    ard_controller = ArduinoController()

    # Create a controller object
    ports = find_ports("USB <-> Serial Converter")
    config = DynamixelConfig(
        port=ports[0],
        baudrate=2000000,
        control_mode=["current"],
        kP=[0.0],
        kI=[0.0],
        kD=[0.0],
        kFF2=[0.0],
        kFF1=[0.0],
        init_pos=[0.0],
    )
    dyn_controller = DynamixelController(config, motor_ids=[motor_id])

    num_cur_levels = 10
    for i in range(num_cur_levels):
        num_repeat = 10 if i == 0 else 5
        for j in range(num_repeat):
            dyn_controller.enable_motors()

            cur_ratio = 1 - i / num_cur_levels
            save_path = os.path.join(result_path, f"{cur_ratio:.1f}_{j}")
            os.makedirs(save_path, exist_ok=True)

            run(
                motor_name,
                motor_id,
                ard_controller,
                dyn_controller,
                command_max,
                save_path,
                cur_ratio=cur_ratio,
            )

            dyn_controller.reboot_motors()
            time.sleep(1)

    ard_controller.close_connection()
    dyn_controller.close_motors()


def evaluate(motor_name, run_name):
    data_path = os.path.join("results", f"{motor_name}_sysID_{run_name}")

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.delaxes(axes[2, 2])
    # Create a new 3D subplot in the same grid location
    axes[2, 2] = fig.add_subplot(3, 3, 9, projection="3d")

    dir_list = os.listdir(data_path)
    cmap = plt.get_cmap("hsv", len(dir_list))

    i = 0
    while i < len(dir_list):
        if dir_list[i].startswith("0") or dir_list[i].startswith("1"):
            i += 1
        else:
            del dir_list[i]

    for i, dir_name in enumerate(dir_list):
        result_path = os.path.join(data_path, dir_name)
        data_list = joblib.load(os.path.join(result_path, "data.lz4"))
        stats = json.load(open(os.path.join(result_path, "stats.json"), "r"))
        plot_data(
            data_list,
            color=cmap(i),
            label=dir_name + f"_v={stats['voltage']:.1f}_t={stats['temperature']:.1f}",
            axes=axes,
        )

    dynamics_params_path = os.path.join(
        "toddlerbot", "descriptions", f"sysID_{motor_name}", "config_dynamics.json"
    )
    dynamics_params = json.load(open(dynamics_params_path, "r"))["joint_0"]
    tau_max = dynamics_params["tau_max"]
    q_dot_tau_max = dynamics_params["q_dot_tau_max"]
    q_dot_max = dynamics_params["q_dot_max"]

    # Simulate joint data
    q_dot = np.linspace(0, q_dot_max, 1000)

    abs_q_dot = np.abs(q_dot)

    # Apply vectorized conditions using np.where
    tau_limit = np.where(
        abs_q_dot <= q_dot_tau_max,  # Condition 1
        tau_max,  # Value when condition 1 is True
        np.where(
            abs_q_dot <= q_dot_max,  # Condition 2
            tau_max / (q_dot_tau_max - q_dot_max) * (abs_q_dot - q_dot_tau_max)
            + tau_max,  # Value when condition 2 is True
            0.0,  # Value when all conditions are False
        ),
    )

    axes[1, 2].plot(
        q_dot,
        tau_limit,
        color="royalblue",
        linewidth=3,
    )

    regression_params = {}
    # Retrieve data from scatter plots
    cur_data = []
    tor_data = []
    for collection in axes[2, 1].collections:
        # get_offsets() returns an Nx2 array of [x, y] points
        offsets = collection.get_offsets()
        cur_data.append(offsets[:, 0])
        tor_data.append(offsets[:, 1])

    cur_data = np.concatenate(cur_data)
    tor_data = np.concatenate(tor_data)
    slope, intercept = fit_tor_cur(axes[2, 1], cur_data, tor_data, "royalblue")

    regression_params["tor_vs_cur"] = {"slope": slope, "intercept": intercept}

    vel_data = []
    tor_data = []
    for collection in axes[1, 2].collections:
        # get_offsets() returns an Nx2 array of [x, y] points
        if "1.0_" in collection.get_label():
            offsets = collection.get_offsets()
            vel_data.append(offsets[:, 0])
            tor_data.append(offsets[:, 1])

    vel_data = np.concatenate(vel_data)
    tor_data = np.concatenate(tor_data)
    x_break_fit, y_offset_fit, end_point, coeffs = fit_tor_vel(
        axes[1, 2], vel_data, tor_data, "royalblue"
    )

    regression_params["tor_vs_vel"] = {
        "x_break": x_break_fit,
        "y_offset": y_offset_fit,
        "end_point": end_point,
        "coeffs": coeffs,
    }

    # Save regression parameters to a JSON file.
    with open(os.path.join(data_path, "regression_params.json"), "w") as f:
        json.dump(regression_params, f, indent=4)

    # plt.legend()

    plt.savefig(os.path.join(data_path, "combined_plots.png"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the motor sysID.")
    parser.add_argument(
        "--motor",
        type=str,
        default="XC330",
        help="The name of the motor",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="The run name to evaluate",
    )
    args = parser.parse_args()

    command_max = {
        "XC330": (910, 120, 220),
        "XM430": (3210, 60, 160),
    }

    if len(args.run_name) > 0:
        evaluate(args.motor, args.run_name)
    else:
        time_str = time.strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join("results", f"{args.motor}_sysID_{time_str}")
        collect_data(args.motor, command_max[args.motor], result_path)
        evaluate(args.motor, time_str)
