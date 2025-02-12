import time

import matplotlib.pyplot as plt
import numpy as np

from toddlerbot.sensing.IMU import IMU

# This script is for visualizing the IMU readings in real-time.

if __name__ == "__main__":
    # Initialize IMU
    imu = IMU()

    # Set up the figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("IMU Readings - Euler Angles and Angular Velocities")

    # Set up the plot for Euler angles
    ax1.set_title("Euler Angles (Radians)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Angle (rad)")
    (line_roll,) = ax1.plot([], [], label="Roll", color="r")
    (line_pitch,) = ax1.plot([], [], label="Pitch", color="g")
    (line_yaw,) = ax1.plot([], [], label="Yaw", color="b")
    ax1.legend()
    ax1.set_ylim(-np.pi, np.pi)

    # Set up the plot for Angular Velocity
    ax2.set_title("Angular Velocity (Rad/s)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angular Velocity (rad/s)")
    (line_ang_x,) = ax2.plot([], [], label="Ang Vel X", color="r")
    (line_ang_y,) = ax2.plot([], [], label="Ang Vel Y", color="g")
    (line_ang_z,) = ax2.plot([], [], label="Ang Vel Z", color="b")
    ax2.legend()
    ax2.set_ylim(-5, 5)  # Adjust this range as necessary

    # Initialize data storage
    euler_data = {"time": [], "roll": [], "pitch": [], "yaw": []}
    ang_vel_data = {"time": [], "x": [], "y": [], "z": []}
    start_time = time.time()

    def update_plot():
        # Get current state from IMU
        state = imu.get_state()
        print(f"Euler: {state['euler']}, Angular Vel: {state['ang_vel']}")

        current_time = time.time() - start_time

        # Store data
        euler_data["time"].append(current_time)
        euler_data["roll"].append(state["euler"][0])
        euler_data["pitch"].append(state["euler"][1])
        euler_data["yaw"].append(state["euler"][2])

        ang_vel_data["time"].append(current_time)
        ang_vel_data["x"].append(state["ang_vel"][0])
        ang_vel_data["y"].append(state["ang_vel"][1])
        ang_vel_data["z"].append(state["ang_vel"][2])

        # Update plot data
        line_roll.set_data(euler_data["time"], euler_data["roll"])
        line_pitch.set_data(euler_data["time"], euler_data["pitch"])
        line_yaw.set_data(euler_data["time"], euler_data["yaw"])

        line_ang_x.set_data(ang_vel_data["time"], ang_vel_data["x"])
        line_ang_y.set_data(ang_vel_data["time"], ang_vel_data["y"])
        line_ang_z.set_data(ang_vel_data["time"], ang_vel_data["z"])

        # Set limits to follow data
        for ax, data in zip([ax1, ax2], [euler_data, ang_vel_data]):
            ax.set_xlim(max(0, current_time - 10), current_time + 1)

        plt.pause(0.01)  # Small pause to update the plot

    try:
        while True:
            update_plot()

    except KeyboardInterrupt:
        print("Real-time plotting stopped.")
    finally:
        imu.close()
        plt.show()
