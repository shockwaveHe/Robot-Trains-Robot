import time
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
from toddlerbot.utils.ft_utils import NetFTSensor

# Pendulum and desired force parameters
m = 3.3
g = 9.81
theta_0 = np.pi / 6  # maximum angle (in radians)
L = 0.6


def desired_force_x(t):
    # Note: This formulation uses small-angle assumptions.
    # For large angles, you might use a more complex model.
    A = m * g * theta_0
    omega = np.sqrt(g / L)
    phi = np.pi / 2
    return A * np.sin(omega * t + phi)


# Setup sensor and plot parameters
sensor = NetFTSensor()
window_size = 100  # number of points to show on the plot
interval_ms = 100  # plot update interval in milliseconds
dt = interval_ms / 1000.0  # time step in seconds

# Deques for measured force/torque data storage
force_data = {
    "x": deque([0.0] * window_size, maxlen=window_size),
    "y": deque([0.0] * window_size, maxlen=window_size),
    "z": deque([0.0] * window_size, maxlen=window_size),
}
torque_data = {
    "x": deque([0.0] * window_size, maxlen=window_size),
    "y": deque([0.0] * window_size, maxlen=window_size),
    "z": deque([0.0] * window_size, maxlen=window_size),
}

# Create figure and axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
# Lines for measured force (each axis)
lines_force = {k: ax1.plot([], [], label=f"Force {k}")[0] for k in force_data}
# Lines for measured torque (each axis)
lines_torque = {k: ax2.plot([], [], label=f"Torque {k}")[0] for k in torque_data}
# Line for desired force (only for x)
(line_desired,) = ax1.plot([], [], label="Desired Force x", color="red", linestyle="--")

# Setup axes limits and grid
for ax in (ax1, ax2):
    ax.set_xlim(0, window_size)
    ax.set_ylim(-40, 40)  # Adjust as needed based on your expected range
    ax.legend()
    ax.grid(True)

# Record the start time for our time reference
start_time = time.time()


def update_plot(frame):
    try:
        # Retrieve the latest sensor data
        force, torque = sensor.get_smoothed_data()
        print(f"Force: {force}, Torque: {torque}")

        # Update measured force and torque deques
        for i, axis in enumerate(["x", "y", "z"]):
            force_data[axis].append(force[i])
            torque_data[axis].append(torque[i])
            lines_force[axis].set_ydata(force_data[axis])
            lines_torque[axis].set_ydata(torque_data[axis])
            # x-data is simply the index (0 to window_size-1)
            lines_force[axis].set_xdata(range(len(force_data[axis])))
            lines_torque[axis].set_xdata(range(len(torque_data[axis])))
    except Exception as e:
        print(f"Error during update: {e}")

    # Compute the desired force curve for the x-axis over the current window.
    # We'll assume each index corresponds to dt seconds.
    current_time = time.time() - start_time
    # Compute time values for the window. For example, if we have 100 points at dt interval:
    t_values = np.linspace(current_time - window_size * dt, current_time, window_size)
    desired_values = [desired_force_x(t) for t in t_values]
    line_desired.set_xdata(range(window_size))
    line_desired.set_ydata(desired_values)

    # Return all updated line objects for blitting
    return list(lines_force.values()) + list(lines_torque.values()) + [line_desired]


# Create the animation
ani = animation.FuncAnimation(fig, update_plot, interval=interval_ms, blit=True)

try:
    plt.tight_layout()
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    sensor.stop()
    print("Stopped plotting and sensor.")
