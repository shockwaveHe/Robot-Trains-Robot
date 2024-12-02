import numpy as np
import joblib
import matplotlib.pyplot as plt
import time

# Load data
data_dict = joblib.load("./datasets/xc330_test.lz4")
print(data_dict.keys())
data_dict["time"] = data_dict["time"] - data_dict["time"][0]
data_dict["velocity"] = data_dict["velocity"]/6.28*60.0

# Create figure and axes
fig, ax = plt.subplots(2, 2, figsize=(13, 10))

# Plot Torque
ax[0, 0].plot(data_dict["time"], data_dict["torque"])
ax[0, 0].set_title("Torque")
ax[0, 0].set_xlabel("Time (s)")
ax[0, 0].set_ylabel("Torque (Nm)")
ax[0, 0].grid()

# Plot Brake
ax[0, 1].plot(data_dict["time"], data_dict["brake"])
ax[0, 1].set_title("Brake")
ax[0, 1].set_xlabel("Time (s)")
ax[0, 1].set_ylabel("Brake Value")
ax[0, 1].grid()

# Plot Current
ax[1, 0].plot(data_dict["time"], data_dict["current"])
ax[1, 0].set_title("Current")
ax[1, 0].set_xlabel("Time (s)")
ax[1, 0].set_ylabel("Current (A)")
ax[1, 0].grid()

# Plot Velocity
ax[1, 1].plot(data_dict["time"], data_dict["velocity"])
ax[1, 1].set_title("Velocity")
ax[1, 1].set_xlabel("Time (s)")
ax[1, 1].set_ylabel("Velocity (rad/s)")
ax[1, 1].grid()

# Improve layout
plt.tight_layout()

fig2, ax2 = plt.subplots(2, 2, figsize=(13, 10))
ax2[0, 0].scatter(data_dict["current"], data_dict["torque"])
ax2[0, 0].set_title("Current vs Torque")
ax2[0, 0].set_xlabel("Current (mA)")
ax2[0, 0].set_ylabel("Torque (Nm)")
ax2[0, 0].grid()

ax2[0, 1].scatter(data_dict["velocity"], data_dict["torque"])
ax2[0, 1].set_title("Velocity vs Torque")
ax2[0, 1].set_xlabel("Velocity (RPM)")
ax2[0, 1].set_ylabel("Torque (Nm)")
ax2[0, 1].grid()

ax2[1, 0].scatter(data_dict["brake"], data_dict["torque"])
ax2[1, 0].set_title("Brake vs Torque")
ax2[1, 0].set_xlabel("Brake Value")
ax2[1, 0].set_ylabel("Torque (Nm)")
ax2[1, 0].grid()


# Improve layout
plt.tight_layout()

# Show plot
plt.show()

time.sleep(100)