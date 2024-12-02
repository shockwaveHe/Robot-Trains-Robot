import matplotlib.pyplot as plt
import numpy as np


def test_torso_pitch():
    # Define torso_pitch_range based on your code
    torso_pitch_range = [-0.1, 0.1]  # Example range; adjust based on your actual values

    # Define a range of torso_pitch values to plot
    torso_pitch_values = np.linspace(-0.5, 0.5, 100)  # Change the range as needed
    rewards = []

    # Calculate reward for each torso_pitch value
    for torso_pitch in torso_pitch_values:
        pitch_min = np.clip(torso_pitch - torso_pitch_range[0], a_min=None, a_max=0.0)
        pitch_max = np.clip(torso_pitch - torso_pitch_range[1], a_min=0.0, a_max=None)
        reward = (
            np.exp(-np.abs(pitch_min) * 100) + np.exp(-np.abs(pitch_max) * 100)
        ) / 2
        rewards.append(reward)

    # Plot the function
    plt.plot(torso_pitch_values, rewards)
    plt.xlabel("Torso Pitch")
    plt.ylabel("Reward")
    plt.title("Reward vs. Torso Pitch")
    plt.grid(True)
    plt.show()


def test_feet_air_time():
    # Define a range of torso_pitch values to plot
    feet_air_time_values = np.linspace(0, 0.72, 100)  # Change the range as needed

    # Reward air time.
    max_feet_air_time = 0.36
    rewards = []

    # Calculate reward for each torso_pitch value
    for feet_air_time in feet_air_time_values:
        reward = -100 * (feet_air_time**2 - 2 * feet_air_time * max_feet_air_time)
        rewards.append(reward)

    # Plot the function
    plt.plot(feet_air_time_values, rewards)
    plt.xlabel("Feet Air Time")
    plt.ylabel("Reward")
    plt.title("Reward vs. Feet Air Time")
    plt.grid(True)
    plt.show()


def test_lin_vel_xy():
    """Reward for track linear velocity in xy"""

    # Define a range of torso_pitch values to plot
    lin_vel_x_values = np.linspace(-0.4, 0.4, 20)  # Change the range as needed
    lin_vel_y_values = np.linspace(-0.4, 0.4, 20)  # Change the range as needed

    lin_vel_xy_ref = np.array([0.1, 0.0])  # Reference linear velocity in xy
    tracking_sigma = 100.0  # Tracking sigma value

    # Calculate rewards for each combination of lin_vel_x and lin_vel_y
    rewards = np.zeros((len(lin_vel_x_values), len(lin_vel_y_values)))
    for i, lin_vel_x in enumerate(lin_vel_x_values):
        for j, lin_vel_y in enumerate(lin_vel_y_values):
            lin_vel_xy = np.array([lin_vel_x, lin_vel_y])
            error = np.linalg.norm(lin_vel_xy - lin_vel_xy_ref, axis=-1)
            rewards[j, i] = np.exp(-tracking_sigma * error**2)

    # Plot the reward surface with contours and labels
    plt.figure(figsize=(8, 6))
    contour = plt.contour(
        lin_vel_x_values,
        lin_vel_y_values,
        rewards,
        levels=10,
        colors="black",
        linewidths=0.8,
    )
    plt.contourf(lin_vel_x_values, lin_vel_y_values, rewards, levels=50, cmap="viridis")
    plt.colorbar(label="Reward")
    plt.clabel(
        contour, inline=True, fontsize=8, fmt="%.2f"
    )  # Add labels to the contours
    plt.scatter(
        lin_vel_xy_ref[0], lin_vel_xy_ref[1], color="red", label="Reference", zorder=5
    )
    plt.xlabel("Linear Velocity X")
    plt.ylabel("Linear Velocity Y")
    plt.title("Reward vs. Linear Velocity in XY")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def test_track_vel_hard():
    """Test and visualize the reward for tracking linear and angular velocities in xy."""

    # Define a range of lin_vel_x and lin_vel_y values for testing
    lin_vel_x_values = np.linspace(-0.4, 0.4, 20)  # Adjust the range if needed
    lin_vel_y_values = np.linspace(-0.4, 0.4, 20)  # Adjust the range if needed

    # Define reference linear and angular velocities
    lin_vel_xy_ref = np.array([0.1, 0.0])  # Linear velocity reference
    ang_vel_xy_ref = np.array([0.0, 0.0])  # Angular velocity reference

    # Tracking sigma for both linear and angular velocities
    tracking_sigma_lin = 10.0
    tracking_sigma_ang = 10.0

    # Calculate rewards for each combination of lin_vel_x and lin_vel_y
    rewards = np.zeros((len(lin_vel_x_values), len(lin_vel_y_values)))
    for i, lin_vel_x in enumerate(lin_vel_x_values):
        for j, lin_vel_y in enumerate(lin_vel_y_values):
            lin_vel_xy = np.array([lin_vel_x, lin_vel_y])
            ang_vel_xy = np.array([lin_vel_x, lin_vel_y])  # Example angular velocity

            # Calculate linear velocity error
            lin_vel_error = np.linalg.norm(lin_vel_xy - lin_vel_xy_ref, axis=-1)
            lin_vel_error_exp = np.exp(-tracking_sigma_lin * lin_vel_error)

            # Calculate angular velocity error
            ang_vel_error = np.linalg.norm(ang_vel_xy - ang_vel_xy_ref, axis=-1)
            ang_vel_error_exp = np.exp(-tracking_sigma_ang * ang_vel_error)

            # Compute the reward
            linear_error = 0.2 * (lin_vel_error + ang_vel_error)
            reward = (lin_vel_error_exp + ang_vel_error_exp) / 2.0 - linear_error
            rewards[j, i] = reward

    # Plot the reward surface with contours and labels
    plt.figure(figsize=(8, 6))
    contour = plt.contour(
        lin_vel_x_values,
        lin_vel_y_values,
        rewards,
        levels=10,
        colors="black",
        linewidths=0.8,
    )
    plt.contourf(lin_vel_x_values, lin_vel_y_values, rewards, levels=50, cmap="viridis")
    plt.colorbar(label="Reward")
    plt.clabel(
        contour, inline=True, fontsize=8, fmt="%.2f"
    )  # Add labels to the contours
    plt.scatter(
        lin_vel_xy_ref[0],
        lin_vel_xy_ref[1],
        color="red",
        label="Linear Velocity Ref",
        zorder=5,
    )
    plt.scatter(
        ang_vel_xy_ref[0],
        ang_vel_xy_ref[1],
        color="blue",
        label="Angular Velocity Ref",
        zorder=5,
    )
    plt.xlabel("Linear Velocity X / Angular Velocity X")
    plt.ylabel("Linear Velocity Y / Angular Velocity Y")
    plt.title("Reward vs. Linear and Angular Velocities in XY")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# test_torso_pitch()
# test_feet_air_time()
# test_lin_vel_xy()
# test_track_vel_hard()
