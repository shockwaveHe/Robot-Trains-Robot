import matplotlib.pyplot as plt
import numpy as np

# Parameters
k_att = 1.0  # Attractive potential gain
k_rep = 100.0  # Repulsive potential gain
Q_star = 0.5  # Influence distance for the repulsive potential
dt = 0.01  # Time step
max_iters = 10000
epsilon = 0.01  # Threshold for convergence
damping_coefficient = 10.0  # Damping coefficient

# Table boundaries (rectangle)
x1, y1 = 1.0, 1.0  # Bottom-left corner
x2, y2 = 3.0, 2.0  # Top-right corner

# Goal position (on one side of the table)
x_goal, y_goal = 2.0, 2.5

# Start position of the humanoid
x_start, y_start = 0.0, 0.0

# Initialize position and velocity
x, y = x_start, y_start
vx, vy = 0.0, 0.0
path = [(x, y)]


def compute_attractive_force(x, y, x_goal, y_goal):
    """Compute the attractive force towards the goal."""
    F_att_x = -k_att * (x - x_goal)
    F_att_y = -k_att * (y - y_goal)
    return F_att_x, F_att_y


def compute_distance_and_gradient(x, y, x1, y1, x2, y2):
    """Compute the minimal distance to the rectangle and its gradient."""
    # Closest point on the rectangle to the point (x, y)
    x_closest = np.clip(x, x1, x2)
    y_closest = np.clip(y, y1, y2)

    dx = x - x_closest
    dy = y - y_closest
    Q = np.hypot(dx, dy)

    if Q != 0:
        grad_Q_x = dx / Q
        grad_Q_y = dy / Q
    else:
        # Inside the rectangle; find the closest edge
        distances = [
            (x - x1, (1, 0)),  # Left edge
            (x2 - x, (-1, 0)),  # Right edge
            (y - y1, (0, 1)),  # Bottom edge
            (y2 - y, (0, -1)),
        ]  # Top edge
        Q, (grad_Q_x, grad_Q_y) = min((abs(d), g) for d, g in distances)
    return Q, grad_Q_x, grad_Q_y


def compute_repulsive_force(x, y, x1, y1, x2, y2):
    """Compute the repulsive force from the table."""
    Q, grad_Q_x, grad_Q_y = compute_distance_and_gradient(x, y, x1, y1, x2, y2)
    if Q > Q_star:
        F_rep_x, F_rep_y = 0.0, 0.0
    else:
        factor = k_rep * (1.0 / Q - 1.0 / Q_star) / (Q**2)
        F_rep_x = factor * grad_Q_x
        F_rep_y = factor * grad_Q_y
    return F_rep_x, F_rep_y


# Simulation loop
for i in range(max_iters):
    # Compute forces
    F_att_x, F_att_y = compute_attractive_force(x, y, x_goal, y_goal)
    F_rep_x, F_rep_y = compute_repulsive_force(x, y, x1, y1, x2, y2)

    # Total force
    F_total_x = F_att_x + F_rep_x - damping_coefficient * vx
    F_total_y = F_att_y + F_rep_y - damping_coefficient * vy

    # Update velocity
    vx += F_total_x * dt
    vy += F_total_y * dt

    # Update position
    x += vx * dt
    y += vy * dt
    path.append((x, y))

    # Check for convergence
    if np.hypot(x - x_goal, y - y_goal) < epsilon:
        print(f"Converged in {i+1} iterations.")
        break
else:
    print("Did not converge within the maximum number of iterations.")

# Extract path coordinates
path = np.array(path)
path_x = path[:, 0]
path_y = path[:, 1]

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(path_x, path_y, "b-", label="Path of the humanoid")
plt.plot(x_start, y_start, "go", label="Start")
plt.plot(x_goal, y_goal, "ro", label="Goal")

# Plot table as a rectangle
table = plt.Rectangle(
    (x1, y1), x2 - x1, y2 - y1, fc="gray", ec="black", alpha=0.5, label="Table"
)
plt.gca().add_patch(table)

# Setting plot limits
plt.xlim(min(x_start, x_goal) - 1, max(x_start, x_goal, x2) + 1)
plt.ylim(min(y_start, y_goal) - 1, max(y_start, y_goal, y2) + 1)

plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Humanoid Path Planning using Artificial Potential Fields with Damping")
plt.legend()
plt.grid(True)
plt.show()
