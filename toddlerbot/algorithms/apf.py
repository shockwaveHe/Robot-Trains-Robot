from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


class APF:
    def __init__(
        self,
        table_bounds: List[float],
        k_att: float = 1.0,
        k_rep: float = 100.0,
        Q_star: float = 0.5,
        dt: float = 0.02,
        max_iters: int = 10000,
        epsilon: float = 0.01,
        damping_coefficient: float = 10.0,
    ) -> None:
        # Table boundaries (rectangle)
        self.x1, self.y1, self.x2, self.y2 = table_bounds

        # Parameters
        self.k_att = k_att
        self.k_rep = k_rep
        self.Q_star = Q_star
        self.dt = dt
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.damping_coefficient = damping_coefficient

    def compute_attractive_force(self, x: float, y: float) -> Tuple[float, float]:
        """Compute the attractive force towards the goal."""
        F_att_x = -self.k_att * (x - self.x_goal)
        F_att_y = -self.k_att * (y - self.y_goal)
        return F_att_x, F_att_y

    def compute_distance_and_gradient(
        self, x: float, y: float
    ) -> Tuple[float, float, float]:
        """Compute the minimal distance to the rectangle and its gradient."""
        # Closest point on the rectangle to the point (x, y)
        x_closest = np.clip(x, self.x1, self.x2)
        y_closest = np.clip(y, self.y1, self.y2)

        dx = x - x_closest
        dy = y - y_closest
        Q = np.hypot(dx, dy)

        if Q != 0:
            grad_Q_x = dx / Q
            grad_Q_y = dy / Q
        else:
            # Inside the rectangle; find the closest edge
            distances = [
                (x - self.x1, (1, 0)),  # Left edge
                (self.x2 - x, (-1, 0)),  # Right edge
                (y - self.y1, (0, 1)),  # Bottom edge
                (self.y2 - y, (0, -1)),  # Top edge
            ]
            Q, (grad_Q_x, grad_Q_y) = min((abs(d), g) for d, g in distances)

        return Q, grad_Q_x, grad_Q_y

    def compute_repulsive_force(self, x: float, y: float) -> Tuple[float, float]:
        """Compute the repulsive force from the table."""
        Q, grad_Q_x, grad_Q_y = self.compute_distance_and_gradient(x, y)
        if Q > self.Q_star:
            F_rep_x, F_rep_y = 0.0, 0.0
        else:
            factor = self.k_rep * (1.0 / Q - 1.0 / self.Q_star) / (Q**2)
            F_rep_x = factor * grad_Q_x
            F_rep_y = factor * grad_Q_y

        return F_rep_x, F_rep_y

    def plan_path(
        self, x_start: float, y_start: float, x_goal: float, y_goal: float
    ) -> None:
        """Plan the path using APF."""
        # Start and goal positions
        self.x_start = x_start
        self.y_start = y_start
        self.x_goal = x_goal
        self.y_goal = y_goal

        # Initialize position and velocity
        self.x = x_start
        self.y = y_start
        self.vx = 0.0
        self.vy = 0.0
        self.path: List[Tuple[float, float]] = [(self.x, self.y)]
        self.velocities: List[Tuple[float, float]] = [(self.vx, self.vy)]

        for i in range(self.max_iters):
            # Compute forces
            F_att_x, F_att_y = self.compute_attractive_force(self.x, self.y)
            F_rep_x, F_rep_y = self.compute_repulsive_force(self.x, self.y)

            # Total force with damping
            F_total_x = F_att_x + F_rep_x - self.damping_coefficient * self.vx
            F_total_y = F_att_y + F_rep_y - self.damping_coefficient * self.vy

            # Update velocity
            self.vx += F_total_x * self.dt
            self.vy += F_total_y * self.dt
            self.velocities.append((self.vx, self.vy))

            # Update position
            self.x += self.vx * self.dt
            self.y += self.vy * self.dt
            self.path.append((self.x, self.y))

            # Check for convergence
            if np.hypot(self.x - self.x_goal, self.y - self.y_goal) < self.epsilon:
                print(f"Converged in {i+1} iterations.")
                break
        else:
            print("Did not converge within the maximum number of iterations.")


def plot_apf(apf: APF) -> None:
    """Plot the path and the repulsive potential field as a heatmap."""
    # Plotting
    plt.figure(figsize=(10, 8))

    path = np.array(apf.path)
    path_x = path[:, 0]
    path_y = path[:, 1]

    x_min = min(apf.x_start, apf.x_goal, apf.x1) - 1
    x_max = max(apf.x_start, apf.x_goal, apf.x2) + 1
    y_min = min(apf.y_start, apf.y_goal, apf.y1) - 1
    y_max = max(apf.y_start, apf.y_goal, apf.y2) + 1

    # Plot the path
    plt.plot(path_x, path_y, "b-", linewidth=2, label="Path of the humanoid")
    plt.plot(apf.x_start, apf.y_start, "go", markersize=8, label="Start")
    plt.plot(apf.x_goal, apf.y_goal, "ro", markersize=8, label="Goal")

    # Plot table as a rectangle
    table = plt.Rectangle(
        (apf.x1, apf.y1),
        apf.x2 - apf.x1,
        apf.y2 - apf.y1,
        fc="gray",
        ec="black",
        alpha=0.5,
        label="Table",
    )
    plt.gca().add_patch(table)

    # Setting plot limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("Humanoid Path Planning with Repulsive Potential Field")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Table boundaries (rectangle)
    table_bounds = [1.0, 1.0, 3.0, 2.0]  # (x1, y1, x2, y2)

    # Start and goal positions
    x_start, y_start = 0.0, 0.0
    x_goal, y_goal = 2.0, 2.5

    # Create an instance of APF
    apf = APF(table_bounds)

    # Plan the path
    apf.plan_path(x_start, y_start, x_goal, y_goal)

    # Plot the results
    plot_apf(apf)
