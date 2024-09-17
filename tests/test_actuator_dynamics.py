import matplotlib.pyplot as plt
import numpy as np


class Actuator:
    def __init__(
        self,
        kP,
        kD,
        epsilon_q_max,
        mu_s,
        mu_d,
        q_dot_s,
        tau_max,
        q_dot_tau_max,
        q_dot_max,
        tau_b,
        b_min,
        b_max,
        sigma_q_0,
        sigma_q_1,
        I_m,
        fd_vel=False,
    ):
        """
        Initialize the actuator with given parameters.

        Parameters:
        - kP, kD: Proportional and derivative gains.
        - epsilon_q_max: Maximum encoder offset.
        - mu_s, mu_d: Static and dynamic friction coefficients.
        - q_dot_s: Friction activation parameter.
        - tau_max: Maximum torque at low velocities.
        - q_dot_tau_max: Velocity where torque starts to decrease.
        - q_dot_max: Velocity where torque limit drops to zero.
        - tau_b: Backlash activation parameter.
        - b_min, b_max: Minimum and maximum backlash.
        - sigma_q_0, sigma_q_1: Noise model parameters.
        - I_m: Reflected inertia of the actuator.
        - head_actuator: Boolean indicating if the actuator is in the head.
        """
        self.kP = kP
        self.kD = kD
        self.epsilon_q_max = epsilon_q_max
        self.mu_s = mu_s
        self.mu_d = mu_d
        self.q_dot_s = q_dot_s
        self.tau_max = tau_max
        self.q_dot_tau_max = q_dot_tau_max
        self.q_dot_max = q_dot_max
        self.tau_b = tau_b
        self.b_min = b_min
        self.b_max = b_max
        self.sigma_q_0 = sigma_q_0
        self.sigma_q_1 = sigma_q_1
        self.I_m = I_m
        self.fd_vel = fd_vel
        # Randomized parameters per episode
        self.reset_episode()

    def reset_episode(self):
        """
        Reset parameters that are randomized at the start of each episode.
        """
        self.epsilon_q = np.random.uniform(-self.epsilon_q_max, self.epsilon_q_max)
        self.b = np.random.uniform(self.b_min, self.b_max)
        # Randomize I_m up to ±20% offset
        self.I_m_randomized = self.I_m * np.random.uniform(0.8, 1.2)
        # Initialize previous error for head actuators
        self.error_prev = 0.0

    def compute_tau_m(self, a, q, q_dot, dt):
        """
        Compute the motor torque τ_m.

        Parameters:
        - a: Joint setpoint.
        - q: Joint position.
        - q_dot: Joint velocity.
        - dt: Time step for numerical differentiation.

        Returns:
        - tau_m: Motor torque.
        """
        q_tilde = q + self.epsilon_q  # Equation (15)
        error = a - q_tilde

        if self.fd_vel:
            # Equation (16)
            d_error_dt = (error - self.error_prev) / dt
            tau_m = self.kP * error + self.kD * d_error_dt
            self.error_prev = error
        else:
            # Equation (14)
            tau_m = self.kP * error - self.kD * q_dot

        return tau_m

    def compute_tau_f(self, q_dot):
        """
        Compute the friction torque τ_f.

        Parameters:
        - q_dot: Joint velocity.

        Returns:
        - tau_f: Friction torque.
        """
        # Equation (17)
        tau_f = self.mu_s * np.tanh(q_dot / self.q_dot_s) + self.mu_d * q_dot
        return tau_f

    def compute_tau_limits(self, q_dot):
        """
        Compute the velocity-dependent torque limits τ_min and τ_max.

        Parameters:
        - q_dot: Joint velocity.

        Returns:
        - tau_min: Minimum torque limit.
        - tau_max: Maximum torque limit.
        """
        abs_q_dot = abs(q_dot)

        if abs_q_dot <= self.q_dot_tau_max:
            tau_limit = self.tau_max
        elif abs_q_dot <= self.q_dot_max:
            # Linear decrease of torque limit
            slope = self.tau_max / (self.q_dot_tau_max - self.q_dot_max)
            tau_limit = slope * (abs_q_dot - self.q_dot_tau_max) + self.tau_max
        else:
            tau_limit = 0.0

        tau_max = tau_limit
        tau_min = -tau_limit
        return tau_min, tau_max

    def compute_total_tau(self, tau_m, q_dot):
        """
        Compute the total torque τ applied at the joint.

        Parameters:
        - tau_m: Motor torque.
        - q_dot: Joint velocity.

        Returns:
        - tau: Total joint torque.
        """
        tau_f = self.compute_tau_f(q_dot)
        tau_min, tau_max = self.compute_tau_limits(q_dot)
        # Equation (18)
        tau_m_clamped = np.clip(tau_m, tau_min, tau_max)
        tau = tau_m_clamped - tau_f
        return tau

    def compute_q_hat(self, q, tau_m, q_dot):
        """
        Compute the measured joint position q̂.

        Parameters:
        - q: Joint position.
        - tau_m: Motor torque.
        - q_dot: Joint velocity.

        Returns:
        - q_hat: Measured joint position.
        """
        q_tilde = q + self.epsilon_q  # Equation (15)
        # Backlash term from Equation (19)
        backlash_term = 0.5 * self.b * np.tanh(tau_m / self.tau_b)
        # Noise model from Equation (20)
        sigma_q = self.sigma_q_0 + self.sigma_q_1 * abs(q_dot)
        noise = np.random.normal(0, sigma_q)
        # Equation (19)
        q_hat = q_tilde + backlash_term + noise
        return q_hat

    def step(self, a, q, q_dot, dt):
        """
        Simulate one time step of the actuator.

        Parameters:
        - a: Joint setpoint.
        - q: Current joint position.
        - q_dot: Current joint velocity.
        - dt: Time step duration.

        Returns:
        - tau: Total joint torque applied.
        - q_hat: Measured joint position.
        """
        tau_m = self.compute_tau_m(a, q, q_dot, dt)
        tau = self.compute_total_tau(tau_m, q_dot)
        q_hat = self.compute_q_hat(q, tau_m, q_dot)
        return tau, q_hat


# Define actuator parameters (example values)
kP = 5.0
kD = 0.2
q_dot_s = 0.01  # rad/s
tau_b = 1.0  # Nm

# Need to sysID
epsilon_q_max = 0.02  # radians
mu_s = 0.05
mu_d = 0.009
tau_max = 4.8  # Nm
q_dot_tau_max = 0.2  # rad/s
q_dot_max = 7.0  # rad/s
b_min = 0.002
b_max = 0.005  # radians
sigma_q_0 = 4.31e-4  # radians
sigma_q_1 = 2.43e-5  # radians per rad/s
I_m = 0.0058  # kg*m^2

# Create an actuator instance
actuator = Actuator(
    kP,
    kD,
    epsilon_q_max,
    mu_s,
    mu_d,
    q_dot_s,
    tau_max,
    q_dot_tau_max,
    q_dot_max,
    tau_b,
    b_min,
    b_max,
    sigma_q_0,
    sigma_q_1,
    I_m,
    fd_vel=False,
)

# Simulation parameters
dt = 0.001  # Time step (1 ms)
num_steps = 1000  # Number of simulation steps
a = np.pi / 4  # Desired joint position (setpoint)
q = 0.0  # Initial joint position
q_dot = 0.0  # Initial joint velocity

# Lists to store simulation data
q_list = []
q_hat_list = []
tau_list = []

# Simulation loop
for _ in range(num_steps):
    # Compute actuator outputs
    tau, q_hat = actuator.step(a, q, q_dot, dt)

    # Simple physics integration (Euler method)
    q_dot += (tau / actuator.I_m_randomized) * dt
    q += q_dot * dt

    # Store data
    q_list.append(q)
    q_hat_list.append(q_hat)
    tau_list.append(tau)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(q_list, label="Actual Position q")
plt.plot(q_hat_list, label="Measured Position q̂")
plt.legend()
plt.ylabel("Position (rad)")
plt.subplot(2, 1, 2)
plt.plot(tau_list, label="Total Torque τ")
plt.legend()
plt.xlabel("Time Steps")
plt.ylabel("Torque (Nm)")
plt.show()
