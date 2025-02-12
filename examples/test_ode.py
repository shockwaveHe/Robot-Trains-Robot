import numpy as np
import matplotlib.pyplot as plt


# Define the ODE as a function: dy/dt = f(t, y)
def f(t, y):
    return y


# Euler's Method
def euler(f, t0, y0, t_end, h):
    t = np.arange(t0, t_end + h, h)  # time grid
    y = np.zeros(len(t))  # array to store the approximate solution
    y[0] = y0
    for i in range(len(t) - 1):
        y[i + 1] = y[i] + h * f(t[i], y[i])
    return t, y


# Midpoint Method (a simple 2nd-order Runge-Kutta method)
def midpoint(f, t0, y0, t_end, h):
    t = np.arange(t0, t_end + h, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(len(t) - 1):
        # Estimate the slope at the beginning of the interval
        k1 = f(t[i], y[i])
        # Use k1 to estimate the state at the midpoint
        y_mid = y[i] + (h / 2) * k1
        # Evaluate the slope at the midpoint
        k2 = f(t[i] + h / 2, y_mid)
        # Update the solution using the slope at the midpoint
        y[i + 1] = y[i] + h * k2
    return t, y


# Classical 4th Order Runge-Kutta Method (RK4)
def rk4(f, t0, y0, t_end, h):
    t = np.arange(t0, t_end + h, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(len(t) - 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h / 2, y[i] + (h / 2) * k1)
        k3 = f(t[i] + h / 2, y[i] + (h / 2) * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y


# True solution for comparison: y(t) = exp(t)
def true_solution(t):
    return np.exp(t)


# Set parameters for the ODE
t0 = 0.0  # initial time
t_end = 2.0  # final time
y0 = 1.0  # initial value y(0)
h = 0.5  # time step size

# Compute the approximate solutions using the numerical methods
t_euler, y_euler = euler(f, t0, y0, t_end, h)
t_mid, y_mid = midpoint(f, t0, y0, t_end, h)
t_rk4, y_rk4 = rk4(f, t0, y0, t_end, h)

# Compute the true solution on a dense time grid for a smooth curve
t_true = np.linspace(t0, t_end, 200)
y_true = true_solution(t_true)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t_true, y_true, "k-", label="True Solution (e^t)", linewidth=2)
plt.plot(t_euler, y_euler, "o-", label="Euler's Method")
plt.plot(t_mid, y_mid, "s-", label="Midpoint Method")
plt.plot(t_rk4, y_rk4, "d-", label="RK4 Method")

plt.xlabel("t", fontsize=14)
plt.ylabel("y(t)", fontsize=14)
plt.title("Comparison of Numerical Methods for Solving the ODE: dy/dt = y", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
