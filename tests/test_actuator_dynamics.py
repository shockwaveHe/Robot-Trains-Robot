import numpy as np


def actuator_dynamics(
    kp: float,
    kd: float,
    desired_pos: float,
    current_pos: float,
    velocity: float,
    tau_max: float,
    tau_min: float,
    mu_s: float,
    mu_d: float,
    backlash_params: dict,
    noise_params: dict,
) -> float:
    """
    Compute the motor torque with dynamic elements: friction, clamping, backlash, and noise.

    Args:
    kp (float): Proportional gain
    kd (float): Derivative gain
    desired_pos (float): Desired actuator position
    current_pos (float): Current actuator position
    velocity (float): Current actuator velocity
    tau_max (float): Maximum torque
    tau_min (float): Minimum torque
    mu_s (float): Static friction coefficient
    mu_d (float): Dynamic friction coefficient
    backlash_params (dict): Dictionary of backlash parameters
    noise_params (dict): Dictionary of noise parameters

    Returns:
    float: Final motor torque
    """

    # --------------------------
    # 1. PD Control Torque (without friction)
    # --------------------------
    pos_error = desired_pos - current_pos
    motor_torque = kp * pos_error - kd * velocity

    # --------------------------
    # 2. Backlash and Noise
    # --------------------------
    # Add noise to position and velocity
    position_noise = np.random.normal(0, noise_params["sigma_q"])
    velocity_noise = np.random.normal(0, noise_params["sigma_q_dot"])

    current_pos_with_noise = current_pos + position_noise
    velocity_with_noise = velocity + velocity_noise

    # Apply backlash: If velocity is small, apply a random backlash offset.
    if abs(velocity_with_noise) < backlash_params["vel_threshold"]:
        backlash_offset = np.random.uniform(
            backlash_params["backlash_min"], backlash_params["backlash_max"]
        )
        current_pos_with_noise += backlash_offset

    # Recalculate the motor torque with backlash and noise applied
    pos_error_with_noise = desired_pos - current_pos_with_noise
    motor_torque = kp * pos_error_with_noise - kd * velocity_with_noise

    # --------------------------
    # 3. Friction Model
    # --------------------------
    # Static friction term (using tanh for smooth approximation)
    friction_static = mu_s * np.tanh(velocity_with_noise / 0.01)

    # Dynamic friction term
    friction_dynamic = mu_d * velocity_with_noise

    # Total friction torque
    friction_torque = friction_static + friction_dynamic

    # Subtract friction from the motor torque
    motor_torque -= friction_torque

    # --------------------------
    # 4. Torque Clamping (limits based on velocity)
    # --------------------------
    tau_m = np.clip(motor_torque, tau_min, tau_max)

    return tau_m


# Example Parameters
kp = 10.0  # Proportional gain
kd = 2.0  # Derivative gain
desired_pos = 1.0  # Desired actuator position
current_pos = 0.8  # Current actuator position
velocity = 0.05  # Current velocity

# Friction parameters
mu_s = 0.1  # Static friction coefficient
mu_d = 0.01  # Dynamic friction coefficient

# Torque limits
tau_max = 5.0
tau_min = -5.0

# Backlash and noise parameters
backlash_params = {"backlash_min": 0.0, "backlash_max": 0.1, "vel_threshold": 0.02}

noise_params = {
    "sigma_q": 0.01,  # Standard deviation of position noise
    "sigma_q_dot": 0.005,  # Standard deviation of velocity noise
}

# Compute the motor torque with dynamic elements
torque = actuator_dynamics(
    kp,
    kd,
    desired_pos,
    current_pos,
    velocity,
    tau_max,
    tau_min,
    mu_s,
    mu_d,
    backlash_params,
    noise_params,
)
print(f"Final motor torque: {torque}")
