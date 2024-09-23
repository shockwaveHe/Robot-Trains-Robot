import jax.numpy as jnp
from scipy.signal import butter

# Example: Low-pass Butterworth filter of order 4 with cutoff frequency at 5 Hz
order = 8
fs = 50  # Sampling frequency
cutoff = 10  # Cutoff frequency
b, a = butter(order, cutoff / (0.5 * fs), btype="low", analog=False)

# Print or save coefficients
print("b coefficients:", b)
print("a coefficients:", a)


# Recursive Butterworth filter implementation in JAX
def butterworth_filter(b, a, x, state):
    """
    Apply Butterworth filter to a single data point `x` using filter coefficients `b` and `a`.
    State holds past input and output values to maintain continuity.

    Arguments:
    - b: Filter numerator coefficients (b_0, b_1, ..., b_m)
    - a: Filter denominator coefficients (a_0, a_1, ..., a_n) with a[0] = 1
    - x: Current input value
    - state: Tuple of (past_inputs, past_outputs)

    Returns:
    - y: Filtered output
    - new_state: Updated state to use in the next step
    """
    past_inputs, past_outputs = state

    # Compute the current output y[n] based on the difference equation
    y = b[0] * x + jnp.sum(b[1:] * past_inputs) - jnp.sum(a[1:] * past_outputs)

    # Update the state with the new input/output for the next iteration
    new_past_inputs = jnp.concatenate(
        [jnp.array([x]), past_inputs[:-1]]
    )  # Shift inputs
    new_past_outputs = jnp.concatenate(
        [jnp.array([y]), past_outputs[:-1]]
    )  # Shift outputs

    new_state = (new_past_inputs, new_past_outputs)

    return y, new_state


# Example usage
state = jnp.zeros(order), jnp.zeros(order)

# Simulate a real-time sequence of target joint angles
joint_angles = jnp.array([0.5, 0.6, 0.8, 0.9, 1.0, 1.2])  # Example inputs

# Apply the filter in a loop (like in real-time data generation)
for angle in joint_angles:
    filtered_angle, state = butterworth_filter(b, a, angle, state)
    print(f"Filtered angle: {filtered_angle}")
