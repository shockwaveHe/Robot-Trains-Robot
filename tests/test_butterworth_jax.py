import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def butterworth_filter_coefficients(f_s, f_c, N):
    """Calculates the coefficients for an Nth-order Butterworth filter."""
    T = 1 / f_s  # Sampling period
    # Pre-warp the cutoff frequency
    omega_c = (2 / T) * jnp.tan(jnp.pi * f_c / f_s)

    # Compute the analog prototype poles
    k = jnp.arange(1, N + 1)
    theta = jnp.pi * (2 * k - 1) / (2 * N)
    s_k = omega_c * -jnp.exp(1j * theta)

    # Apply the bilinear transform to get digital poles
    z_k = (1 + s_k * T / 2) / (1 - s_k * T / 2)

    # Ensure stability inside the unit circle
    z_k = z_k / jnp.abs(z_k)

    # Compute denominator polynomial from poles
    a = jnp.poly(z_k)

    # Numerator polynomial is scaled to achieve unity gain at zero frequency
    b = jnp.array([jnp.prod(1 - z_k)])

    # Normalize the coefficients
    a = a / a[0]
    b = b / a[0]

    return b.real, a.real


def jax_freqz(b, a, worN=512):
    """Computes the frequency response of a digital filter."""
    w = jnp.linspace(0, jnp.pi, worN)
    z = jnp.exp(-1j * w)
    h = jnp.polyval(b, z) / jnp.polyval(a, z)
    return w, h


# Define parameters
cutoff_freq = 10  # Desired cutoff frequency (Hz)
order = 8  # Filter order
fs = 200  # Sampling frequency (Hz)
num_points = 512  # Number of frequency points for plotting

# SciPy implementation for comparison
import scipy.signal as signal

b_scipy, a_scipy = signal.butter(order, cutoff_freq / (fs / 2), btype="low")
w_scipy, h_scipy = signal.freqz(b_scipy, a_scipy, worN=num_points)
frequencies_scipy = (w_scipy / (2 * np.pi)) * fs

# JAX implementation
b_jax, a_jax = butterworth_filter_coefficients(fs, cutoff_freq, order)

# Compute frequency response using JAX
w_jax, h_jax = jax_freqz(b_jax, a_jax, num_points)
frequencies_jax = (w_jax / (2 * jnp.pi)) * fs

# Plot to compare the frequency responses
plt.figure(figsize=(10, 6))

plt.plot(
    frequencies_scipy,
    20 * np.log10(np.abs(h_scipy)),
    label="SciPy Frequency Response",
    color="blue",
)
plt.plot(
    frequencies_jax,
    20 * np.log10(np.abs(h_jax)),
    label="JAX Frequency Response",
    linestyle="dashed",
    color="red",
)

plt.title("Frequency Response Comparison (SciPy vs JAX)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.legend()
plt.grid(True)
plt.show()
