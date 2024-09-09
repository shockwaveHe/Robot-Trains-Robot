import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Define parameters
alpha = 0.3  # EMA smoothing factor, controls the cutoff frequency
fs = 100  # Sampling frequency (Hz)
num_points = 512  # Number of frequency points for plotting

# EMA filter transfer function coefficients
b = [alpha]  # Numerator (feed-forward)
a = [1, -(1 - alpha)]  # Denominator (feedback)

# Frequency response
w, h = signal.freqz(b, a, worN=num_points)

# Convert frequency to Hz
frequencies = (w / (2 * np.pi)) * fs

# Plot magnitude response (linear scale)
plt.figure(figsize=(10, 6))
plt.plot(frequencies, np.abs(h), label=f"EMA Filter (alpha = {alpha})")
plt.title("Frequency Response of EMA Filter")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (Linear)")
plt.grid(True)
plt.legend(loc="best")
plt.xlim(0, fs / 2)
plt.ylim(0, 1.1)  # Set y-axis for magnitude from 0 to 1
plt.show()
