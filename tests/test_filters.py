import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

filter_name = "butterworth"
fs = 50  # Sampling frequency (Hz)
num_points = 512  # Number of frequency points for plotting

if filter_name == "ema":
    # Define parameters
    alpha = 0.55  # EMA smoothing factor, controls the cutoff frequency

    # EMA filter transfer function coefficients
    b = [alpha]  # Numerator (feed-forward)
    a = [1, -(1 - alpha)]  # Denominator (feedback)

elif filter_name == "butterworth":
    # Define parameters
    cutoff_freq = 10  # Desired cutoff frequency (Hz)
    order = 8  # Higher order for a steeper cutoff

    # Design an 8th-order Butterworth low-pass filter
    b, a = signal.butter(order, cutoff_freq / (fs / 2), btype="low")

else:
    raise ValueError("Invalid filter name")

# Frequency response
w, h = signal.freqz(b, a, worN=num_points)

# Convert frequency to Hz
frequencies = (w / (2 * np.pi)) * fs

# Plot magnitude response (not in dB, linear magnitude)
plt.figure(figsize=(10, 6))
plt.plot(frequencies, np.abs(h), label=f"{filter_name} Filter")
plt.title("Frequency Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (Linear)")
plt.grid(True)
plt.legend(loc="best")
plt.xlim(0, fs / 2)  # Limit x-axis to half the sampling frequency (Nyquist frequency)
plt.ylim(0, 1.1)  # Limit y-axis for better visualization of magnitude
plt.show()
