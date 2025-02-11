# import matplotlib.pyplot as plt
# import numpy as np
# from scipy import signal

# # Define parameters
# alpha = 0.3  # EMA smoothing factor, controls the cutoff frequency
# fs = 100  # Sampling frequency (Hz)
# num_points = 512  # Number of frequency points for plotting

# # EMA filter transfer function coefficients
# b = [alpha]  # Numerator (feed-forward)
# a = [1, -(1 - alpha)]  # Denominator (feedback)

# # Frequency response
# w, h = signal.freqz(b, a, worN=num_points)

# # Convert frequency to Hz
# frequencies = (w / (2 * np.pi)) * fs

# # Plot magnitude response (linear scale)
# plt.figure(figsize=(10, 6))
# plt.plot(frequencies, np.abs(h), label=f"EMA Filter (alpha = {alpha})")
# plt.title("Frequency Response of EMA Filter")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude (Linear)")
# plt.grid(True)
# plt.legend(loc="best")
# plt.xlim(0, fs / 2)
# plt.ylim(0, 1.1)  # Set y-axis for magnitude from 0 to 1
# plt.show()


import numpy as np
import timeit
from scipy.signal import lfilter, lfilter_zi

# 1. Original loop-based EMA
def smooth_data_loop(data, smoothing_factor=0.9):
    """Compute EMA via an explicit Python loop."""
    smoothed = np.empty_like(data, dtype=np.float64)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = (1 - smoothing_factor) * data[i] + smoothing_factor * smoothed[i - 1]
    return smoothed

# 2. Optimized version using SciPy's lfilter
def smooth_data_lfilter(data, alpha=0.9):
    """
    Compute EMA using an IIR filter.
    
    The recurrence:
        y[0] = x[0]
        y[i] = (1 - smoothing_factor)*x[i] + smoothing_factor*y[i-1]
    is equivalent to the filter with
        b = [1 - smoothing_factor]
        a = [1, -smoothing_factor]
    We set the initial condition so that for a constant input x[0] we get y[0] = x[0].
    """
    b = [1 - alpha]
    a = [1, -alpha]
    # Compute steady-state initial condition for a constant input equal to data[0]
    zi = lfilter_zi(b, a) * data[0]
    y, _ = lfilter(b, a, data, zi=zi)
    return y

# Test data (a large random array)
np.random.seed(42)py
test_data = np.random.rand(100_000)

# Verify correctness
smoothed_loop = smooth_data_loop(test_data, smoothing_factor=0.9)
smoothed_lfilter = smooth_data_lfilter(test_data, alpha=0.9)

if np.allclose(smoothed_loop, smoothed_lfilter, atol=1e-8):
    print("Correctness check passed: both methods produce nearly identical results.")
else:
    print("Mismatch between loop and lfilter implementations!")

# Benchmark performance (10 runs each)
loop_time = timeit.timeit(lambda: smooth_data_loop(test_data, 0.9), number=100)
lfilter_time = timeit.timeit(lambda: smooth_data_lfilter(test_data, 0.9), number=100)

print("\nBenchmark results (10 runs):")
print(f"Loop-based EMA time:   {loop_time:.4f} sec")
print(f"lfilter-based EMA time: {lfilter_time:.4f} sec")
print(f"Speedup factor: {loop_time / lfilter_time:.2f}x faster using lfilter")
