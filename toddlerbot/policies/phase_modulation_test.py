import matplotlib
import numpy as np


def main():
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from scipy.signal import hilbert

    # Create a sample sinusoidal signal.
    fs = 1000  # Sampling frequency (Hz)
    t = np.linspace(0, 1, fs, endpoint=False)
    f = 50  # Frequency in Hz
    phi = 0.5  # Phase offset
    original_phase = (2 * np.pi * f * t + phi + np.pi - np.pi / 2) % (2 * np.pi) - np.pi
    signal = np.sin(2 * np.pi * f * t + phi)

    # Compute the analytic signal using the Hilbert transform.
    analytic_signal = hilbert(signal)
    instantaneous_phase = np.angle(analytic_signal)  # cos signal

    print(instantaneous_phase[0] + np.pi / 2)
    # Plot the instantaneous phase.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))
    ax1.plot(t, original_phase)
    ax1.set_title("Original phase")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Original Phase [rad]")
    ax1.grid()
    ax2.plot(t, instantaneous_phase)
    ax2.set_title("Instantaneous Phase")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Phase [rad]")
    ax2.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
