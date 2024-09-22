import math
from dataclasses import is_dataclass
from typing import Any, Dict, List, Optional, Tuple

from scipy.signal import chirp

from toddlerbot.utils.array_utils import ArrayType
from toddlerbot.utils.array_utils import array_lib as np


def get_random_sine_signal_config(
    duration: float,
    control_dt: float,
    mean: float,
    frequency_range: List[float],
    amplitude_range: List[float],
):
    frequency = np.random.uniform(*frequency_range)
    amplitude = np.random.uniform(*amplitude_range)

    sine_signal_config: Dict[str, float] = {
        "frequency": frequency,
        "amplitude": amplitude,
        "duration": duration,
        "control_dt": control_dt,
        "mean": mean,
    }

    return sine_signal_config


def get_sine_signal(sine_signal_config: Dict[str, float]):
    """
    Generates a sinusoidal signal based on the given parameters.
    """
    t = np.linspace(
        0,
        sine_signal_config["duration"],
        int(sine_signal_config["duration"] / sine_signal_config["control_dt"]),
        endpoint=False,
        dtype=np.float32,
    )
    signal = sine_signal_config["mean"] + sine_signal_config["amplitude"] * np.sin(
        2 * np.pi * sine_signal_config["frequency"] * t
    )
    return t, signal.astype(np.float32)


def get_chirp_signal(
    duration: float,
    control_dt: float,
    mean: float,
    initial_frequency: float,
    final_frequency: float,
    amplitude: float,
    decay_rate: float,
    method: str = "linear",  # "linear", "quadratic", "logarithmic", etc.
) -> Tuple[ArrayType, ArrayType]:
    t = np.linspace(
        0, duration, int(duration / control_dt), endpoint=False, dtype=np.float32
    )

    # Generate chirp signal without amplitude modulation
    chirp_signal = chirp(
        t, f0=initial_frequency, f1=final_frequency, t1=duration, method=method, phi=-90
    )

    # Apply an amplitude decay envelope based on time (or frequency)
    amplitude_envelope = amplitude * np.exp(-decay_rate * t)

    # Modulate the chirp signal with the decayed amplitude
    signal = mean + amplitude_envelope * chirp_signal

    return t, signal.astype(np.float32)


def round_floats(obj: Any, precision: int = 6) -> Any:
    """
    Recursively round floats in a list-like structure to a given precision.

    Args:
        obj: The list, tuple, or numpy array to round.
        precision (int): The number of decimal places to round to.

    Returns:
        The rounded list, tuple, or numpy array.
    """
    if isinstance(obj, float):
        return round(obj, precision)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(round_floats(x, precision) for x in obj)
    elif isinstance(obj, np.ndarray):
        return list(np.round(obj, decimals=precision))
    elif isinstance(obj, dict):
        return {k: round_floats(v, precision) for k, v in obj.items()}
    elif is_dataclass(obj):
        return type(obj)(  # type: ignore
            **{
                field.name: round_floats(getattr(obj, field.name), precision)
                for field in obj.__dataclass_fields__.values()
            }
        )

    return obj


def round_to_sig_digits(x: float, digits: int):
    if x == 0.0:
        return 0.0  # Zero is zero in any significant figure
    return round(x, digits - int(math.floor(math.log10(abs(x)))) - 1)


def quat2euler(quat: ArrayType, order: str = "wxyz") -> ArrayType:
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw).

    Args:
        quat: Quaternion as [w, x, y, z] or [x, y, z, w].

    Returns:
        Euler angles as [roll, pitch, yaw].
    """
    if order == "xyzw":
        x, y, z, w = quat
    else:
        w, x, y, z = quat

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return np.array([roll, pitch, yaw])


def euler2quat(euler: ArrayType, order: str = "wxyz") -> ArrayType:
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion.

    Args:
        euler: Euler angles as [roll, pitch, yaw].
        order: Output quaternion order, either "wxyz" or "xyzw".

    Returns:
        Quaternion as [w, x, y, z] or [x, y, z, w].
    """
    roll, pitch, yaw = euler

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    if order == "xyzw":
        return np.array([x, y, z, w])
    else:
        return np.array([w, x, y, z])


def quat_inv(quat: ArrayType, order: str = "wxyz") -> ArrayType:
    """Compute the inverse of a quaternion."""
    if order == "xyzw":
        x, y, z, w = quat
    else:
        w, x, y, z = quat

    norm = w**2 + x**2 + y**2 + z**2
    return np.array([w, -x, -y, -z]) / norm


def quat_mult(q1: ArrayType, q2: ArrayType, order: str = "wxyz") -> ArrayType:
    """Multiply two quaternions."""
    if order == "wxyz":
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
    else:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def rotate_vec(vector: ArrayType, quat: ArrayType):
    """Rotate a vector by a quaternion."""
    v = np.array([0.0] + list(vector))
    q_inv = quat_inv(quat)
    v_rotated = quat_mult(quat_mult(quat, v), q_inv)
    return v_rotated[1:]


def exponential_moving_average(
    alpha: ArrayType | float,
    current_value: ArrayType | float,
    previous_filtered_value: Optional[ArrayType | float] = None,
) -> ArrayType | float:
    if previous_filtered_value is None:
        return current_value
    return alpha * current_value + (1 - alpha) * previous_filtered_value


def gaussian_basis_functions(phase: ArrayType, N: int = 50):
    centers = np.linspace(0, 1, N)
    # Compute the Gaussian basis functions
    basis = np.exp(-np.square(phase - centers) / (2 * N**2))
    return basis


def interpolate(
    p_start: ArrayType | float,
    p_end: ArrayType | float,
    duration: ArrayType | float,
    t: ArrayType | float,
    interp_type: str = "linear",
) -> ArrayType | float:
    """
    Interpolate position at time t using specified interpolation type.

    Parameters:
    - p_start: Initial position.
    - p_end: Desired end position.
    - duration: Total duration from start to end.
    - t: Current time (within 0 to duration).
    - interp_type: Type of interpolation ('linear', 'quadratic', 'cubic').

    Returns:
    - Position at time t.
    """
    if t <= 0:
        return p_start

    if t >= duration:
        return p_end

    if interp_type == "linear":
        return p_start + (p_end - p_start) * (t / duration)
    elif interp_type == "quadratic":
        a = (-p_end + p_start) / duration**2
        b = (2 * p_end - 2 * p_start) / duration
        return a * t**2 + b * t + p_start
    elif interp_type == "cubic":
        a = (2 * p_start - 2 * p_end) / duration**3
        b = (3 * p_end - 3 * p_start) / duration**2
        return a * t**3 + b * t**2 + p_start
    else:
        raise ValueError("Unsupported interpolation type: {}".format(interp_type))


def binary_search(arr: ArrayType, t: ArrayType | float) -> int:
    # Implement binary search using either NumPy or JAX.
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] < t:
            low = mid + 1
        elif arr[mid] > t:
            high = mid - 1
        else:
            return mid
    return low - 1


def interpolate_action(
    t: ArrayType | float,
    time_arr: ArrayType,
    action_arr: ArrayType,
    interp_type: str = "linear",
):
    if t <= time_arr[0]:
        return action_arr[0]
    elif t >= time_arr[-1]:
        return action_arr[-1]

    # Use binary search to find the segment containing current_time
    idx = binary_search(time_arr, t)
    idx = max(0, min(idx, len(time_arr) - 2))  # Ensure idx is within valid range

    p_start = action_arr[idx]
    p_end = action_arr[idx + 1]
    duration = time_arr[idx + 1] - time_arr[idx]
    return interpolate(p_start, p_end, duration, t - time_arr[idx], interp_type)


# def interpolate_pos(
#     set_pos: Callable[[npt.NDArray[np.float32]], None],
#     pos_start: npt.NDArray[np.float32],
#     pos: npt.NDArray[np.float32],
#     duration: float,
#     interp_type: str,
#     sleep_time: float = 0.0,
# ):
#     time_start = time.time()
#     time_curr = 0
#     counter = 0
#     while time_curr <= duration:
#         time_curr = time.time() - time_start
#         pos_interp = interpolate(
#             pos_start, pos, duration, time_curr, interp_type=interp_type
#         )
#         set_pos(pos_interp)

#         time_elapsed = time.time() - time_start - time_curr
#         time_until_next_step = sleep_time - time_elapsed
#         if time_until_next_step > 0:
#             precise_sleep(time_until_next_step)

#         counter += 1


def resample_trajectory(
    trajectory: List[Tuple[float, Dict[str, float]]],
    desired_interval: float = 0.01,
    interp_type: str = "linear",
) -> List[Tuple[float, Dict[str, float]]]:
    resampled_trajectory: List[Tuple[float, Dict[str, float]]] = []
    for i in range(len(trajectory) - 1):
        t0, joint_angles_0 = trajectory[i]
        t1, joint_angles_1 = trajectory[i + 1]
        duration = t1 - t0

        # Add an epsilon to the desired interval to avoid floating point errors
        if duration > desired_interval + 1e-6:
            # More points needed, interpolate
            num_steps = int(duration / desired_interval)
            for j in range(num_steps):
                t = j * desired_interval
                interpolated_joint_angles: Dict[str, float] = {}
                for joint_name, p_start in joint_angles_0.items():
                    p_end = joint_angles_1[joint_name]
                    p_interp = interpolate(p_start, p_end, duration, t, interp_type)
                    interpolated_joint_angles[joint_name] = float(p_interp)
                resampled_trajectory.append((t0 + t, interpolated_joint_angles))
        else:
            # Interval is fine, keep the original point
            resampled_trajectory.append((t0, joint_angles_0))

    resampled_trajectory.append(trajectory[-1])

    return resampled_trajectory
