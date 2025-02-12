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
    """Generates a random sinusoidal signal configuration based on specified parameters.

    Args:
        duration (float): The total duration of the signal in seconds.
        control_dt (float): The time step for signal generation.
        mean (float): The mean value around which the sinusoidal signal oscillates.
        frequency_range (List[float]): A list containing the minimum and maximum frequency values for the signal.
        amplitude_range (List[float]): A list containing the minimum and maximum amplitude values for the signal.

    Returns:
        Tuple[ArrayType, ArrayType]: A tuple containing the time array and the generated sinusoidal signal array.
    """
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
    """Generate a sine signal based on the provided configuration.

    Args:
        sine_signal_config (Dict[str, float]): Configuration dictionary containing parameters for the sine signal, such as amplitude, frequency, and phase.

    Returns:
        np.ndarray: Array representing the generated sine signal.
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
    """Generate a chirp signal over a specified duration with varying frequency and amplitude.

    Args:
        duration: Total duration of the chirp signal in seconds.
        control_dt: Time step for the signal generation.
        mean: Mean value of the signal.
        initial_frequency: Starting frequency of the chirp in Hz.
        final_frequency: Ending frequency of the chirp in Hz.
        amplitude: Amplitude of the chirp signal.
        decay_rate: Rate at which the amplitude decays over time.
        method: Method of frequency variation, e.g., "linear", "quadratic", "logarithmic".

    Returns:
        A tuple containing:
        - Time array for the chirp signal.
        - Generated chirp signal array.
    """
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
    """Round a floating-point number to a specified number of significant digits.

    Args:
        x: The number to be rounded.
        digits: The number of significant digits to round to.

    Returns:
        The number rounded to the specified number of significant digits.
    """
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


def mat2quat(mat: ArrayType, order: str = "wxyz") -> ArrayType:
    """
    Convert a 3x3 rotation matrix to a quaternion.

    Args:
        mat: 3x3 rotation matrix.
        order: Order of the output quaternion, either "wxyz" or "xyzw".

    Returns:
        Quaternion as [w, x, y, z] or [x, y, z, w].
    """
    # Extract matrix elements
    m00, m01, m02 = mat[0, 0], mat[0, 1], mat[0, 2]
    m10, m11, m12 = mat[1, 0], mat[1, 1], mat[1, 2]
    m20, m21, m22 = mat[2, 0], mat[2, 1], mat[2, 2]

    # Calculate the trace of the matrix
    trace = m00 + m11 + m22

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif (m00 > m11) and (m00 > m22):
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    # Return quaternion in specified order
    if order == "xyzw":
        return np.array([x, y, z, w])
    else:
        return np.array([w, x, y, z])


def quat2mat(quat: ArrayType, order: str = "wxyz") -> ArrayType:
    """
    Convert a quaternion to a 3x3 rotation matrix.

    Args:
        quat: Quaternion as [w, x, y, z] or [x, y, z, w].
        order: Order of the input quaternion, either "wxyz" or "xyzw".

    Returns:
        A 3x3 rotation matrix.
    """
    # Extract quaternion components based on order
    if order == "xyzw":
        x, y, z, w = quat
    else:
        w, x, y, z = quat

    # Calculate the elements of the rotation matrix
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    mat = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    )

    return mat


def euler2mat(euler: ArrayType, order: str = "zyx") -> ArrayType:
    """
    Convert Euler angles (roll, pitch, yaw) to a 3x3 rotation matrix.

    Args:
        euler: Euler angles as [roll, pitch, yaw].

    Returns:
        A 3x3 rotation matrix.
    """
    roll, pitch, yaw = euler

    # Compute individual rotation matrices
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # Rotation matrix for each axis
    R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

    R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])

    R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    if order == "xyz":
        R = R_roll @ R_pitch @ R_yaw
    elif order == "xzy":
        R = R_roll @ R_yaw @ R_pitch
    elif order == "yxz":
        R = R_pitch @ R_roll @ R_yaw
    elif order == "yzx":
        R = R_pitch @ R_yaw @ R_roll
    elif order == "zxy":
        R = R_yaw @ R_roll @ R_pitch
    elif order == "zyx":
        R = R_yaw @ R_pitch @ R_roll
    else:
        raise ValueError(
            "Invalid order. Must be one of 'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'."
        )

    return R


def mat2euler(mat: ArrayType, order: str = "zyx") -> ArrayType:
    """
    Convert a 3x3 rotation matrix to Euler angles (roll, pitch, yaw).

    Args:
        mat: 3x3 rotation matrix.

    Returns:
        Euler angles as [roll, pitch, yaw].
    """
    if order == "zyx":
        # ZYX order (yaw, pitch, roll)
        if np.isclose(mat[2, 0], -1.0):
            # Gimbal lock case (pitch = 90 degrees)
            pitch = np.pi / 2
            roll = np.arctan2(mat[0, 1], mat[0, 2])
            yaw = 0
        elif np.isclose(mat[2, 0], 1.0):
            # Gimbal lock case (pitch = -90 degrees)
            pitch = -np.pi / 2
            roll = np.arctan2(-mat[0, 1], -mat[0, 2])
            yaw = 0
        else:
            # General case
            pitch = np.arcsin(-mat[2, 0])
            roll = np.arctan2(mat[2, 1] / np.cos(pitch), mat[2, 2] / np.cos(pitch))
            yaw = np.arctan2(mat[1, 0] / np.cos(pitch), mat[0, 0] / np.cos(pitch))

    elif order == "xyz":
        # XYZ order (roll, pitch, yaw)
        if np.isclose(mat[0, 2], -1.0):
            # Gimbal lock case (pitch = 90 degrees)
            pitch = np.pi / 2
            yaw = np.arctan2(mat[1, 0], mat[1, 1])
            roll = 0
        elif np.isclose(mat[0, 2], 1.0):
            # Gimbal lock case (pitch = -90 degrees)
            pitch = -np.pi / 2
            yaw = np.arctan2(-mat[1, 0], -mat[1, 1])
            roll = 0
        else:
            # General case
            pitch = np.arcsin(mat[0, 2])
            yaw = np.arctan2(-mat[0, 1] / np.cos(pitch), mat[0, 0] / np.cos(pitch))
            roll = np.arctan2(-mat[1, 2] / np.cos(pitch), mat[2, 2] / np.cos(pitch))

    else:
        raise ValueError("Invalid order. Supported orders are 'zyx' and 'xyz'.")

    return np.array([roll, pitch, yaw])


def quat_inv(quat: ArrayType, order: str = "wxyz") -> ArrayType:
    """Calculates the inverse of a quaternion.

    Args:
        quat (ArrayType): A quaternion represented as an array of four elements.
        order (str, optional): The order of the quaternion components. Either "wxyz" or "xyzw". Defaults to "wxyz".

    Returns:
        ArrayType: The inverse of the input quaternion.
    """
    if order == "xyzw":
        x, y, z, w = quat
    else:
        w, x, y, z = quat

    norm = w**2 + x**2 + y**2 + z**2
    return np.array([w, -x, -y, -z]) / np.sqrt(norm)


def quat_mult(q1: ArrayType, q2: ArrayType, order: str = "wxyz") -> ArrayType:
    """Multiplies two quaternions and returns the resulting quaternion.

    Args:
        q1 (ArrayType): The first quaternion, represented as an array of four elements.
        q2 (ArrayType): The second quaternion, represented as an array of four elements.
        order (str, optional): The order of quaternion components, either "wxyz" or "xyzw". Defaults to "wxyz".

    Returns:
        ArrayType: The resulting quaternion from the multiplication, in the specified order.
    """
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
    """Rotate a vector using a quaternion.

    Args:
        vector (ArrayType): The vector to be rotated.
        quat (ArrayType): The quaternion representing the rotation.

    Returns:
        ArrayType: The rotated vector.
    """
    v = np.array([0.0] + list(vector))
    q_inv = quat_inv(quat)
    v_rotated = quat_mult(quat_mult(quat, v), q_inv)
    return v_rotated[1:]


def exponential_moving_average(
    alpha: ArrayType | float,
    current_value: ArrayType | float,
    previous_filtered_value: Optional[ArrayType | float] = None,
) -> ArrayType | float:
    """Calculate the exponential moving average of a current value.

    This function computes the exponential moving average (EMA) for a given current value using a specified smoothing factor, `alpha`. If a previous filtered value is provided, it is used to compute the EMA; otherwise, the current value is used as the initial EMA.

    Args:
        alpha (ArrayType | float): The smoothing factor, where 0 < alpha <= 1.
        current_value (ArrayType | float): The current data point to be filtered.
        previous_filtered_value (Optional[ArrayType | float]): The previous EMA value. If None, the current value is used as the initial EMA.

    Returns:
        ArrayType | float: The updated exponential moving average.
    """
    if previous_filtered_value is None:
        return current_value
    return alpha * current_value + (1 - alpha) * previous_filtered_value


# Recursive Butterworth filter implementation in JAX
def butterworth(
    b: ArrayType,
    a: ArrayType,
    x: ArrayType,
    past_inputs: ArrayType,
    past_outputs: ArrayType,
) -> Tuple[ArrayType, ArrayType, ArrayType]:
    """
    Apply Butterworth filter to a single data point `x` using filter coefficients `b` and `a`.
    State holds past input and output values to maintain continuity.

    Args:
        b: Filter numerator coefficients (b_0, b_1, ..., b_m)
        a: Filter denominator coefficients (a_0, a_1, ..., a_n) with a[0] = 1
        x: Current input value
        state: Tuple of (past_inputs, past_outputs)

    Returns:
        y: Filtered output
        new_state: Updated state to use in the next step
    """
    # Compute the current output y[n] based on the difference equation
    y = (
        b[0] * x
        + np.sum(b[1:] * past_inputs, axis=0)
        - np.sum(a[1:] * past_outputs, axis=0)
    )

    # Update the state with the new input/output for the next iteration
    new_past_inputs = np.concatenate([x[None], past_inputs[:-1]], axis=0)
    new_past_outputs = np.concatenate([y[None], past_outputs[:-1]], axis=0)

    return y, new_past_inputs, new_past_outputs


def gaussian_basis_functions(phase: ArrayType, N: int = 50):
    """Resample a trajectory to a specified time interval using interpolation.

    Args:
        trajectory (List[Tuple[float, Dict[str, float]]]): The original trajectory, where each element is a tuple containing a timestamp and a dictionary of joint angles.
        desired_interval (float, optional): The desired time interval between resampled points. Defaults to 0.01.
        interp_type (str, optional): The type of interpolation to use ('linear', 'quadratic', 'cubic'). Defaults to 'linear'.

    Returns:
        List[Tuple[float, Dict[str, float]]]: The resampled trajectory with interpolated joint angles at the specified time intervals.
    """
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

    Args:
        p_start: Initial position.
        p_end: Desired end position.
        duration: Total duration from start to end.
        t: Current time (within 0 to duration).
        interp_type: Type of interpolation ('linear', 'quadratic', 'cubic').

    Returns:
        Position at time t.
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
    """Performs a binary search on a sorted array to find the index of a target value.

    Args:
        arr (ArrayType): A sorted array of numbers.
        t (ArrayType | float): The target value to search for.

    Returns:
        int: The index of the target value if found; otherwise, the index of the largest element less than the target.
    """
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
    """Interpolates an action value at a given time using specified interpolation method.

    Args:
        t (ArrayType | float): The time at which to interpolate the action.
        time_arr (ArrayType): An array of time points corresponding to the action values.
        action_arr (ArrayType): An array of action values corresponding to the time points.
        interp_type (str, optional): The type of interpolation to use. Defaults to "linear".

    Returns:
        The interpolated action value at time `t`.
    """
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


def resample_trajectory(
    trajectory: List[Tuple[float, Dict[str, float]]],
    desired_interval: float = 0.01,
    interp_type: str = "linear",
) -> List[Tuple[float, Dict[str, float]]]:
    """Resamples a trajectory of joint angles over time to a specified time interval using interpolation.

    Args:
        trajectory (List[Tuple[float, Dict[str, float]]]): A list of tuples where each tuple contains a timestamp and a dictionary of joint angles.
        desired_interval (float, optional): The desired time interval between resampled points. Defaults to 0.01.
        interp_type (str, optional): The type of interpolation to use ('linear', etc.). Defaults to 'linear'.

    Returns:
        List[Tuple[float, Dict[str, float]]]: A resampled list of tuples with timestamps and interpolated joint angles.
    """
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
