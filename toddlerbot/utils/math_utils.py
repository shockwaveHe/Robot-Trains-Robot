import time
from dataclasses import is_dataclass
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
import numpy.typing as npt

from toddlerbot.utils.misc_utils import precise_sleep


def get_random_sine_signal_config(
    duration: float,
    control_dt: float,
    mean: float,
    frequency_range: List[float],
    amplitude_range: List[float],
):
    frequency = np.random.uniform(*frequency_range)  # type: ignore
    amplitude = np.random.uniform(*amplitude_range)  # type: ignore

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
    )
    signal = sine_signal_config["mean"] + sine_signal_config["amplitude"] * np.sin(
        2 * np.pi * sine_signal_config["frequency"] * t
    )
    return t, signal


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
        return type(obj)(round_floats(x, precision) for x in obj)  # type: ignore
    elif isinstance(obj, np.ndarray):
        return list(np.round(obj, decimals=precision))  # type: ignore
    elif isinstance(obj, dict):
        return {k: round_floats(v, precision) for k, v in obj.items()}  # type: ignore
    elif is_dataclass(obj):
        return type(obj)(
            **{
                field.name: round_floats(getattr(obj, field.name), precision)
                for field in obj.__dataclass_fields__.values()
            }
        )

    return obj


def quaternion_to_euler_array(
    quat: Iterable[float], order: str = "wxyz"
) -> npt.NDArray[np.float32]:
    if order == "xyzw":
        x, y, z, w = quat
    else:
        w, x, y, z = quat

    # Roll (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    euler_angles = np.array([roll_x, pitch_y, yaw_z])
    euler_angles[euler_angles > np.pi] -= 2 * np.pi

    return euler_angles


def interpolate(
    p_start: Union[npt.NDArray[np.float32], float],
    p_end: Union[npt.NDArray[np.float32], float],
    delta_t: float,
    t: float,
    interp_type: str = "linear",
) -> Union[npt.NDArray[np.float32], float]:
    """
    Interpolate position at time t using specified interpolation type.

    Parameters:
    - p_start: Initial position.
    - p_end: Desired end position.
    - delta_t: Total duration from start to end.
    - t: Current time (within 0 to delta_t).
    - interp_type: Type of interpolation ('linear', 'quadratic', 'cubic').

    Returns:
    - Position at time t.
    """
    if t <= 0:
        return p_start

    if t >= delta_t:
        return p_end

    if interp_type == "linear":
        return p_start + (p_end - p_start) * (t / delta_t)
    elif interp_type == "quadratic":
        a = (-p_end + p_start) / delta_t**2
        b = (2 * p_end - 2 * p_start) / delta_t
        return a * t**2 + b * t + p_start
    elif interp_type == "cubic":
        a = (2 * p_start - 2 * p_end) / delta_t**3
        b = (3 * p_end - 3 * p_start) / delta_t**2
        return a * t**3 + b * t**2 + p_start
    else:
        raise ValueError("Unsupported interpolation type: {}".format(interp_type))


def interpolate_arr(
    t: str,
    time_arr: npt.NDArray[np.float32],
    action_arr: npt.NDArray[np.float32],
    interp_type: str = "linear",
):
    if t <= time_arr[0]:
        return action_arr[0]
    elif t >= time_arr[-1]:
        return action_arr[-1]

    # Find the segment containing current_time
    for i in range(len(time_arr) - 1):
        if time_arr[i] <= t < time_arr[i + 1]:
            p_start = action_arr[i]
            p_end = action_arr[i + 1]
            delta_t = time_arr[i + 1] - time_arr[i]
            return interpolate(p_start, p_end, delta_t, t - time_arr[i], interp_type)

    # Fallback (shouldn't be reached)
    return action_arr[-1]


def interpolate_pos(
    set_pos: Callable[[npt.NDArray[np.float32]], None],
    pos_start: npt.NDArray[np.float32],
    pos: npt.NDArray[np.float32],
    delta_t: float,
    interp_type: str,
    sleep_time: float = 0.0,
):
    time_start = time.time()
    time_curr = 0
    counter = 0
    while time_curr <= delta_t:
        time_curr = time.time() - time_start
        pos_interp = interpolate(
            pos_start, pos, delta_t, time_curr, interp_type=interp_type
        )
        set_pos(pos_interp)  # type: ignore

        time_elapsed = time.time() - time_start - time_curr
        time_until_next_step = sleep_time - time_elapsed
        if time_until_next_step > 0:
            precise_sleep(time_until_next_step)

        counter += 1


def resample_trajectory(
    trajectory: List[Tuple[float, Dict[str, float]]],
    desired_interval: float = 0.01,
    interp_type: str = "linear",
) -> List[Tuple[float, Dict[str, float]]]:
    resampled_trajectory: List[Tuple[float, Dict[str, float]]] = []
    for i in range(len(trajectory) - 1):
        t0, joint_angles_0 = trajectory[i]
        t1, joint_angles_1 = trajectory[i + 1]
        delta_t = t1 - t0

        # Add an epislon to the desired interval to avoid floating point errors
        if delta_t > desired_interval + 1e-6:
            # More points needed, interpolate
            num_steps = int(delta_t / desired_interval)
            for j in range(num_steps):
                t = j * desired_interval
                interpolated_joint_angles: Dict[str, float] = {}
                for joint_name, p_start in joint_angles_0.items():
                    p_end = joint_angles_1[joint_name]
                    p_interp = interpolate(p_start, p_end, delta_t, t, interp_type)
                    interpolated_joint_angles[joint_name] = p_interp  # type: ignore
                resampled_trajectory.append((t0 + t, interpolated_joint_angles))
        else:
            # Interval is fine, keep the original point
            resampled_trajectory.append((t0, joint_angles_0))

    resampled_trajectory.append(trajectory[-1])

    return resampled_trajectory
