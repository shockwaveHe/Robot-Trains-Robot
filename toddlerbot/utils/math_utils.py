from dataclasses import is_dataclass

import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat


def round_floats(obj, precision):
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
        return np.round(obj, decimals=precision)
    elif is_dataclass(obj):
        return type(obj)(
            **{
                field.name: round_floats(getattr(obj, field.name), precision)
                for field in obj.__dataclass_fields__.values()
            }
        )

    return obj


def quatxyzw2mat(quat):
    return quat2mat([quat[3], quat[0], quat[1], quat[2]])


def mat2quatxyzw(mat):
    quat = mat2quat(mat)
    return [quat[1], quat[2], quat[3], quat[0]]


def interpolate(p_start, p_end, delta_t, t, interp_type="linear"):
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
