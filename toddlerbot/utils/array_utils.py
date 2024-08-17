import os
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import scipy  # type: ignore

USE_JAX = os.getenv("USE_JAX", "True").lower() == "true"

array_lib = jnp if USE_JAX else np
ArrayType = jax.Array | npt.NDArray[np.float32]
expm = jax.scipy.linalg.expm if USE_JAX else scipy.linalg.expm


def inplace_update(
    array: ArrayType, idx: int | slice | tuple[int | slice, ...], value: Any
) -> ArrayType:
    """Updates the array at the specified index with the given value."""
    if USE_JAX:
        # JAX requires using .at[idx].set(value) for in-place updates
        return array.at[idx].set(value)  # type: ignore
    else:
        # Numpy allows direct in-place updates
        array[idx] = value
        return array


def inplace_add(
    array: ArrayType, idx: int | slice | tuple[int | slice, ...], value: Any
) -> ArrayType:
    """Performs an in-place addition to the array at the specified index."""
    if USE_JAX:
        return array.at[idx].add(value)  # type: ignore
    else:
        array[idx] += value
        return array


def interpolate(
    p_start: ArrayType | float,
    p_end: ArrayType | float,
    duration: float,
    t: float,
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
                    interpolated_joint_angles[joint_name] = p_interp  # type: ignore
                resampled_trajectory.append((t0 + t, interpolated_joint_angles))
        else:
            # Interval is fine, keep the original point
            resampled_trajectory.append((t0, joint_angles_0))

    resampled_trajectory.append(trajectory[-1])

    return resampled_trajectory
