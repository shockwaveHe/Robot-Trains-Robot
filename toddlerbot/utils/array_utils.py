import os
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import scipy  # type: ignore

USE_JAX = os.getenv("USE_JAX", "false").lower() == "true"

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


def loop_update(
    update_step: Callable[
        [Tuple[ArrayType, ArrayType], int],
        Tuple[Tuple[ArrayType, ArrayType], ArrayType],
    ],
    x: ArrayType,
    u: ArrayType,
    index_range: Tuple[int, int],
) -> ArrayType:
    """
    A general function to perform loop updates compatible with both JAX and NumPy.

    Args:
        N: Number of steps.
        traj_x: The state trajectory array.
        traj_u: The control input trajectory array.
        update_step: A function that defines how to update the state at each step.
        USE_JAX: A flag to determine whether to use JAX or NumPy.

    Returns:
        The updated trajectory array.
    """
    if USE_JAX:
        # Use jax.lax.scan for JAX-compatible looping
        (final_traj_x, _), _ = jax.lax.scan(
            update_step,
            (x, u),
            jnp.arange(*index_range),  # type: ignore
        )
    else:
        # Use a standard loop for NumPy
        for i in range(*index_range):
            (x, u), _ = update_step((x, u), i)

        final_traj_x = x

    return final_traj_x
