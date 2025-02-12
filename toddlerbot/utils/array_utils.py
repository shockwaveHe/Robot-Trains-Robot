import os
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import scipy

USE_JAX = os.getenv("USE_JAX", "false").lower() == "true"

array_lib = jnp if USE_JAX else np
ArrayType = jax.Array | npt.NDArray[np.float32]
expm = jax.scipy.linalg.expm if USE_JAX else scipy.linalg.expm


def inplace_update(
    array: ArrayType,
    idx: int | slice | ArrayType | tuple[int | slice | ArrayType, ...],
    value: Any,
) -> ArrayType:
    """Updates the specified elements of an array in place with a given value.

    Args:
        array (ArrayType): The array to be updated.
        idx (int | slice | ArrayType | tuple[int | slice | ArrayType, ...]): The indices of the elements to update. Can be an integer, slice, array, or a tuple of these.
        value (Any): The value to set at the specified indices.

    Returns:
        ArrayType: The updated array.
    """
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
    """Performs an in-place addition to an array at specified indices.

    Args:
        array (ArrayType): The array to be updated.
        idx (int | slice | tuple[int | slice, ...]): The index or indices where the addition should occur.
        value (Any): The value to add to the specified indices.

    Returns:
        ArrayType: The updated array after performing the in-place addition.
    """
    if USE_JAX:
        return array.at[idx].add(value)  # type: ignore
    else:
        array[idx] += value
        return array


def conditional_update(
    condition: bool | jax.Array | npt.NDArray[np.bool_],
    true_func: Callable[[], ArrayType],
    false_func: Callable[[], ArrayType],
) -> ArrayType:
    """
    Performs a conditional update using `jax.lax.cond` if USE_JAX is True, or
    a standard if-else statement for NumPy.

    Args:
        condition: The condition to check.
        true_func: Function to execute if the condition is True.
        false_func: Function to execute if the condition is False.

    Returns:
        The result of true_func if condition is True, otherwise the result of false_func.
    """
    if USE_JAX:
        # Use jax.lax.cond to perform the conditional update
        return jax.lax.cond(condition, true_func, false_func)
    else:
        # Use a standard if-else for NumPy
        return true_func() if condition else false_func()


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
            update_step,  # type: ignore
            (x, u),
            jnp.arange(*index_range),
        )
    else:
        # Use a standard loop for NumPy
        for i in range(*index_range):
            (x, u), _ = update_step((x, u), i)

        final_traj_x = x

    return final_traj_x
