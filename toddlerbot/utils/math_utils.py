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


def quatwxyz2mat(quat):
    return quat2mat(quat)


def mat2quatwxyz(mat):
    return mat2quat(mat)
