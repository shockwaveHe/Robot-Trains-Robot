import time
from dataclasses import is_dataclass

import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat

from toddlerbot.utils.misc_utils import precise_sleep


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
        return list(np.round(obj, decimals=precision))
    elif isinstance(obj, dict):
        return {k: round_floats(v, precision) for k, v in obj.items()}
    elif is_dataclass(obj):
        return type(obj)(
            **{
                field.name: round_floats(getattr(obj, field.name), precision)
                for field in obj.__dataclass_fields__.values()
            }
        )

    return obj


def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
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


def interpolate_pos(
    set_pos, pos_start, pos, delta_t, interp_type, actuator_type, sleep_time=0.0
):
    time_start = time.time()
    time_curr = 0
    counter = 0
    while time_curr <= delta_t:
        time_curr = time.time() - time_start
        pos_interp = interpolate(
            pos_start, pos, delta_t, time_curr, interp_type=interp_type
        )
        set_pos(pos_interp)
        # log(f"Setting position: {pos_interp}", header=actuator_type, level="debug")

        time_elapsed = time.time() - time_start - time_curr
        time_until_next_step = sleep_time - time_elapsed
        if time_until_next_step > 0:
            precise_sleep(time_until_next_step)

        counter += 1


def resample_trajectory(trajectory, desired_interval=0.01, interp_type="linear"):
    resampled_trajectory = []
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
                interpolated_joint_angles = {}
                for joint_name, p_start in joint_angles_0.items():
                    p_end = joint_angles_1[joint_name]
                    interpolated_joint_angles[joint_name] = interpolate(
                        p_start, p_end, delta_t, t, interp_type
                    )
                resampled_trajectory.append((t0 + t, interpolated_joint_angles))
        else:
            # Interval is fine, keep the original point
            resampled_trajectory.append((t0, joint_angles_0))

    resampled_trajectory.append(trajectory[-1])

    return resampled_trajectory
