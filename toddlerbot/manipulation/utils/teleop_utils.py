
import numpy as np
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R

def yaml_table_2_dict(yaml_table):
    """
    Convert a yaml table to a dictionary
    """
    yaml_dict = {}
    for key, value in yaml_table.items():
        if key in yaml_dict.keys():
            for sub_key, sub_value in value.items():
                yaml_dict[key].append(sub_value)
        else:
            yaml_dict[key] = []
            for sub_key, sub_value in value.items():
                yaml_dict[key].append(sub_value)
    return yaml_dict

def trans_unity_2_robot(translation, rotation, is_quat=True):
    """
    translation: np.array, [x, y, z]
    rotation: np.array, [x, y, z, w]
    Transform the coordinate convention from unity to robot
    """

    if is_quat:
        rotation = R.from_quat(rotation, scalar_first=False).as_matrix()
        
    R_ = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, 1,  0]
    ])
    # left-handed to right-handed
    translation = R_ @ translation
    rotation = R_ @ rotation @ R_.T
    if is_quat:
        rotation = R.from_matrix(rotation).as_quat()
    return translation, rotation

def R_x(degree):
    """
    Rotation matrix around x axis
    """
    rad = np.deg2rad(degree)
    return np.array([
        [1, 0, 0],
        [0, np.cos(rad), -np.sin(rad)],
        [0, np.sin(rad), np.cos(rad)]
    ])

def R_y(degree):
    """
    Rotation matrix around y axis
    """
    rad = np.deg2rad(degree)
    return np.array([
        [np.cos(rad), 0, np.sin(rad)],
        [0, 1, 0],
        [-np.sin(rad), 0, np.cos(rad)]
    ])

def R_z(degree):
    """
    Rotation matrix around z axis
    """
    rad = np.deg2rad(degree)
    return np.array([
        [np.cos(rad), -np.sin(rad), 0],
        [np.sin(rad), np.cos(rad), 0],
        [0, 0, 1]
    ])


def retarget_orientation(initial_rots, current_rot_src, initial_order, target_order):
    """
    Retarget the orientation of the robot
    all input in matrix form
    """
    initial_rot_src = initial_rots[0]
    initial_rot_target = initial_rots[1]

    # initial_rot_src @ initial_rot_offset = initial_rot_target
    initial_rot_offset = initial_rot_src.T @ initial_rot_target

    # initial_rot_src @ delta_rot = current_rot_src
    delta_rot = initial_rot_src.T @ current_rot_src

    delta_rot_euler = R.from_matrix(delta_rot).as_euler(initial_order, degrees=True)

    delta_rot_target = np.eye(3)
    
    function_dict = {
        'x': R_x,
        'y': R_y,
        'z': R_z
    }
    # e.g. initial_order = 'xyz', target_order = 'yxz'
    # target_order = R_y(x) @ R_x(y) @ R_z(z)
    for i in range(3):
        delta_rot_target = delta_rot_target @ function_dict[target_order[i]](delta_rot_euler[initial_order.index(target_order[i])])
    
    target_rot = initial_rot_target @ delta_rot_target

    return target_rot