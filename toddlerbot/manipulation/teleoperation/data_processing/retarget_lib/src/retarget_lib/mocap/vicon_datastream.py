import numpy as np
import pyvicon_datastream as pv
from scipy.spatial.transform import Rotation as R


SEGMENT_NAMES = {}

GLOBAL_Z = np.array([0, 0, 1])


def cache_segment_names(client, subject):
    if subject in SEGMENT_NAMES and SEGMENT_NAMES[subject]:
        return
    result = []
    for segment in range(client.get_segment_count(subject)):
        name = client.get_segment_name(subject, segment)
        result.append(name)
    SEGMENT_NAMES[subject] = result


def get_subject_names(client):
    return [client.get_subject_name(i) for i in range(client.get_subject_count())]


def setup_vicon(host):
    client = pv.PyViconDatastream()
    client.connect(host)

    client.set_stream_mode(pv.StreamMode.ServerPush)
    client.enable_segment_data()
    client.enable_marker_data()

    print("Waiting for first frame...")
    while client.get_frame() == pv.Result.NoFrame:
        pass

    return client


def setup_vicon_cache(client, subject=None):
    if subject is not None:
        cache_segment_names(client, subject)


def get_vicon_data(client, subject):
    """
    Returns the global positions and orientations of every segment in the Vicon skeleton.

    Make sure to call setup_vicon_cache before calling this function!
    """
    client.get_frame()
    result = {}
    for segment in SEGMENT_NAMES[subject]:
        pos = client.get_segment_global_translation(subject, segment)
        rot = client.get_segment_global_quaternion(subject, segment)
        result[segment] = (
            0.001 * np.array(pos),
            np.array(rot),  # Already comes in MuJoCo quaternion convention
        )
    result["RHEL"] = 0.001 * np.array(
        client.get_marker_global_translation(subject, "RHEL")
    )
    result["LHEL"] = 0.001 * np.array(
        client.get_marker_global_translation(subject, "LHEL")
    )

    # Modify foot orientation
    right_foot_y = result["RightToeBase"][0] - result["RHEL"]
    right_foot_y = right_foot_y / np.linalg.norm(right_foot_y)
    right_foot_z = GLOBAL_Z - np.dot(GLOBAL_Z, right_foot_y) * right_foot_y
    right_foot_z = right_foot_z / np.linalg.norm(right_foot_z)
    right_foot_x = np.cross(right_foot_y, right_foot_z)
    rot_mat = np.array([right_foot_x, right_foot_y, right_foot_z]).T
    result["RightFootMod"] = (
        result["RightFoot"][0],
        R.from_matrix(rot_mat).as_quat(scalar_first=True),
    )

    left_foot_y = result["LeftToeBase"][0] - result["LHEL"]
    left_foot_y = left_foot_y / np.linalg.norm(left_foot_y)
    left_foot_z = GLOBAL_Z - np.dot(GLOBAL_Z, left_foot_y) * left_foot_y
    left_foot_z = left_foot_z / np.linalg.norm(left_foot_z)
    left_foot_x = np.cross(left_foot_y, left_foot_z)
    rot_mat = np.array([left_foot_x, left_foot_y, left_foot_z]).T
    result["LeftFootMod"] = (
        result["LeftFoot"][0],
        R.from_matrix(rot_mat).as_quat(scalar_first=True),
    )

    return result
