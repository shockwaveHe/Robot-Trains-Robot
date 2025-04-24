import mujoco as mj
from scipy.spatial.transform import Rotation as R
import numpy as np

def draw_frame(
    pos,
    mat,
    v,
    size,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
):
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        mj.mjv_initGeom(
            v.user_scn.geoms[v.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos,
            mat=mat.flatten(),
            rgba=rgba_list[i],
        )
        fix = orientation_correction.as_matrix()
        mj.mjv_connector(
            v.user_scn.geoms[v.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos,
            to=pos + size * (mat @ fix)[:, i],
        )
        v.user_scn.ngeom += 1

def draw_frame_batch(
    poses,
    rots,
    v,
    sizes,
    orientation_corrections,
):
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    v.user_scn.ngeom = 0
    for frame in range(len(poses)):
        for i in range(3):
            mj.mjv_initGeom(
                v.user_scn.geoms[v.user_scn.ngeom],
                type=mj.mjtGeom.mjGEOM_ARROW,
                size=[0.01, 0.01, 0.01],
                pos=poses[frame],
                mat=rots[frame].flatten(),
                rgba=rgba_list[i],
            )
            fix = np.eye(3) # orientation_corrections[frame].as_matrix()
            mj.mjv_connector(
                v.user_scn.geoms[v.user_scn.ngeom],
                type=mj.mjtGeom.mjGEOM_ARROW,
                width=0.005,
                from_=poses[frame],
                to=poses[frame] + sizes[frame] * (rots[frame] @ fix)[:, i],
            )
            v.user_scn.ngeom += 1