import os
import pickle
import queue
from typing import Any, Dict, List

import mediapy as media
import mujoco  # type: ignore
import mujoco.rollout  # type: ignore
import mujoco.viewer  # type: ignore
import numpy as np
import numpy.typing as npt
from moviepy.editor import VideoFileClip, clips_array  # type: ignore


class MuJoCoViewer:
    def __init__(self, model: Any, data: Any):
        self.viewer = mujoco.viewer.launch_passive(model, data)  # type: ignore
        self.model = model

    def visualize(self, data: Any, vis_data: Dict[str, Any] = {}):
        # with self.viewer.lock():
        #     self.viewer.user_scn.ngeom = 0  # type: ignore
        #     if "foot_steps" in vis_data:
        #         self.vis_foot_steps(vis_data["foot_steps"])
        #     if "com_ref_traj" in vis_data:
        #         self.vis_com_ref_traj(vis_data["com_ref_traj"])
        #     if "path" in vis_data:
        #         self.vis_path(vis_data["path"])
        #     if "torso" in vis_data:
        #         self.vis_torso(data)

        self.viewer.sync()

    # def vis_foot_steps(self, foot_steps):
    #     i = self.viewer.user_scn.ngeom
    #     for foot_step in foot_steps:
    #         if foot_step.support_leg == "both":
    #             continue

    #         mujoco.mjv_initGeom(
    #             self.viewer.user_scn.geoms[i],
    #             type=mujoco.mjtGeom.mjGEOM_LINEBOX,
    #             size=[
    #                 self.foot_size[0] / 2,
    #                 self.foot_size[1] / 2,
    #                 self.foot_size[2] / 2,
    #             ],
    #             pos=np.array(
    #                 [
    #                     foot_step.position[0],
    #                     foot_step.position[1],
    #                     self.foot_size[2] / 2,
    #                 ]
    #             ),
    #             mat=euler2mat(0, 0, foot_step.position[2]).flatten(),
    #             rgba=(
    #                 [0, 0, 1, 1] if foot_step.support_leg == "left" else [0, 1, 0, 1]
    #             ),
    #         )
    #         i += 1
    #     self.viewer.user_scn.ngeom = i

    # def vis_com_ref_traj(self, com_ref_traj):
    #     i = self.viewer.user_scn.ngeom
    #     for com_pos in com_ref_traj:
    #         mujoco.mjv_initGeom(
    #             self.viewer.user_scn.geoms[i],
    #             type=mujoco.mjtGeom.mjGEOM_SPHERE,
    #             size=np.array([0.001, 0.001, 0.001]),
    #             pos=np.array([com_pos[0], com_pos[1], 0.005]),
    #             mat=np.eye(3).flatten(),
    #             rgba=[1, 0, 0, 1],
    #         )
    #         i += 1
    #     self.viewer.user_scn.ngeom = i

    # def vis_path(self, path):
    #     i = self.viewer.user_scn.ngeom
    #     for j in range(len(path) - 1):
    #         mujoco.mjv_initGeom(
    #             self.viewer.user_scn.geoms[i],
    #             type=mujoco.mjtGeom.mjGEOM_LINE,
    #             size=np.array([1, 1, 1]),
    #             pos=np.array([0, 0, 0]),
    #             mat=np.eye(3).flatten(),
    #             rgba=[0, 0, 0, 1],
    #         )
    #         mujoco.mjv_connector(
    #             self.viewer.user_scn.geoms[i],
    #             mujoco.mjtGeom.mjGEOM_LINE,
    #             100,
    #             np.array([*path[j], 0.0]),
    #             np.array([*path[j + 1], 0.0]),
    #         )
    #         i += 1
    #     self.viewer.user_scn.ngeom = i

    # def vis_torso(self, data):
    #     i = self.viewer.user_scn.ngeom
    #     torso_pos = data.site("torso").xpos
    #     torso_mat = data.site("torso").xmat
    #     mujoco.mjv_initGeom(
    #         self.viewer.user_scn.geoms[i],
    #         type=mujoco.mjtGeom.mjGEOM_ARROW,
    #         size=np.array([0.005, 0.005, 0.15]),
    #         pos=torso_pos,
    #         mat=torso_mat,
    #         rgba=[1, 0, 0, 1],
    #     )
    #     self.viewer.user_scn.ngeom = i + 1

    def close(self):
        self.viewer.close()


class MuJoCoRenderer:
    def __init__(self, model: Any, height: int = 360, width: int = 640):
        self.model = model
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.anim_data: Dict[str, Any] = {}
        self.qpos_data: List[Any] = []
        self.qvel_data: List[Any] = []

    def visualize(self, data: Any, vis_data: Dict[str, Any] = {}):
        self.anim_pose_callback(data)
        self.qpos_data.append(data.qpos.copy())
        self.qvel_data.append(data.qvel.copy())

    def save_recording(
        self,
        exp_folder_path: str,
        dt: float,
        render_every: int,
        name: str = "mujoco.mp4",
        dump_data: bool = False,
    ):
        if dump_data:
            anim_data_path = os.path.join(exp_folder_path, "anim_data.pkl")
            with open(anim_data_path, "wb") as f:
                pickle.dump(self.anim_data, f)

        # Define paths for each camera's video
        video_paths: List[str] = []
        # Render and save videos for each camera
        for camera in ["perspective", "side", "top", "front"]:
            video_path = os.path.join(exp_folder_path, f"{camera}.mp4")
            video_frames: List[npt.NDArray[np.float32]] = []
            for qpos, qvel in zip(
                self.qpos_data[::render_every], self.qvel_data[::render_every]
            ):
                d = mujoco.MjData(self.model)  # type: ignore
                d.qpos, d.qvel = qpos, qvel
                mujoco.mj_forward(self.model, d)  # type: ignore
                self.renderer.update_scene(d, camera=camera)  # type: ignore
                video_frames.append(self.renderer.render())  # type: ignore

            media.write_video(video_path, video_frames, fps=1.0 / dt / render_every)
            video_paths.append(video_path)

        # Load the video clips using moviepy
        clips = [VideoFileClip(path) for path in video_paths]
        # Arrange the clips in a 2x2 grid
        final_video = clips_array([[clips[0], clips[1]], [clips[2], clips[3]]])
        # Save the final concatenated video
        final_video.write_videofile(os.path.join(exp_folder_path, name))

    def anim_pose_callback(self, data: Any):
        for i in range(self.model.nbody):
            body_name = data.body(i).name
            pos = data.body(i).xpos.copy()
            quat = data.body(i).xquat.copy()

            data_tuple = (data.time, pos, quat)
            if body_name in self.anim_data:
                self.anim_data[body_name].append(data_tuple)
            else:
                self.anim_data[body_name] = [data_tuple]

    def close(self):
        self.renderer.close()
