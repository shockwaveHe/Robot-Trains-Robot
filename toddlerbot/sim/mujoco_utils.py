import os
import pickle
import time
import warnings
from typing import Any, Dict, List

import mediapy as media
import mujoco
import mujoco.rollout
import mujoco.viewer
import numpy as np
import numpy.typing as npt
from moviepy.editor import VideoFileClip, clips_array

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="moviepy")
os.environ["MUJOCO_GL"] = "egl"  # For headless rendering


class MuJoCoViewer:
    def __init__(self, model: Any, data: Any):
        self.viewer = mujoco.viewer.launch_passive(model, data)
        self.model = model

    def visualize(self, data: Any, vis_flags: Dict[str, bool] = {"com": True}):
        with self.viewer.lock():
            self.viewer.user_scn.ngeom = 0
            if vis_flags["com"]:
                self.visualize_com(data)

        self.viewer.sync()

    def visualize_com(self, data: Any):
        i = self.viewer.user_scn.ngeom
        com_pos = np.array(data.body(0).subtree_com, dtype=np.float32)
        mujoco.mjv_initGeom(
            self.viewer.user_scn.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.01, 0.01, 0.01]),  # Adjust size of the sphere
            pos=com_pos,
            mat=np.eye(3).flatten(),
            rgba=[1, 0, 0, 1],
        )
        self.viewer.user_scn.ngeom = i + 1

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
                d = mujoco.MjData(self.model)
                d.qpos, d.qvel = qpos, qvel
                mujoco.mj_forward(self.model, d)
                self.renderer.update_scene(d, camera=camera)
                video_frames.append(self.renderer.render())

            media.write_video(video_path, video_frames, fps=1.0 / dt / render_every)
            video_paths.append(video_path)

        # Delay to ensure the video files are fully written
        time.sleep(1)

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
