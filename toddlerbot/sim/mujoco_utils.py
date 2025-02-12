import os
import pickle
import time
import warnings
from typing import Any, Dict, List

import cv2
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import mujoco.rollout
import mujoco.viewer
import numpy as np
import numpy.typing as npt
from moviepy.editor import VideoFileClip, clips_array

from toddlerbot.sim.robot import Robot

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="moviepy")


def mj_render(model, data, lib="plt"):
    """Renders a MuJoCo simulation scene using the specified library.

    Args:
        model: The MuJoCo model to be rendered.
        data: The simulation data associated with the model.
        lib (str): The library to use for rendering. Options are "plt" for matplotlib and any other value for OpenCV. Defaults to "plt".

    Raises:
        ValueError: If the specified library is not supported.
    """
    renderer = mujoco.Renderer(model)
    renderer.update_scene(data)
    pixels = renderer.render()

    if lib == "plt":
        plt.imshow(pixels)
        plt.show()
    else:
        pixels_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        cv2.imshow("Simulation", pixels_bgr)
        cv2.waitKey(1)  # This ensures the window updates without blocking


class MuJoCoViewer:
    """A class for visualizing MuJoCo simulation data."""

    def __init__(self, robot: Robot, model: Any, data: Any):
        """Initializes the class with a robot, model, and data, and sets up the viewer and foot geometry.

        Args:
            robot (Robot): The robot instance containing configuration and state information.
            model (Any): The model object representing the simulation environment.
            data (Any): The data object containing the simulation state.

        Attributes:
            robot (Robot): Stores the robot instance.
            model (Any): Stores the model object.
            viewer: Launches a passive viewer for the simulation using the provided model and data.
            foot_names (list of str): List of foot collision geometry names based on the robot's foot name.
            local_bbox_corners (np.ndarray): Local coordinates of the bounding box corners for the foot geometry.
        """
        self.robot = robot
        self.model = model
        self.viewer = mujoco.viewer.launch_passive(model, data)

        self.foot_names = [
            f"{self.robot.foot_name}_collision",
            f"{self.robot.foot_name}_2_collision",
        ]
        foot_geom_size = np.array(self.model.geom(self.foot_names[0]).size)
        # Define the local coordinates of the bounding box corners
        self.local_bbox_corners = np.array(
            [
                [0.0, -foot_geom_size[1], -foot_geom_size[2]],
                [0.0, -foot_geom_size[1], foot_geom_size[2]],
                [0.0, foot_geom_size[1], foot_geom_size[2]],
                [0.0, foot_geom_size[1], -foot_geom_size[2]],
            ]
        )

        # self.path_frame_mid = model.body("path_frame").mocapid[0]

    def visualize(
        self,
        data: Any,
        vis_flags: List[str] = ["com", "support_poly"],
    ):
        """Visualizes specified components of the data using the viewer.

        Args:
            data (Any): The data to be visualized.
            vis_flags (List[str], optional): A list of visualization flags indicating which components to visualize. Defaults to ["com", "support_poly"].
        """
        with self.viewer.lock():
            self.viewer.user_scn.ngeom = 0
            if "com" in vis_flags:
                self.visualize_com(data)
            if "support_poly" in vis_flags:
                self.visualize_support_poly(data)
            # if "path_frame" in vis_flags:
            #     self.visualize_path_frame(data)

        self.viewer.sync()

    def visualize_com(self, data: Any):
        """Visualize the center of mass (COM) of a given body in the simulation environment.

        This function adds a visual representation of the center of mass for a specified body
        to the simulation viewer. The COM is depicted as a small red sphere.

        Args:
            data (Any): The simulation data object containing information about the bodies,
                including their center of mass positions.
        """
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

    def visualize_support_poly(self, data: Any):
        """Visualizes the support polygon of the feet by drawing lines between the transformed bounding box corners in the world coordinates.

        Args:
            data (Any): The simulation data object containing geometric information about the feet.
        """
        i = self.viewer.user_scn.ngeom

        for foot_name in self.foot_names:
            foot_geom_pos = np.array(data.geom(foot_name).xpos)
            foot_geom_mat = np.array(data.geom(foot_name).xmat).reshape(3, 3)

            # Transform local bounding box corners to world coordinates
            world_bbox_corners = (
                foot_geom_mat @ self.local_bbox_corners.T
            ).T + foot_geom_pos

            for j in range(len(world_bbox_corners)):
                p1 = world_bbox_corners[j]
                p2 = world_bbox_corners[(j + 1) % len(world_bbox_corners)]
                p1[2] = 0.0
                p2[2] = 0.0

                # Create a line geometry
                mujoco.mjv_initGeom(
                    self.viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_LINE,
                    size=np.zeros(3),
                    pos=np.zeros(3),
                    mat=np.eye(3).flatten(),
                    rgba=[0, 0, 1, 1],
                )
                mujoco.mjv_connector(
                    self.viewer.user_scn.geoms[i],
                    mujoco.mjtGeom.mjGEOM_LINE,
                    2,
                    p1,
                    p2,
                )
                i += 1

        self.viewer.user_scn.ngeom = i

    # def visualize_path_frame(self, data: Any):
    #     i = self.viewer.user_scn.ngeom

    #     path_pos = data.mocap_pos[self.path_frame_mid]
    #     path_mat = quat2mat(data.mocap_quat[self.path_frame_mid]).reshape(3, 3)

    #     # Define axes in local coordinates
    #     axes = np.eye(3)  # X, Y, Z axes

    #     # Transform axes to world coordinates
    #     world_axes = path_mat @ axes + path_pos[:, None]

    #     # Define colors for axes: X (red), Y (green), Z (blue)
    #     colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]

    #     for j in range(3):
    #         p1 = path_pos
    #         p2 = world_axes[:, j]

    #         # Create a line for each axis
    #         mujoco.mjv_initGeom(
    #             self.viewer.user_scn.geoms[i],
    #             type=mujoco.mjtGeom.mjGEOM_LINE,
    #             size=np.zeros(3),
    #             pos=np.zeros(3),
    #             mat=np.eye(3).flatten(),
    #             rgba=colors[j],
    #         )
    #         mujoco.mjv_connector(
    #             self.viewer.user_scn.geoms[i],
    #             mujoco.mjtGeom.mjGEOM_LINE,
    #             2,
    #             p1,
    #             p2,
    #         )
    #         i += 1

    #     self.viewer.user_scn.ngeom = i

    def close(self):
        """Closes the viewer associated with the current instance."""
        self.viewer.close()


class MuJoCoRenderer:
    """A class for rendering MuJoCo simulation data and saving video recordings."""

    def __init__(self, model: Any, height: int = 360, width: int = 640):
        """Initializes the object with a given model and sets up a renderer.

        Args:
            model (Any): The model to be used for rendering.
            height (int, optional): The height of the rendering window. Defaults to 360.
            width (int, optional): The width of the rendering window. Defaults to 640.
        """
        self.model = model
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.anim_data: Dict[str, Any] = {}
        self.qpos_data: List[Any] = []
        self.qvel_data: List[Any] = []

    def visualize(self, data: Any, vis_data: Dict[str, Any] = {}):
        """Visualizes the given data by updating pose, position, and velocity information.

        This method processes the input data to update the animation pose and appends
        the position and velocity data to their respective lists for further visualization.

        Args:
            data (Any): The input data containing pose, position, and velocity information.
            vis_data (Dict[str, Any], optional): Additional visualization data. Defaults to an empty dictionary.
        """
        self.anim_pose_callback(data)
        self.qpos_data.append(data.qpos.copy())
        self.qvel_data.append(data.qvel.copy())

    def init_recording(self):
        self.anim_data: Dict[str, Any] = {}
        self.qpos_data: List[Any] = []
        self.qvel_data: List[Any] = []

    def save_recording(
        self,
        exp_folder_path: str,
        dt: float,
        render_every: int = 1,
        name: str = "mujoco.mp4",
        dump_data: bool = False,
        cameras: List[str] = ["perspective", "side", "top", "front"],
    ):
        """Saves a recording of the simulation from multiple camera angles and optionally dumps animation data.

        Args:
            exp_folder_path (str): The path to the folder where the recording and data will be saved.
            dt (float): The time step duration for rendering frames.
            render_every (int): The interval at which frames are rendered.
            name (str, optional): The name of the final video file. Defaults to "mujoco.mp4".
            dump_data (bool, optional): If True, dumps the animation data to a pickle file. Defaults to False.
        """
        if dump_data:
            anim_data_path = os.path.join(exp_folder_path, "anim_data.pkl")
            with open(anim_data_path, "wb") as f:
                pickle.dump(self.anim_data, f)

        # Define paths for each camera's video
        video_paths: List[str] = []
        # Render and save videos for each camera
        for camera in cameras:
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
            print(f"Saved video to {video_path}")

        # Delay to ensure the video files are fully written
        time.sleep(1)

        if len(cameras) == 4:
            # Load the video clips using moviepy
            clips = [VideoFileClip(path) for path in video_paths]
            # Arrange the clips in a 2x2 grid
            final_video = clips_array([[clips[0], clips[1]], [clips[2], clips[3]]])
            # Save the final concatenated video
            final_video.write_videofile(os.path.join(exp_folder_path, name))

    def anim_pose_callback(self, data: Any):
        """Processes animation pose data and updates the animation data dictionary.

        Args:
            data (Any): An object containing pose data for multiple bodies, including
                body names, positions, orientations, and a timestamp.

        Updates:
            self.anim_data (dict): A dictionary where each key is a body name and each
            value is a list of tuples. Each tuple contains a timestamp, position, and
            orientation for the corresponding body.
        """
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
        """Closes the renderer associated with the current instance.

        This method ensures that the renderer is properly closed and any resources
        associated with it are released.
        """
        self.renderer.close()
