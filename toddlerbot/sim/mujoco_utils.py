import os
import pickle
import queue
from typing import Any, Dict

import mediapy as media
import mujoco  # type: ignore
import mujoco.rollout  # type: ignore
import mujoco.viewer  # type: ignore
import numpy as np
import numpy.typing as npt


class MuJoCoViewer:
    def __init__(self, model: Any, data: Any):
        self.viewer = mujoco.viewer.launch_passive(model, data)  # type: ignore

    def visualize(self, model: Any, data: Any, vis_data: Dict[str, Any] = {}):
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
    def __init__(self, model: Any, data: Any, height: int = 720, width: int = 1280):
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.anim_data: Dict[str, Any] = {}
        self.video_frames: list[npt.NDArray[np.float32]] = []

    def visualize(self, model: Any, data: Any, vis_data: Dict[str, Any] = {}):
        self.anim_pose_callback(model, data)

        self.renderer.update_scene(data)  # type: ignore
        self.video_frames.append(self.renderer.render())  # type: ignore

    def save_recording(
        self,
        exp_folder_path: str,
        dt: float,
        render_every: int,
        name: str = "mujoco.mp4",
    ):
        anim_data_path = os.path.join(exp_folder_path, "anim_data.pkl")
        with open(anim_data_path, "wb") as f:
            pickle.dump(self.anim_data, f)

        video_path = os.path.join(exp_folder_path, name)
        media.write_video(
            video_path, self.video_frames[::render_every], fps=1 / dt / render_every
        )

    def anim_pose_callback(self, model: Any, data: Any):
        for i in range(model.nbody):
            body_name = model.body(i).name
            pos = data.body(i).xpos.copy()
            quat = data.body(i).xquat.copy()

            data_tuple = (data.time, pos, quat)
            if body_name in self.anim_data:
                self.anim_data[body_name].append(data_tuple)
            else:
                self.anim_data[body_name] = [data_tuple]

    def close(self):
        self.renderer.close()


class MuJoCoController:
    def __init__(self):
        self.command_queue = queue.Queue()  # type: ignore

    def add_command(self, motor_ctrls: Dict[str, float] | npt.NDArray[np.float32]):
        self.command_queue.put(motor_ctrls)  # type: ignore

    def process_commands(self, model: Any, data: Any):
        while not self.command_queue.empty():  # type: ignore
            motor_ctrls = self.command_queue.get()  # type: ignore
            if isinstance(motor_ctrls, dict):
                for name, ctrl in motor_ctrls.items():  # type: ignore
                    data.actuator(name).ctrl = ctrl
            else:
                for i, ctrl in enumerate(motor_ctrls):  # type: ignore
                    data.actuator(i).ctrl = ctrl
