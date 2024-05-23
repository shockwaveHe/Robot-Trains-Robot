import os
import pickle
import queue
import threading
import time

import mediapy as media
import mujoco
import mujoco.rollout
import mujoco.viewer
import numpy as np
from transforms3d.euler import euler2mat

from toddlerbot.actuation import JointState
from toddlerbot.sim import BaseSim
from toddlerbot.utils.constants import GRAVITY, SIM_TIMESTEP
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.misc_utils import precise_sleep


class MuJoCoViewer:
    def __init__(self, robot, model, data):
        self.viewer = mujoco.viewer.launch_passive(model, data)
        self.foot_size = robot.foot_size

    def visualize(self, model, data, vis_data):
        if vis_data is not None:
            with self.viewer.lock():
                self.viewer.user_scn.ngeom = 0
                if "foot_steps" in vis_data:
                    self.vis_foot_steps(vis_data["foot_steps"])
                if "com_ref_traj" in vis_data:
                    self.vis_com_ref_traj(vis_data["com_ref_traj"])
                if "path" in vis_data:
                    self.vis_path(vis_data["path"])
                if "torso" in vis_data:
                    self.vis_torso(data)

        self.viewer.sync()

    def vis_foot_steps(self, foot_steps):
        i = self.viewer.user_scn.ngeom
        for foot_step in foot_steps:
            if foot_step.support_leg == "both":
                continue

            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_LINEBOX,
                size=[
                    self.foot_size[0] / 2,
                    self.foot_size[1] / 2,
                    self.foot_size[2] / 2,
                ],
                pos=np.array(
                    [
                        foot_step.position[0],
                        foot_step.position[1],
                        self.foot_size[2] / 2,
                    ]
                ),
                mat=euler2mat(0, 0, foot_step.position[2]).flatten(),
                rgba=(
                    [0, 0, 1, 1] if foot_step.support_leg == "left" else [0, 1, 0, 1]
                ),
            )
            i += 1
        self.viewer.user_scn.ngeom = i

    def vis_com_ref_traj(self, com_ref_traj):
        i = self.viewer.user_scn.ngeom
        for com_pos in com_ref_traj:
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.001, 0.001, 0.001]),
                pos=np.array([com_pos[0], com_pos[1], 0.005]),
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 1],
            )
            i += 1
        self.viewer.user_scn.ngeom = i

    def vis_path(self, path):
        i = self.viewer.user_scn.ngeom
        for j in range(len(path) - 1):
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=np.array([1, 1, 1]),
                pos=np.array([0, 0, 0]),
                mat=np.eye(3).flatten(),
                rgba=[0, 0, 0, 1],
            )
            mujoco.mjv_connector(
                self.viewer.user_scn.geoms[i],
                mujoco.mjtGeom.mjGEOM_LINE,
                100,
                np.array([*path[j], 0.0]),
                np.array([*path[j + 1], 0.0]),
            )
            i += 1
        self.viewer.user_scn.ngeom = i

    def vis_torso(self, data):
        i = self.viewer.user_scn.ngeom
        torso_pos = data.site("torso").xpos
        torso_mat = data.site("torso").xmat
        mujoco.mjv_initGeom(
            self.viewer.user_scn.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=np.array([0.005, 0.005, 0.15]),
            pos=torso_pos,
            mat=torso_mat,
            rgba=[1, 0, 0, 1],
        )
        self.viewer.user_scn.ngeom = i + 1

    def close(self):
        self.viewer.close()


class MuJoCoRenderer:
    def __init__(self, robot, model, data, height=720, width=1280, frame_rate=24):
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.frame_rate = frame_rate
        self.anim_data = {}
        self.video_frames = []

    def visualize(self, model, data, vis_data):
        if len(self.video_frames) < data.time * self.frame_rate:
            self.anim_pose_callback(model, data)

            self.renderer.update_scene(data)
            self.video_frames.append(self.renderer.render())

    def save_recording(self, exp_folder_path):
        anim_data_path = os.path.join(exp_folder_path, "anim_data.pkl")
        with open(anim_data_path, "wb") as f:
            pickle.dump(self.anim_data, f)

        video_path = os.path.join(exp_folder_path, "mujoco.mp4")
        media.write_video(video_path, self.video_frames, fps=self.frame_rate)

    def anim_pose_callback(self, model, data):
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
        pass


class MuJoCoController:
    def __init__(self):
        self.command_queue = queue.Queue()

    def add_command(self, joint_ctrls):
        self.command_queue.put(joint_ctrls)

    def process_commands(self, model, data):
        while not self.command_queue.empty():
            joint_ctrls = self.command_queue.get()
            if isinstance(joint_ctrls, dict):
                for name, ctrl in joint_ctrls.items():
                    data.actuator(f"{name}_act").ctrl = ctrl
            else:
                for i, ctrl in enumerate(joint_ctrls):
                    data.actuator(i).ctrl = ctrl


class MuJoCoSim(BaseSim):
    def __init__(
        self, robot, xml_str=None, assets=None, xml_path=None, fixed: bool = False
    ):
        """Initialize the MuJoCo simulation environment."""
        super().__init__()

        self.robot = robot

        self.name = "mujoco"
        if xml_str is not None and assets is not None:
            self.model = mujoco.MjModel.from_xml_string(xml_str, assets)
        elif xml_path is not None:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            xml_path = find_robot_file_path(robot.name, suffix="_scene.xml")
            self.model = mujoco.MjModel.from_xml_path(xml_path)

        self.model.opt.timestep = SIM_TIMESTEP
        self.data = mujoco.MjData(self.model)

        self.controller = MuJoCoController()

        if not fixed:
            self.put_robot_on_ground()

        self.thread = None
        self.stop_event = threading.Event()

    def put_robot_on_ground(self, z_offset: float = 0.02):
        lowest_z = float("inf")

        mujoco.mj_kinematics(self.model, self.data)
        # Iterate through all body parts to find the lowest point
        for i in range(self.model.nbody):
            if self.data.body(i).name == "world":
                continue
            # To correpond to the PyBullet code, we use xipos instead of xpos
            body_pos = self.data.body(i).xipos
            lowest_z = min(lowest_z, body_pos[2])

        body_link_name = self.robot.config.canonical_name2link_name["body_link"]
        base_pos = self.data.body(body_link_name).xpos
        desired_z = base_pos[2] - lowest_z + z_offset
        if lowest_z < 0:
            raise ValueError(
                "Robot is below the ground.\n"
                + f"Change the z value of {body_link_name} to be {desired_z}"
            )
        elif lowest_z > z_offset:
            raise ValueError(
                "Robot is too high above the ground.\n"
                + f" Change the z value of {body_link_name} as {desired_z}"
            )

    def _compute_dmom(self):
        if not hasattr(self, "t_last"):
            self.last_mom = np.zeros(3)
            self.last_angmom = np.zeros(3)
            self.t_last = self.data.time
        else:
            # Calculate current momentum and angular momentum
            mom = np.zeros(3)
            angmom = np.zeros(3)
            for i in range(self.model.nbody):
                body_name = self.model.body(i).name
                if body_name == "world":
                    continue

                body_mass = self.model.body(i).mass
                body_inertia = self.model.body(i).inertia
                body_pos = self.data.body(i).xipos  # Position
                body_xmat = self.data.body(i).xmat.reshape(3, 3)  # Rotation matrix
                # the translation component comes after the rotation component
                body_vel = self.data.body(i).cvel[3:]  # Linear Velocity
                body_ang_vel = self.data.body(i).cvel[:3]  # Angular velocity

                # Eq. 3.63-3.66 on p.94 in "Introduction to Humanoid Robotics"
                P = body_mass * body_vel
                L = (
                    np.cross(body_pos, P)
                    + body_xmat @ np.diag(body_inertia) @ body_xmat.T @ body_ang_vel
                )
                mom += P
                angmom += L

            t_curr = self.data.time
            dt = t_curr - self.t_last

            self.dP = (mom - self.last_mom) / dt
            self.dL = (angmom - self.last_angmom) / dt

            # Update last states
            self.last_mom = mom
            self.last_angmom = angmom
            self.t_last = t_curr

    def get_link_pos(self, link_name: str):
        mujoco.mj_kinematics(self.model, self.data)
        link_pos = self.data.body(link_name).xpos
        return np.array(link_pos)

    def get_link_quat(self, link_name: str):
        mujoco.mj_kinematics(self.model, self.data)
        link_quat = self.data.body(link_name).xquat
        return np.array(link_quat)

    def get_torso_pose(self):
        mujoco.mj_kinematics(self.model, self.data)
        torso_pos = self.data.site("torso").xpos.copy()
        torso_mat = self.data.site("torso").xmat.copy().reshape(3, 3)
        return torso_pos, torso_mat

    def get_joint_state(self):
        joint_state_dict = {}
        time_curr = time.time()
        for name, info in self.robot.joints_info.items():
            if info["active"]:
                joint_state_dict[name] = JointState(
                    time=time_curr,
                    pos=self.data.joint(name).qpos.item(),
                    vel=self.data.joint(name).qvel.item(),
                )

        return joint_state_dict
    
    def get_base_orientation(self):
        return self.data.sensor("orientation").data.copy()
    
    def get_base_angular_velocity(self):
        return self.data.sensor("angular_velocity").data.copy()

    def get_zmp(self, com_pos, pz=0.0):
        M = self.model.body(0).subtreemass
        cx, cy = com_pos
        # print(f"dP: {self.dP}, dL: {self.dL}")
        # Eq. 3.73-3.74 on p.96 in "Introduction to Humanoid Robotics" by Shuuji Kajita
        px_zmp = (M * GRAVITY * cx + pz * self.dP[0] - self.dL[1]) / (
            M * GRAVITY + self.dP[2]
        )
        py_zmp = (M * GRAVITY * cy + pz * self.dP[1] + self.dL[0]) / (
            M * GRAVITY + self.dP[2]
        )
        zmp = [px_zmp.item(), py_zmp.item()]
        return zmp

    def _compute_dynamics(self):
        """Compute dynamics properties for active joints including the filtered mass matrix and bias forces."""
        # Compute the full mass matrix
        full_mass_matrix = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, full_mass_matrix, self.data.qM)

        # Copy the full bias forces from the simulation
        full_bias_forces = self.data.qfrc_bias.copy()

        # Identify active joint indices
        active_indices = []
        for name, info in self.robot.joints_info.items():
            if info["active"]:
                joint_id = self.model.joint(name).id
                active_indices.append(joint_id)

        # Filter the mass matrix and bias forces for active joints
        self.mass_matrix = full_mass_matrix[np.ix_(active_indices, active_indices)]
        self.bias_forces = full_bias_forces[active_indices]

    def set_joint_angles(self, joint_angles):
        self.controller.add_command(joint_angles)

    def run_simulation(self, headless=True, vis_data=None):
        self.thread = threading.Thread(target=self.simulate, args=(headless, vis_data))
        self.thread.start()

    def simulate(self, headless, vis_data):
        mujoco.set_mjcb_control(self.controller.process_commands)

        if headless:
            self.visualizer = MuJoCoRenderer(self.robot, self.model, self.data)
        else:
            self.visualizer = MuJoCoViewer(self.robot, self.model, self.data)

        # self.counter = 0
        while not self.stop_event.is_set():
            step_start = time.time()
            mujoco.mj_step(self.model, self.data)

            # self._compute_dynamics()

            self.visualizer.visualize(self.model, self.data, vis_data)

            time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                precise_sleep(time_until_next_step)

            # log(f"Step {self.counter}", header="MuJoCo", level="debug")
            # self.counter += 1
            # step_end = time.time()
            # log(f"Step Time: {step_end - step_start}", header="MuJoCo", level="debug")

        self.visualizer.close()

    def rollout(self, joint_control_traj):
        n_state = mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        initial_state = np.empty(n_state)
        mujoco.mj_getState(
            self.model, self.data, initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS
        )

        control = np.zeros((len(joint_control_traj), self.model.nu))
        for i, joint_angles in enumerate(joint_control_traj):
            for name, angle in joint_angles.items():
                control[i, self.model.actuator(f"{name}_act").id] = angle

        state_traj, _ = mujoco.rollout.rollout(
            self.model, self.data, initial_state, control
        )
        state_traj = state_traj.squeeze()

        joint_state_traj = []
        # mjSTATE_TIME ｜ mjSTATE_QPOS | mjSTATE_QVEL | mjSTATE_ACT
        for state in state_traj:
            joint_state = {}
            for name, info in self.robot.joints_info.items():
                if info["active"]:
                    joint_state[name] = JointState(
                        time=state[0], pos=state[1 + self.model.joint(name).id]
                    )
            joint_state_traj.append(joint_state)

        return joint_state_traj

    def close(self):
        if self.thread is not None and threading.current_thread() is not self.thread:
            # Wait for the thread to finish if it's not the current thread
            self.stop_event.set()
            self.thread.join()
