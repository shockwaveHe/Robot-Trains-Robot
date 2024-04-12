import queue
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
from transforms3d.euler import euler2mat

from toddlerbot.sim import BaseSim
from toddlerbot.sim.robot import HumanoidRobot, JointState
from toddlerbot.utils.constants import GRAVITY
from toddlerbot.utils.file_utils import find_description_path
from toddlerbot.utils.misc_utils import sleep


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
            xml_path = find_description_path(robot.name, suffix="_scene.xml")
            self.model = mujoco.MjModel.from_xml_path(xml_path)

        self.data = mujoco.MjData(self.model)

        if not fixed:
            self.put_robot_on_ground()

        self.foot_size = robot.foot_size
        self.last_t = None

        self.queue = queue.Queue()
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
                    time=time_curr, pos=self.data.joint(name).qpos.item()
                )

        return joint_state_dict

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
        self.zmp = [px_zmp.item(), py_zmp.item()]

        return self.zmp

    def _set_joint_angles(self):
        while not self.queue.empty():
            joint_angles = self.queue.get()
            force_dict = {}
            for name, angle in joint_angles.items():
                kp = self.robot.config.motor_params[name].kp
                kv = self.robot.config.motor_params[name].kv
                force = (
                    kp * (angle - self.data.joint(name).qpos)
                    - kv * self.data.joint(name).qvel
                )
                force_dict[name] = force.item()
                self.data.actuator(f"{name}_act").ctrl = force

            # log(f"Force: {force_dict}", header=self.name, level="debug")

    def set_joint_angles(self, joint_angles):
        self.queue.put((joint_angles))

    def _compute_dmom(self):
        if self.last_t is None:
            self.last_mom = np.zeros(3)
            self.last_angmom = np.zeros(3)
            self.last_t = time.time()
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

            time_curr = time.time()
            dt = time_curr - self.last_t

            self.dP = (mom - self.last_mom) / dt
            self.dL = (angmom - self.last_angmom) / dt

            # Update last states
            self.last_mom = mom
            self.last_angmom = angmom
            self.last_t = time_curr

    def vis_foot_steps(self, viewer, foot_steps):
        i = viewer.user_scn.ngeom
        for foot_step in foot_steps:
            if foot_step.support_leg == "both":
                continue

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
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
        viewer.user_scn.ngeom = i

    def vis_com_ref_traj(self, viewer, com_ref_traj):
        i = viewer.user_scn.ngeom
        for com_pos in com_ref_traj:
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.001, 0.001, 0.001]),
                pos=np.array([com_pos[0], com_pos[1], 0.005]),
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 1],
            )
            i += 1
        viewer.user_scn.ngeom = i

    def vis_path(self, viewer, path):
        i = viewer.user_scn.ngeom
        for j in range(len(path) - 1):
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=np.array([1, 1, 1]),
                pos=np.array([0, 0, 0]),
                mat=np.eye(3).flatten(),
                rgba=[0, 0, 0, 1],
            )
            mujoco.mjv_connector(
                viewer.user_scn.geoms[i],
                mujoco.mjtGeom.mjGEOM_LINE,
                100,
                np.array([*path[j], 0.0]),
                np.array([*path[j + 1], 0.0]),
            )
            i += 1
        viewer.user_scn.ngeom = i

    def vis_torso(self, viewer):
        i = viewer.user_scn.ngeom
        torso_pos = self.data.site("torso").xpos
        torso_mat = self.data.site("torso").xmat
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=np.array([0.005, 0.005, 0.15]),
            pos=torso_pos,
            mat=torso_mat,
            rgba=[1, 0, 0, 1],
        )
        viewer.user_scn.ngeom = i + 1

    def simulate_worker(self, vis_data, sleep_time):
        viewer = mujoco.viewer.launch_passive(self.model, self.data)

        try:
            while viewer.is_running():
                step_start = time.time()

                self._set_joint_angles()

                mujoco.mj_step(self.model, self.data)

                # self._compute_dmom()

                with viewer.lock():
                    if vis_data is not None:
                        viewer.user_scn.ngeom = 0
                        if "foot_steps" in vis_data:
                            self.vis_foot_steps(viewer, vis_data["foot_steps"])
                        if "com_ref_traj" in vis_data:
                            self.vis_com_ref_traj(viewer, vis_data["com_ref_traj"])
                        if "path" in vis_data:
                            self.vis_path(viewer, vis_data["path"])
                        if "torso" in vis_data:
                            self.vis_torso(viewer)

                viewer.sync()

                time_until_next_step = sleep_time - (time.time() - step_start)
                if time_until_next_step > 0:
                    sleep(time_until_next_step)

        except KeyboardInterrupt:
            pass

        finally:
            viewer.close()
            self.close()

    def simulate(self, vis_data=None, sleep_time: float = 0.0):
        self.sim_thread = threading.Thread(
            target=self.simulate_worker, args=(vis_data, sleep_time)
        )
        self.sim_thread.start()

    def close(self):
        self.stop_event.set()
        if threading.current_thread() is not self.sim_thread:
            # Wait for the thread to finish if it's not the current thread
            self.sim_thread.join()


if __name__ == "__main__":
    robot = HumanoidRobot("robotis_op3")
    sim = MuJoCoSim()
    sim.simulate_worker()
