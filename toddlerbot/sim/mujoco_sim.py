import threading
import time
from typing import Any, Dict, List

import mujoco  # type: ignore
import mujoco.rollout  # type: ignore
import mujoco.viewer  # type: ignore
import numpy as np
import numpy.typing as npt

from toddlerbot.actuation import JointState
from toddlerbot.sim import BaseSim
from toddlerbot.sim.mujoco_utils import MuJoCoController, MuJoCoRenderer, MuJoCoViewer
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.math_utils import quat_to_euler_arr
from toddlerbot.utils.misc_utils import precise_sleep


class MuJoCoSim(BaseSim):
    def __init__(
        self,
        robot: Robot,
        fixed: bool = False,
        xml_path: str = "",
        xml_str: str = "",
        assets: Any = None,
    ):
        """Initialize the MuJoCo simulation environment."""
        super().__init__()
        self.name = "mujoco"
        self.robot = robot

        if len(xml_str) > 0 and assets is not None:
            self.model = mujoco.MjModel.from_xml_string(xml_str, assets)  # type: ignore
        else:
            if len(xml_path) == 0:
                if fixed:
                    xml_path = find_robot_file_path(
                        robot.name, suffix="_fixed_scene.xml"
                    )
                else:
                    xml_path = find_robot_file_path(robot.name, suffix="_scene.xml")

            self.model = mujoco.MjModel.from_xml_path(xml_path)  # type: ignore

        self.model.opt.timestep = self.dt  # type: ignore
        self.data = mujoco.MjData(self.model)  # type: ignore

        self.controller = MuJoCoController()

        self.thread = None
        self.stop_event = threading.Event()

    def get_link_pos(self, link_name: str):
        mujoco.mj_kinematics(self.model, self.data)  # type: ignore
        link_pos: npt.NDArray[np.float32] = np.array(
            self.data.body(link_name).xpos,  # type: ignore
            copy=True,
        )
        return link_pos

    def get_link_quat(self, link_name: str):
        mujoco.mj_kinematics(self.model, self.data)  # type: ignore
        link_quat: npt.NDArray[np.float32] = np.array(
            self.data.body(link_name).xquat,  # type: ignore
            copy=True,
        )
        return link_quat

    def get_torso_pose(self):
        mujoco.mj_kinematics(self.model, self.data)  # type: ignore
        torso_pos: npt.NDArray[np.float32] = np.array(
            self.data.site("torso").xpos,  # type: ignore
            copy=True,
        )
        torso_mat: npt.NDArray[np.float32] = np.array(
            self.data.site("torso"),  # type: ignore
            copy=True,
        ).reshape(3, 3)
        return torso_pos, torso_mat

    def get_motor_state(self):
        motor_state_dict: Dict[str, JointState] = {}
        for name in self.robot.motor_ordering:
            motor_state_dict[name] = JointState(
                time=time.time() - self.start_time,
                pos=self.data.joint(name).qpos.item(),  # type: ignore
                vel=self.data.joint(name).qvel.item(),  # type: ignore
            )

        return motor_state_dict

    def get_joint_state(self):
        joint_state_dict: Dict[str, JointState] = {}
        for name in self.robot.joint_ordering:
            joint_state_dict[name] = JointState(
                time=time.time() - self.start_time,
                pos=self.data.joint(name).qpos.item(),  # type: ignore
                vel=self.data.joint(name).qvel.item(),  # type: ignore
            )

        return joint_state_dict

    def get_observation(self):
        obs_dict: Dict[str, npt.NDArray[np.float32]] = {}
        motor_state_dict = self.get_motor_state()
        joint_state_dict = self.get_joint_state()

        obs = self.robot.state_to_obs(motor_state_dict, joint_state_dict)
        for k, v in obs.items():
            obs_dict[k] = v

        obs_dict["imu_quat"] = np.array(
            self.data.sensor("orientation").data,  # type: ignore
            copy=True,
        )
        obs_dict["imu_euler"] = quat_to_euler_arr(obs_dict["imu_quat"])
        obs_dict["imu_ang_vel"] = np.array(
            self.data.sensor("angular_velocity").data,  # type: ignore
            copy=True,
        )

        return obs_dict

    def get_mass(self) -> float:
        subtree_mass = float(self.model.body(0).subtreemass)  # type: ignore
        return subtree_mass

    def get_com(self) -> npt.NDArray[np.float32]:
        mujoco.mj_fwdPosition(self.model, self.data)  # type: ignore
        subtree_com = np.array(self.data.body(0).subtree_com, dtype=np.float32)  # type: ignore
        return subtree_com

    # def get_zmp(self, com_pos, pz=0.0):
    #     M = self.model.body(0).subtreemass
    #     cx, cy = com_pos
    #     # print(f"dP: {self.dP}, dL: {self.dL}")
    #     # Eq. 3.73-3.74 on p.96 in "Introduction to Humanoid Robotics" by Shuuji Kajita
    #     px_zmp = (M * GRAVITY * cx + pz * self.dP[0] - self.dL[1]) / (
    #         M * GRAVITY + self.dP[2]
    #     )
    #     py_zmp = (M * GRAVITY * cy + pz * self.dP[1] + self.dL[0]) / (
    #         M * GRAVITY + self.dP[2]
    #     )
    #     zmp = [px_zmp.item(), py_zmp.item()]
    #     return zmp

    # def _compute_dynamics(self):
    #     """Compute dynamics properties for active joints including the filtered mass matrix and bias forces."""
    #     # Compute the full mass matrix
    #     full_mass_matrix = np.zeros((self.model.nv, self.model.nv))
    #     mujoco.mj_fullM(self.model, full_mass_matrix, self.data.qM)

    #     # Copy the full bias forces from the simulation
    #     full_bias_forces = self.data.qfrc_bias.copy()

    #     # Identify active joint indices
    #     active_indices = []
    #     for name in self.robot.config["joints"]:
    #         joint_id = self.model.joint(name).id
    #         active_indices.append(joint_id)

    #     # Filter the mass matrix and bias forces for active joints
    #     self.mass_matrix = full_mass_matrix[np.ix_(active_indices, active_indices)]
    #     self.bias_forces = full_bias_forces[active_indices]

    # def _compute_dmom(self):
    #     if not hasattr(self, "t_last"):
    #         self.last_mom = np.zeros(3)
    #         self.last_angmom = np.zeros(3)
    #         self.t_last = self.data.time
    #     else:
    #         # Calculate current momentum and angular momentum
    #         mom = np.zeros(3)
    #         angmom = np.zeros(3)
    #         for i in range(self.model.nbody):
    #             body_name = self.model.body(i).name
    #             if body_name == "world":
    #                 continue

    #             body_mass = self.model.body(i).mass
    #             body_inertia = self.model.body(i).inertia
    #             body_pos = self.data.body(i).xipos  # Position
    #             body_xmat = self.data.body(i).xmat.reshape(3, 3)  # Rotation matrix
    #             # the translation component comes after the rotation component
    #             body_vel = self.data.body(i).cvel[3:]  # Linear Velocity
    #             body_ang_vel = self.data.body(i).cvel[:3]  # Angular velocity

    #             # Eq. 3.63-3.66 on p.94 in "Introduction to Humanoid Robotics"
    #             P = body_mass * body_vel
    #             L = (
    #                 np.cross(body_pos, P)
    #                 + body_xmat @ np.diag(body_inertia) @ body_xmat.T @ body_ang_vel
    #             )
    #             mom += P
    #             angmom += L

    #         t_curr = self.data.time
    #         dt = t_curr - self.t_last

    #         self.dP = (mom - self.last_mom) / dt
    #         self.dL = (angmom - self.last_angmom) / dt

    #         # Update last states
    #         self.last_mom = mom
    #         self.last_angmom = angmom
    #         self.t_last = t_curr

    def set_motor_angles(self, motor_angles: Dict[str, float]):
        self.controller.add_command(motor_angles)

    def simulate(self, vis_type: str = "", vis_data: Dict[str, Any] = {}):
        self.start_time = time.time()

        self.thread = threading.Thread(
            target=self._simulate_worker, args=(vis_type, vis_data)
        )
        self.thread.start()

    def _simulate_worker(self, vis_type: str = "", vis_data: Dict[str, Any] = {}):
        mujoco.set_mjcb_control(self.controller.process_commands)  # type: ignore

        self.visualizer = None
        if vis_type == "render":
            self.visualizer = MuJoCoRenderer(self.model, self.data)  # type: ignore
        elif vis_type == "view":
            self.visualizer = MuJoCoViewer(self.model, self.data)  # type: ignore

        # self.counter = 0
        while not self.stop_event.is_set():
            step_start = time.time()
            mujoco.mj_step(self.model, self.data)  # type: ignore

            # self._compute_dynamics()

            if self.visualizer is not None:
                self.visualizer.visualize(self.model, self.data, vis_data)  # type: ignore

            time_until_next_step = float(
                self.model.opt.timestep  # type: ignore
                - (time.time() - step_start)
            )
            if time_until_next_step > 0:
                precise_sleep(time_until_next_step)

            # log(f"Step {self.counter}", header="MuJoCo", level="debug")
            # self.counter += 1
            # step_end = time.time()
            # log(f"Step Time: {step_end - step_start}", header="MuJoCo", level="debug")

        if self.visualizer is not None:
            self.visualizer.close()

    def rollout(self, motor_angles_list: List[Dict[str, float]]):
        n_state = mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS)  # type: ignore
        initial_state = np.empty(n_state, dtype=np.float32)  # type: ignore
        mujoco.mj_getState(  # type: ignore
            self.model,  # type: ignore
            self.data,  # type: ignore
            initial_state,
            mujoco.mjtState.mjSTATE_FULLPHYSICS,  # type: ignore
        )

        control = np.zeros(
            (len(motor_angles_list), int(self.model.nu)),  # type: ignore
            dtype=np.float32,
        )
        for i, joint_angles in enumerate(motor_angles_list):
            for name, angle in joint_angles.items():
                control[i, self.model.actuator(name).id] = angle  # type: ignore

        state_traj, _ = mujoco.rollout.rollout(  # type: ignore
            self.model,  # type: ignore
            self.data,  # type: ignore
            initial_state,
            control,
        )
        state_traj = np.array(state_traj, dtype=np.float32).squeeze()

        joint_state_list: List[Dict[str, JointState]] = []
        # mjSTATE_TIME ｜ mjSTATE_QPOS | mjSTATE_QVEL | mjSTATE_ACT
        for state in state_traj:
            joint_state_dict: Dict[str, JointState] = {}
            for name in self.robot.joint_ordering:
                joint_state_dict[name] = JointState(
                    time=state[0],
                    pos=state[1 + self.model.joint(name).id],  # type: ignore
                )
            joint_state_list.append(joint_state_dict)

        return joint_state_list

    def save_recording(self, exp_folder_path: str):
        if isinstance(self.visualizer, MuJoCoRenderer):
            self.visualizer.save_recording(exp_folder_path)

    def close(self):
        if self.thread is not None and threading.current_thread() is not self.thread:
            # Wait for the thread to finish if it's not the current thread
            self.stop_event.set()
            self.thread.join()
