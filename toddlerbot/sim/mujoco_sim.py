import time
from typing import Any, Dict, List, Union

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


class MuJoCoSim(BaseSim):
    def __init__(
        self,
        robot: Robot,
        fixed_base: bool = False,
        xml_path: str = "",
        xml_str: str = "",
        assets: Any = None,
        vis_type: str = "",
    ):
        """Initialize the MuJoCo simulation environment."""
        super().__init__()
        self.name = "mujoco"
        self.robot = robot
        self.fixed_base = fixed_base

        if len(xml_str) > 0 and assets is not None:
            self.model = mujoco.MjModel.from_xml_string(xml_str, assets)  # type: ignore
        else:
            if len(xml_path) == 0:
                if fixed_base:
                    xml_path = find_robot_file_path(
                        robot.name, suffix="_fixed_scene.xml"
                    )
                else:
                    xml_path = find_robot_file_path(robot.name, suffix="_scene.xml")

            self.model = mujoco.MjModel.from_xml_path(xml_path)  # type: ignore

        self.model.opt.timestep = self.dt  # type: ignore
        self.data = mujoco.MjData(self.model)  # type: ignore

        self.controller = MuJoCoController()
        mujoco.set_mjcb_control(self.controller.process_commands)  # type: ignore

        # Initialize push state variables
        self.apply_push_flag = False
        self.push = np.zeros(6, dtype=np.float32)  # Initial push as zero vector
        self.push_duration: float = 0.0
        self.push_count: int = 0

        self.visualizer = None
        if vis_type == "render":
            self.visualizer = MuJoCoRenderer(self.model, self.data)  # type: ignore
        elif vis_type == "view":
            self.visualizer = MuJoCoViewer(self.model, self.data)  # type: ignore

        # self.thread = None
        # self.stop_event = threading.Event()

    def get_root_state(self):
        root_state = np.zeros(13, dtype=np.float32)
        root_state[:3] = np.array(
            self.data.sensor("position").data,  # type: ignore
            copy=True,
        )
        root_state[3:7] = np.array(
            self.data.sensor("orientation").data,  # type: ignore
            copy=True,
        )
        root_state[7:10] = np.array(
            self.data.sensor("linear_velocity").data,  # type: ignore
            copy=True,
        )
        root_state[10:] = np.array(
            self.data.sensor("angular_velocity").data,  # type: ignore
            copy=True,
        )
        return root_state

    def get_dof_state(self):
        dof_state = np.zeros((len(self.robot.motor_ordering), 2), dtype=np.float32)  # type: ignore
        for i, name in enumerate(self.robot.motor_ordering):
            dof_state[i, 0] = self.data.joint(name).qpos.item()  # type: ignore
            dof_state[i, 1] = self.data.joint(name).qvel.item()  # type: ignore

        return dof_state

    def get_body_state(self):
        dof_state = np.zeros((len(self.robot.collider_names), 13), dtype=np.float32)
        for i, name in enumerate(self.robot.collider_names):
            dof_state[i, :3] = self.data.body(name).xpos.copy()  # type: ignore
            dof_state[i, 3:7] = self.data.body(name).xquat.copy()  # type: ignore
            dof_state[i, 7:10] = self.data.body(name).cvel[3:].copy()  # type: ignore
            dof_state[i, 10:] = self.data.body(name).cvel[:3].copy()  # type: ignore

        return dof_state

    def get_contact_forces(self):
        # TODO: check if this is the correct way to get contact forces
        contact_forces = np.zeros((len(self.robot.collider_names), 3), dtype=np.float32)
        # Access contact information
        for i in range(self.data.ncon):  # type: ignore
            contact = self.data.contact[i]  # type: ignore
            # Check if the contact involves the ground plane
            geom1 = self.model.geom(contact.geom[0])  # type: ignore
            geom2 = self.model.geom(contact.geom[1])  # type: ignore

            body_name = ""
            if "floor" in geom1.name:  # type: ignore
                body_name = str(self.model.body(geom2.bodyid).name)  # type: ignore
            elif "floor" in geom2.name:  # type: ignore
                body_name = str(self.model.body(geom1.bodyid).name)  # type: ignore
            else:
                continue

            # Extract the contact forces
            c_array = np.zeros(6)  # To hold contact forces
            mujoco.mj_contactForce(self.model, self.data, i, c_array)  # type: ignore
            contact_force_local = c_array[:3].astype(np.float32)
            contact_force_global = contact.frame.reshape(-1, 3).T @ contact_force_local  # type: ignore
            contact_forces[self.robot.collider_names.index(body_name)] = (
                contact_force_global
            )

        return contact_forces

    # TODO: consider merging these methods
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

    def start_push(self, push: npt.NDArray[np.float32], push_duration: float = 0.1):
        self.apply_push_flag = True
        self.push = push
        self.push_duration = push_duration
        self.push_count = 0

    def _apply_push_cb(self):
        """Applies random pushes to the robot's torso."""
        # Unpack the random push tensor
        linear_vel = self.push[:3]  # xy linear velocity
        angular_vel = self.push[3:]  # xyz angular velocity
        mass = self.model.body("torso").mass  # type: ignore
        inertia = self.model.body("torso").inertia  # type: ignore
        # Apply the forces and torques
        self.data.body("torso").xfrc_applied[:3] = (  # type: ignore
            mass * linear_vel
        ) / self.push_duration
        self.data.body("torso").xfrc_applied[3:] = (  # type: ignore
            inertia * angular_vel
        ) / self.push_duration

    def _reset_push(self):
        # Reset applied forces and torques
        self.apply_push_flag = False
        self.data.body("torso").xfrc_applied[:] = 0  # type: ignore

    def set_root_state(self, root_state: npt.NDArray[np.float32]):
        # Assume the free joint is the first joint
        self.data.joint(0).qpos[:3] = root_state[:3].copy()  # type: ignore
        self.data.joint(0).qpos[3:] = root_state[3:7].copy()  # type: ignore
        # Set linear velocity (3) and angular velocity (3) in qvel
        self.data.joint(0).qvel[:3] = root_state[7:10].copy()  # type: ignore
        self.data.joint(0).qvel[3:] = root_state[10:13].copy()  # type: ignore
        # mujoco.mj_resetData(self.model, self.data)  # type: ignore

    def set_dof_state(self, dof_state: npt.NDArray[np.float32]):
        for i, name in enumerate(self.robot.motor_ordering):
            self.data.joint(name).qpos = dof_state[i, 0].copy()  # type: ignore
            self.data.joint(name).qvel = dof_state[i, 1].copy()  # type: ignore

    def set_motor_angles(
        self, motor_angles: Union[Dict[str, float], npt.NDArray[np.float32]]
    ):
        self.controller.add_command(motor_angles)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)  # type: ignore

    def step(self):
        # step_start = time.time()
        if self.apply_push_flag and self.push_count < (self.push_duration / self.dt):
            self._apply_push_cb()
            self.push_count += 1
        else:
            self._reset_push()

        mujoco.mj_step(self.model, self.data)  # type: ignore

        if self.visualizer is not None:
            self.visualizer.visualize(self.model, self.data)  # type: ignore

        # time_until_next_step = float(
        #     self.model.opt.timestep  # type: ignore
        #     - (time.time() - step_start)
        # )
        # if time_until_next_step > 0:
        #     precise_sleep(time_until_next_step)

    # def _simulate_worker(self, vis_type: str = "", vis_data: Dict[str, Any] = {}):
    #     while not self.stop_event.is_set():
    #         self.step()

    # def simulate(self, vis_type: str = "", vis_data: Dict[str, Any] = {}):
    #     self.start_time = time.time()

    #     self.thread = threading.Thread(
    #         target=self._simulate_worker, args=(vis_type, vis_data)
    #     )
    #     self.thread.start()

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
        if self.visualizer is not None:
            self.visualizer.close()

        # if self.thread is not None and threading.current_thread() is not self.thread:
        #     # Wait for the thread to finish if it's not the current thread
        #     self.stop_event.set()
        #     self.thread.join()
