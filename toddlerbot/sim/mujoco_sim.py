import time
from typing import Any, Dict, List

import mujoco
import mujoco.rollout
import mujoco.viewer
import numpy as np
import numpy.typing as npt

from toddlerbot.actuation import JointState
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.mujoco_control import (
    MotorController,
    PositionController,
)
from toddlerbot.sim.mujoco_utils import MuJoCoRenderer, MuJoCoViewer
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.math_utils import quat2euler, quat_inv, rotate_vec


class MuJoCoSim(BaseSim):
    """A class for the MuJoCo simulation environment."""

    def __init__(
        self,
        robot: Robot,
        n_frames: int = 20,
        dt: float = 0.001,
        fixed_base: bool = False,
        xml_path: str = "",
        xml_str: str = "",
        assets: Any = None,
        vis_type: str = "",
    ):
        """Initializes the simulation environment for a robot using the MuJoCo physics engine.

        Args:
            robot (Robot): The robot object to be simulated.
            n_frames (int, optional): Number of frames per control step. Defaults to 20.
            dt (float, optional): Time step for the simulation in seconds. Defaults to 0.001.
            fixed_base (bool, optional): Whether the robot has a fixed base. Defaults to False.
            xml_path (str, optional): Path to the XML file defining the robot model. Defaults to an empty string.
            xml_str (str, optional): XML string defining the robot model. Defaults to an empty string.
            assets (Any, optional): Additional assets required for the XML model. Defaults to None.
            vis_type (str, optional): Type of visualization to use ('render' or 'view'). Defaults to an empty string.
        """
        super().__init__("mujoco")

        self.robot = robot
        self.n_frames = n_frames
        self.dt = dt
        self.control_dt = n_frames * dt
        self.fixed_base = fixed_base

        if len(xml_str) > 0 and assets is not None:
            model = mujoco.MjModel.from_xml_string(xml_str, assets)
        else:
            if len(xml_path) == 0:
                if fixed_base:
                    xml_path = find_robot_file_path(
                        robot.name, suffix="_fixed_scene.xml"
                    )
                else:
                    xml_path = find_robot_file_path(robot.name, suffix="_scene.xml")

            model = mujoco.MjModel.from_xml_path(xml_path)

        self.model = model
        self.data = mujoco.MjData(model)

        self.model.opt.timestep = self.dt
        self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.model.opt.iterations = 1
        self.model.opt.ls_iterations = 4
        # self.model.opt.gravity[2] = -1.0

        # Assume imu is the first site
        self.torso_euler_prev = np.zeros(3, dtype=np.float32)
        self.motor_vel_prev = np.zeros(self.model.nu, dtype=np.float32)

        # if fixed_base:
        #     self.controller = PositionController()
        # else:
        self.motor_indices = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.motor_ordering
            ]
        )
        if not self.fixed_base:
            # Disregard the free joint
            self.motor_indices -= 1

        self.q_start_idx = 0 if self.fixed_base else 7
        self.qd_start_idx = 0 if self.fixed_base else 6

        self.left_foot_name = "ank_roll_link"
        self.right_foot_name = "ank_roll_link_2"

        self.controller: MotorController | PositionController
        # Check if the actuator is a motor or position actuator
        if (
            self.model.actuator(0).gainprm[0] == 1
            and self.model.actuator(0).biasprm[1] == 0
        ):
            self.controller = MotorController(robot)
        else:
            self.controller = PositionController()

        self.target_motor_angles = np.zeros(self.model.nu, dtype=np.float32)

        self.visualizer: MuJoCoRenderer | MuJoCoViewer | None = None
        if vis_type == "render":
            self.visualizer = MuJoCoRenderer(self.model)
        elif vis_type == "view":
            self.visualizer = MuJoCoViewer(robot, self.model, self.data)

        try:
            self.default_qpos = np.array(
                self.model.keyframe("home").qpos, dtype=np.float32
            )
            self.data.qpos = self.default_qpos.copy()
            self.forward()

            self.left_foot_transform = self.get_body_transofrm(self.left_foot_name)
            self.right_foot_transform = self.get_body_transofrm(self.right_foot_name)

        except KeyError:
            print("No keyframe named 'home' found in the model.")

    def get_body_transofrm(self, body_name: str):
        """Computes the transformation matrix for a specified body.

        Args:
            body_name (str): The name of the body for which to compute the transformation matrix.

        Returns:
            np.ndarray: A 4x4 transformation matrix representing the position and orientation of the body.
        """
        transformation = np.eye(4)
        body_pos = self.data.body(body_name).xpos.copy()
        body_mat = self.data.body(body_name).xmat.reshape(3, 3).copy()
        transformation[:3, :3] = body_mat
        transformation[:3, 3] = body_pos
        return transformation

    def get_site_transform(self, site_name: str):
        """Retrieves the transformation matrix for a specified site.

        This method constructs a 4x4 transformation matrix for the given site name,
        using the site's position and orientation matrix. The transformation matrix
        is composed of a 3x3 rotation matrix and a 3x1 translation vector.

        Args:
            site_name (str): The name of the site for which to retrieve the transformation matrix.

        Returns:
            numpy.ndarray: A 4x4 transformation matrix representing the site's position and orientation.
        """
        transformation = np.eye(4)
        site_pos = self.data.site(site_name).xpos.copy()
        site_mat = self.data.site(site_name).xmat.reshape(3, 3).copy()
        transformation[:3, :3] = site_mat
        transformation[:3, 3] = site_pos
        return transformation

    def get_motor_state(self) -> Dict[str, JointState]:
        """Retrieve the current state of each motor in the robot.

        Returns:
            Dict[str, JointState]: A dictionary mapping each motor's name to its
            current state, including position, velocity, and torque.
        """
        motor_state_dict: Dict[str, JointState] = {}
        for name in self.robot.motor_ordering:
            motor_state_dict[name] = JointState(
                time=time.time(),
                pos=self.data.joint(name).qpos.item(),
                vel=self.data.joint(name).qvel.item(),
                tor=self.data.actuator(name).force.item(),
            )

        return motor_state_dict

    def get_motor_angles(
        self, type: str = "dict"
    ) -> Dict[str, float] | npt.NDArray[np.float32]:
        """Retrieves the current angles of the robot's motors.

        Args:
            type (str): The format in which to return the motor angles.
                Options are "dict" for a dictionary format or "array" for a NumPy array.
                Defaults to "dict".

        Returns:
            Dict[str, float] or npt.NDArray[np.float32]: The motor angles in the specified format.
            If "dict", returns a dictionary with motor names as keys and angles as values.
            If "array", returns a NumPy array of motor angles.
        """
        motor_angles: Dict[str, float] = {}
        for name in self.robot.motor_ordering:
            motor_angles[name] = self.data.joint(name).qpos.item()

        if type == "array":
            motor_pos_arr = np.array(list(motor_angles.values()), dtype=np.float32)
            return motor_pos_arr
        else:
            return motor_angles

    def get_joint_state(self) -> Dict[str, JointState]:
        """Retrieves the current state of each joint in the robot.

        Returns:
            Dict[str, JointState]: A dictionary mapping each joint's name to its current state,
            which includes the timestamp, position, and velocity.
        """
        joint_state_dict: Dict[str, JointState] = {}
        for name in self.robot.joint_ordering:
            joint_state_dict[name] = JointState(
                time=time.time(),
                pos=self.data.joint(name).qpos.item(),
                vel=self.data.joint(name).qvel.item(),
            )

        return joint_state_dict

    def get_joint_angles(
        self, type: str = "dict"
    ) -> Dict[str, float] | npt.NDArray[np.float32]:
        """Retrieves the current joint angles of the robot.

        Args:
            type (str): The format in which to return the joint angles.
                Options are "dict" for a dictionary format or "array" for a NumPy array.
                Defaults to "dict".

        Returns:
            Dict[str, float] or npt.NDArray[np.float32]: The joint angles of the robot.
                Returns a dictionary with joint names as keys and angles as values if
                `type` is "dict". Returns a NumPy array of joint angles if `type` is "array".
        """
        joint_angles: Dict[str, float] = {}
        for name in self.robot.joint_ordering:
            joint_angles[name] = self.data.joint(name).qpos.item()

        if type == "array":
            joint_pos_arr = np.array(list(joint_angles.values()), dtype=np.float32)
            return joint_pos_arr
        else:
            return joint_angles

    def get_observation(self) -> Obs:
        """Retrieves the current observation of the robot's state, including motor and joint states, and torso dynamics.

        Returns:
            Obs: An observation object containing the following attributes:
                - time (float): The timestamp of the observation.
                - motor_pos (np.ndarray): Array of motor positions.
                - motor_vel (np.ndarray): Array of motor velocities.
                - motor_tor (np.ndarray): Array of motor torques.
                - lin_vel (np.ndarray): Linear velocity of the torso.
                - ang_vel (np.ndarray): Angular velocity of the torso.
                - pos (np.ndarray): Position of the torso.
                - euler (np.ndarray): Euler angles of the torso.
                - joint_pos (np.ndarray): Array of joint positions.
                - joint_vel (np.ndarray): Array of joint velocities.
        """
        motor_state_dict = self.get_motor_state()
        joint_state_dict = self.get_joint_state()

        time = list(motor_state_dict.values())[0].time

        # joints_config = self.robot.config["joints"]
        motor_pos: List[float] = []
        motor_vel: List[float] = []
        motor_tor: List[float] = []
        for motor_name in motor_state_dict:
            motor_pos.append(motor_state_dict[motor_name].pos)
            motor_vel.append(motor_state_dict[motor_name].vel)
            motor_tor.append(motor_state_dict[motor_name].tor)

        motor_pos_arr = np.array(motor_pos, dtype=np.float32)
        motor_vel_arr = np.array(motor_vel, dtype=np.float32)
        motor_tor_arr = np.array(motor_tor, dtype=np.float32)

        joint_pos: List[float] = []
        joint_vel: List[float] = []
        for joint_name in joint_state_dict:
            joint_pos.append(joint_state_dict[joint_name].pos)
            joint_vel.append(joint_state_dict[joint_name].vel)

        joint_pos_arr = np.array(joint_pos, dtype=np.float32)
        joint_vel_arr = np.array(joint_vel, dtype=np.float32)

        if self.fixed_base:
            torso_lin_vel = np.zeros(3, dtype=np.float32)
            torso_ang_vel = np.zeros(3, dtype=np.float32)
            torso_pos = np.zeros(3, dtype=np.float32)
            torso_euler = np.zeros(3, dtype=np.float32)
        else:
            lin_vel_global = np.array(
                self.data.body("torso").cvel[3:],
                dtype=np.float32,
                copy=True,
            )
            ang_vel_global = np.array(
                self.data.body("torso").cvel[:3],
                dtype=np.float32,
                copy=True,
            )
            torso_pos = np.array(
                self.data.body("torso").xpos,
                dtype=np.float32,
                copy=True,
            )
            torso_quat = np.array(
                self.data.body("torso").xquat,
                dtype=np.float32,
                copy=True,
            )
            if np.linalg.norm(torso_quat) == 0:
                torso_quat = np.array([1, 0, 0, 0], dtype=np.float32)

            torso_lin_vel = np.asarray(rotate_vec(lin_vel_global, quat_inv(torso_quat)))
            torso_ang_vel = np.asarray(rotate_vec(ang_vel_global, quat_inv(torso_quat)))

            torso_euler = np.asarray(quat2euler(torso_quat))
            torso_euler_delta = torso_euler - self.torso_euler_prev
            torso_euler_delta = (torso_euler_delta + np.pi) % (2 * np.pi) - np.pi
            torso_euler = self.torso_euler_prev + torso_euler_delta
            self.torso_euler_prev = np.asarray(torso_euler, dtype=np.float32)

        obs = Obs(
            time=time,
            motor_pos=motor_pos_arr,
            motor_vel=motor_vel_arr,
            motor_tor=motor_tor_arr,
            lin_vel=torso_lin_vel,
            ang_vel=torso_ang_vel,
            pos=torso_pos,
            euler=torso_euler,
            joint_pos=joint_pos_arr,
            joint_vel=joint_vel_arr,
        )
        return obs

    def get_mass(self) -> float:
        """Calculate and return the mass of the subtree.

        Returns:
            float: The mass of the subtree as a floating-point number.
        """
        subtree_mass = float(self.model.body(0).subtreemass)
        return subtree_mass

    def set_torso_pos(self, torso_pos: npt.NDArray[np.float32]):
        """Set the position of the torso in the simulation.

        Args:
            torso_pos (npt.NDArray[np.float32]): A numpy array representing the desired position of the torso in 3D space.
        """
        self.data.joint(0).qpos[:3] = torso_pos

    def set_torso_quat(self, torso_quat: npt.NDArray[np.float32]):
        """Set the quaternion of the torso joint.

        This method updates the quaternion orientation of the torso joint in the
        simulation data.

        Args:
            torso_quat (npt.NDArray[np.float32]): A numpy array representing the
                quaternion to set for the torso joint, with four elements
                corresponding to the quaternion components.
        """
        self.data.joint(0).qpos[3:7] = torso_quat

    def set_motor_kps(self, motor_kps: Dict[str, float]):
        """Sets the proportional gain (Kp) values for the motors.

        This method updates the Kp values for each motor specified in the `motor_kps` dictionary. If the controller is an instance of `MotorController`, it sets the Kp value directly in the controller. Otherwise, it adjusts the gain and bias parameters of the actuator model.

        Args:
            motor_kps (Dict[str, float]): A dictionary where keys are motor names and values are the Kp values to be set.
        """
        for name, kp in motor_kps.items():
            if isinstance(self.controller, MotorController):
                self.controller.kp[self.model.actuator(name).id] = kp / 128
            else:
                self.model.actuator(name).gainprm[0] = kp / 128
                self.model.actuator(name).biasprm[1] = -kp / 128

    def set_motor_target(
        self, motor_angles: Dict[str, float] | npt.NDArray[np.float32]
    ):
        """Sets the target angles for the motors.

        Args:
            motor_angles (Dict[str, float] | npt.NDArray[np.float32]): A dictionary mapping motor names to their target angles or a NumPy array of target angles. If a dictionary is provided, the values are converted to a NumPy array of type float32.
        """
        if isinstance(motor_angles, dict):
            self.target_motor_angles = np.array(
                list(motor_angles.values()), dtype=np.float32
            )
        else:
            self.target_motor_angles = motor_angles

    def set_motor_angles(
        self, motor_angles: Dict[str, float] | npt.NDArray[np.float32]
    ):
        """Sets the motor angles for the robot and updates the corresponding joint and passive angles.

        Args:
            motor_angles (Dict[str, float] | npt.NDArray[np.float32]): A dictionary mapping motor names to angles or an array of motor angles in the order specified by the robot's motor ordering.
        """
        if not isinstance(motor_angles, dict):
            motor_angles = dict(zip(self.robot.motor_ordering, motor_angles))

        for name in motor_angles:
            self.data.joint(name).qpos = motor_angles[name]

        joint_angles = self.robot.motor_to_joint_angles(motor_angles)
        for name in joint_angles:
            self.data.joint(name).qpos = joint_angles[name]

        passive_angles = self.robot.joint_to_passive_angles(joint_angles)
        for name in passive_angles:
            self.data.joint(name).qpos = passive_angles[name]

    def set_joint_angles(
        self, joint_angles: Dict[str, float] | npt.NDArray[np.float32]
    ):
        """Sets the joint angles of the robot.

        This method updates the joint positions of the robot based on the provided joint angles. It converts the input joint angles to motor and passive angles and updates the robot's data structure accordingly.

        Args:
            joint_angles (Dict[str, float] | npt.NDArray[np.float32]): A dictionary mapping joint names to their respective angles, or a NumPy array of joint angles in the order specified by the robot's joint ordering.
        """
        if not isinstance(joint_angles, dict):
            joint_angles = dict(zip(self.robot.joint_ordering, joint_angles))

        for name in joint_angles:
            self.data.joint(name).qpos = joint_angles[name]

        motor_angles = self.robot.joint_to_motor_angles(joint_angles)
        for name in motor_angles:
            self.data.joint(name).qpos = motor_angles[name]

        passive_angles = self.robot.joint_to_passive_angles(joint_angles)
        for name in passive_angles:
            self.data.joint(name).qpos = passive_angles[name]

    def set_qpos(self, qpos: npt.NDArray[np.float32]):
        """Set the position of the system's generalized coordinates.

        Args:
            qpos (npt.NDArray[np.float32]): An array representing the desired positions of the system's generalized coordinates.
        """
        self.data.qpos = qpos

    def set_joint_dynamics(self, joint_dyn: Dict[str, Dict[str, float]]):
        """Sets the dynamics parameters for specified joints in the model.

        Args:
            joint_dyn (Dict[str, Dict[str, float]]): A dictionary where each key is a joint name and the value is another dictionary containing dynamics parameters and their corresponding values to be set for that joint.
        """
        for joint_name, dyn in joint_dyn.items():
            for key, value in dyn.items():
                setattr(self.model.joint(joint_name), key, value)

    def set_motor_dynamics(self, motor_dyn: Dict[str, float]):
        """Sets the motor dynamics by updating the controller's attributes.

        Args:
            motor_dyn (Dict[str, float]): A dictionary where keys are attribute names and values are the corresponding dynamics values to be set on the controller.
        """
        for key, value in motor_dyn.items():
            setattr(self.controller, key, value)

    def forward(self):
        """Advances the simulation forward by a specified number of frames and visualizes the result if a visualizer is available.

        Iterates through the simulation for the number of frames specified by `self.n_frames`, updating the model state at each step. If a visualizer is provided, it visualizes the current state of the simulation data.
        """
        for _ in range(self.n_frames):
            mujoco.mj_forward(self.model, self.data)

        if self.visualizer is not None:
            self.visualizer.visualize(self.data)

    def step(self):
        """Advances the simulation by a specified number of frames and updates the visualizer.

        This method iterates over the number of frames defined by `n_frames`, updating the control inputs using the controller's step method based on the current position and velocity of the motors. It then advances the simulation state using Mujoco's `mj_step` function. If a visualizer is provided, it updates the visualization with the current simulation data.
        """
        for _ in range(self.n_frames):
            self.data.ctrl = self.controller.step(
                self.data.qpos[self.q_start_idx + self.motor_indices],
                self.data.qvel[self.qd_start_idx + self.motor_indices],
                self.target_motor_angles,
            )
            mujoco.mj_step(self.model, self.data)

        if self.visualizer is not None:
            self.visualizer.visualize(self.data)

    def rollout(
        self,
        motor_ctrls_list: List[Dict[str, float]]  # Either motor angles or motor torques
        | List[npt.NDArray[np.float32]]
        | npt.NDArray[np.float32],
    ) -> List[Dict[str, JointState]]:
        """Simulate the dynamics of a robotic system using a sequence of motor controls.

        This function performs a simulation rollout of a robotic system using the provided motor controls, which can be specified as either motor angles or torques. The simulation is executed over a series of frames, and the resulting joint states are returned.

        Args:
            motor_ctrls_list: A list containing motor control inputs. Each element can be a dictionary mapping motor names to control values or a NumPy array of control values. The controls can represent either motor angles or torques.

        Returns:
            A list of dictionaries, where each dictionary represents the joint states at a specific time step. Each dictionary maps joint names to their corresponding `JointState`, which includes the time and position of the joint.
        """
        n_state = mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        initial_state = np.empty(n_state, dtype=np.float64)
        mujoco.mj_getState(
            self.model,
            self.data,
            initial_state,
            mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )

        control = np.zeros(
            (len(motor_ctrls_list) * self.n_frames, int(self.model.nu)),
            dtype=np.float64,
        )
        for i, motor_ctrls in enumerate(motor_ctrls_list):
            if isinstance(motor_ctrls, np.ndarray):
                control[self.n_frames * i : self.n_frames * (i + 1)] = motor_ctrls
            else:
                for name, ctrl in motor_ctrls.items():
                    control[
                        self.n_frames * i : self.n_frames * (i + 1),
                        self.model.actuator(name).id,
                    ] = ctrl

        state_traj, _ = mujoco.rollout.rollout(
            self.model,
            self.data,
            initial_state,
            control,
        )
        state_traj = np.array(state_traj, dtype=np.float32).squeeze()[:: self.n_frames]
        # mjSTATE_TIME ｜ mjSTATE_QPOS | mjSTATE_QVEL | mjSTATE_ACT

        # joints_config = self.robot.config["joints"]
        joint_state_list: List[Dict[str, JointState]] = []
        for state in state_traj:
            joint_state: Dict[str, JointState] = {}
            for joint_name in self.robot.joint_ordering:
                joint_pos = state[1 + self.model.joint(joint_name).id]
                joint_state[joint_name] = JointState(time=state[0], pos=joint_pos)

            joint_state_list.append(joint_state)

        return joint_state_list

    def save_recording(
        self,
        exp_folder_path: str,
        dt: float,
        render_every: int,
        name: str = "mujoco.mp4",
    ):
        """Saves a recording of the current MuJoCo simulation.

        Args:
            exp_folder_path (str): The path to the folder where the recording will be saved.
            dt (float): The time step interval for the recording.
            render_every (int): The frequency at which frames are rendered.
            name (str, optional): The name of the output video file. Defaults to "mujoco.mp4".
        """
        if isinstance(self.visualizer, MuJoCoRenderer):
            self.visualizer.save_recording(exp_folder_path, dt, render_every, name)

    def close(self):
        """Closes the visualizer if it is currently open."""
        if self.visualizer is not None:
            self.visualizer.close()
