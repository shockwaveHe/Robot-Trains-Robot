import time
from typing import Any, Dict, List

import mujoco  # type: ignore
import mujoco.rollout  # type: ignore
import mujoco.viewer  # type: ignore
import numpy as np
import numpy.typing as npt

from toddlerbot.actuation import JointState
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.mujoco_utils import MuJoCoRenderer, MuJoCoViewer
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.math_utils import (
    exponential_moving_average,
    quat2euler,
    quat_inv,
    rotate_vec,
)


class MuJoCoSim(BaseSim):
    def __init__(
        self,
        robot: Robot,
        n_frames: int = 6,
        dt: float = 0.002,
        fixed_base: bool = False,
        xml_path: str = "",
        xml_str: str = "",
        assets: Any = None,
        vis_type: str = "",
    ):
        """Initialize the MuJoCo simulation environment."""
        super().__init__("mujoco")

        self.robot = robot
        self.n_frames = n_frames
        self.dt = dt
        self.control_dt = n_frames * dt
        self.fixed_base = fixed_base

        if len(xml_str) > 0 and assets is not None:
            model = mujoco.MjModel.from_xml_string(xml_str, assets)  # type: ignore
        else:
            if len(xml_path) == 0:
                if fixed_base:
                    xml_path = find_robot_file_path(
                        robot.name, suffix="_fixed_scene.xml"
                    )
                else:
                    xml_path = find_robot_file_path(robot.name, suffix="_scene.xml")

            model = mujoco.MjModel.from_xml_path(xml_path)  # type: ignore

        self.model = model  # type: ignore
        self.data = mujoco.MjData(model)  # type: ignore

        self.model.opt.timestep = self.dt  # type: ignore
        self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON  # type: ignore
        self.model.opt.iterations = 1  # type: ignore
        self.model.opt.ls_iterations = 4  # type: ignore

        # Assume imu is the first site
        self.torso_euler_prev = np.zeros(3, dtype=np.float32)
        self.motor_vel_prev = np.zeros(self.model.nu, dtype=np.float32)  # type: ignore

        self.visualizer = None
        if vis_type == "render":
            self.visualizer = MuJoCoRenderer(self.model)  # type: ignore
        elif vis_type == "view":
            self.visualizer = MuJoCoViewer(self.model, self.data)  # type: ignore

    def load_keyframe(self):
        default_qpos = np.array(self.model.keyframe("home").qpos, dtype=np.float32)  # type:ignore
        self.data.qpos = default_qpos.copy()  # type: ignore
        self.data.qvel = np.zeros(self.model.nv, dtype=np.float32)  # type: ignore
        self.forward()

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
        dof_state = np.zeros((len(self.robot.joint_ordering), 2), dtype=np.float32)  # type: ignore
        for i, name in enumerate(self.robot.joint_ordering):
            dof_state[i, 0] = self.data.joint(name).qpos.item()  # type: ignore
            dof_state[i, 1] = self.data.joint(name).qvel.item()  # type: ignore

        return dof_state

    def get_body_state(self):
        dof_state = np.zeros((len(self.robot.collider_names), 13), dtype=np.float32)
        for i, name in enumerate(self.robot.collider_names):
            dof_state[i, :3] = self.data.body(name).xpos.copy()  # type: ignore
            dof_state[i, 3:7] = self.data.body(name).xquat.copy()  # type: ignore
            # rot goes before lin in cvel
            dof_state[i, 7:10] = self.data.body(name).cvel[3:].copy()  # type: ignore
            dof_state[i, 10:] = self.data.body(name).cvel[:3].copy()  # type: ignore

        return dof_state

    def get_motor_state(self):
        motor_state_dict: Dict[str, JointState] = {}
        for name in self.robot.motor_ordering:
            motor_state_dict[name] = JointState(
                time=time.time(),
                pos=self.data.joint(name).qpos.item(),  # type: ignore
                vel=self.data.joint(name).qvel.item(),  # type: ignore
            )

        return motor_state_dict

    def get_joint_state(self):
        joint_state_dict: Dict[str, JointState] = {}
        for name in self.robot.joint_ordering:
            joint_state_dict[name] = JointState(
                time=time.time(),
                pos=self.data.joint(name).qpos.item(),  # type: ignore
                vel=self.data.joint(name).qvel.item(),  # type: ignore
            )

        return joint_state_dict

    def get_observation(self):
        motor_state_dict = self.get_motor_state()
        joint_state_dict = self.get_joint_state()

        time = list(motor_state_dict.values())[0].time

        # joints_config = self.robot.config["joints"]
        motor_pos: List[float] = []
        motor_vel: List[float] = []
        for motor_name in motor_state_dict:
            # transmission = joints_config[motor_name]["transmission"]
            # if transmission == "gears":
            #     joint_name = self.robot.motor_to_joint_name[motor_name]
            #     motor_pos.append(
            #         joint_state_dict[joint_name].pos
            #         / joints_config[motor_name]["gear_ratio"]
            #     )
            #     motor_vel.append(
            #         joint_state_dict[joint_name].vel
            #         * joints_config[motor_name]["gear_ratio"]
            #     )
            # else:
            motor_pos.append(motor_state_dict[motor_name].pos)
            motor_vel.append(motor_state_dict[motor_name].vel)

        motor_pos_arr = np.array(motor_pos, dtype=np.float32)
        motor_vel_arr = np.array(motor_vel, dtype=np.float32)

        joint_pos: List[float] = []
        joint_vel: List[float] = []
        for joint_name in joint_state_dict:
            joint_pos.append(joint_state_dict[joint_name].pos)
            joint_vel.append(joint_state_dict[joint_name].vel)

        joint_pos_arr = np.array(motor_pos, dtype=np.float32)
        joint_vel_arr = np.array(motor_vel, dtype=np.float32)

        if self.fixed_base:
            # torso_lin_vel = np.zeros(3, dtype=np.float32)
            torso_ang_vel = np.zeros(3, dtype=np.float32)
            torso_euler = np.zeros(3, dtype=np.float32)
        else:
            # lin_vel_global = np.array(
            #     self.data.body("torso").cvel[3:],  # type: ignore
            #     dtype=np.float32,
            #     copy=True,
            # )
            ang_vel_global = np.array(
                self.data.body("torso").cvel[:3],  # type: ignore
                dtype=np.float32,
                copy=True,
            )
            torso_quat = np.array(
                self.data.body("torso").xquat,  # type: ignore
                dtype=np.float32,
                copy=True,
            )
            if np.linalg.norm(torso_quat) == 0:  # type: ignore
                torso_quat = np.array([1, 0, 0, 0], dtype=np.float32)

            # torso_lin_vel = np.asarray(rotate_vec(lin_vel_global, quat_inv(torso_quat)))
            torso_ang_vel = np.asarray(rotate_vec(ang_vel_global, quat_inv(torso_quat)))

            torso_euler = np.asarray(quat2euler(torso_quat))
            torso_euler_delta = torso_euler - self.torso_euler_prev
            torso_euler_delta = (torso_euler_delta + np.pi) % (2 * np.pi) - np.pi
            torso_euler = self.torso_euler_prev + torso_euler_delta
            self.torso_euler_prev = np.asarray(torso_euler, dtype=np.float32)

        # Add sensor noise
        # obs.euler += np.random.normal(0, self.imu_euler_noise_std, size=obs.euler.shape)
        # obs.ang_vel += np.random.normal(
        #     0, self.imu_gyro_noise_std, size=obs.ang_vel.shape
        # )

        filtered_motor_vel = np.asarray(
            exponential_moving_average(0.1, motor_vel_arr, self.motor_vel_prev),
            dtype=np.float32,
        )
        self.motor_vel_prev = motor_vel_arr

        obs = Obs(
            time=time,
            motor_pos=motor_pos_arr,
            motor_vel=filtered_motor_vel,
            # lin_vel=torso_lin_vel,
            ang_vel=torso_ang_vel,
            euler=torso_euler,
            joint_pos=joint_pos_arr,
            joint_vel=joint_vel_arr,
        )
        return obs

    def get_mass(self) -> float:
        subtree_mass = float(self.model.body(0).subtreemass)  # type: ignore
        return subtree_mass

    def get_com(self) -> npt.NDArray[np.float32]:
        subtree_com = np.array(self.data.body(0).subtree_com, dtype=np.float32)  # type: ignore
        return subtree_com

    def set_motor_angles(
        self, motor_angles: Dict[str, float] | npt.NDArray[np.float32]
    ):
        if isinstance(motor_angles, dict):
            for name, ctrl in motor_angles.items():
                self.data.actuator(name).ctrl = ctrl  # type: ignore
        else:
            self.data.ctrl = motor_angles  # type: ignore

    def set_joint_angles(
        self, joint_angles: Dict[str, float] | npt.NDArray[np.float32]
    ):
        if isinstance(joint_angles, np.ndarray):
            self.data.qpos = joint_angles.copy()  # type: ignore
        else:
            for name in joint_angles:
                self.data.joint(name).qpos = joint_angles[name]  # type: ignore

    def set_joint_dynamics(self, joint_dyn: Dict[str, Dict[str, float]]):
        for joint_name, dyn in joint_dyn.items():
            for key, value in dyn.items():
                setattr(self.model.joint(joint_name), key, value)  # type: ignore

    def forward(self):
        mujoco.mj_forward(self.model, self.data)  # type: ignore

    def step(self):
        for _ in range(self.n_frames):
            mujoco.mj_step(self.model, self.data)  # type: ignore

        if self.visualizer is not None:
            self.visualizer.visualize(self.data)  # type: ignore

    def rollout(
        self,
        motor_angles_list: List[Dict[str, float]]
        | List[npt.NDArray[np.float32]]
        | npt.NDArray[np.float32],
    ):
        n_state = mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS)  # type: ignore
        initial_state = np.empty(n_state, dtype=np.float64)  # type: ignore
        mujoco.mj_getState(  # type: ignore
            self.model,  # type: ignore
            self.data,  # type: ignore
            initial_state,
            mujoco.mjtState.mjSTATE_FULLPHYSICS,  # type: ignore
        )

        control = np.zeros(
            (len(motor_angles_list) * self.n_frames, int(self.model.nu)),  # type: ignore
            dtype=np.float64,
        )
        for i, motor_angles in enumerate(motor_angles_list):
            if isinstance(motor_angles, np.ndarray):
                control[self.n_frames * i : self.n_frames * (i + 1)] = motor_angles
            else:
                for name, angle in motor_angles.items():
                    control[
                        self.n_frames * i : self.n_frames * (i + 1),
                        self.model.actuator(name).id,  # type: ignore
                    ] = angle

        state_traj, _ = mujoco.rollout.rollout(  # type: ignore
            self.model,  # type: ignore
            self.data,  # type: ignore
            initial_state,
            control,
        )
        state_traj = np.array(state_traj, dtype=np.float32).squeeze()[:: self.n_frames]
        # mjSTATE_TIME ｜ mjSTATE_QPOS | mjSTATE_QVEL | mjSTATE_ACT

        # joints_config = self.robot.config["joints"]
        motor_state_list: List[Dict[str, JointState]] = []
        for state in state_traj:
            motor_state: Dict[str, JointState] = {}
            for motor_name in self.robot.motor_ordering:
                # transmission = joints_config[motor_name]["transmission"]
                # if transmission == "gears":
                #     joint_name = self.robot.motor_to_joint_name[motor_name]
                #     motor_pos = (
                #         state[1 + self.model.joint(joint_name).id]  # type: ignore
                #         / joints_config[motor_name]["gear_ratio"]
                #     )
                # else:
                motor_pos = state[1 + self.model.joint(motor_name).id]  # type: ignore
                motor_state[motor_name] = JointState(time=state[0], pos=motor_pos)

            motor_state_list.append(motor_state)

        return motor_state_list

    def save_recording(
        self,
        exp_folder_path: str,
        dt: float,
        render_every: int,
        name: str = "mujoco.mp4",
    ):
        if isinstance(self.visualizer, MuJoCoRenderer):
            self.visualizer.save_recording(exp_folder_path, dt, render_every, name)

    def close(self):
        if self.visualizer is not None:
            self.visualizer.close()
