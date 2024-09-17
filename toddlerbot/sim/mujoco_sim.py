import time
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt

import mujoco
import mujoco.rollout
import mujoco.viewer
from toddlerbot.actuation import JointState
from toddlerbot.sim import BaseSim, Obs
from toddlerbot.sim.mujoco_utils import (
    MotorController,
    MuJoCoRenderer,
    MuJoCoViewer,
)
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.math_utils import quat2euler, quat_inv, rotate_vec

"""
vis_type: ["render", "view"]
"""


class MuJoCoSim(BaseSim):
    def __init__(
        self,
        robot: Robot,
        n_frames: int = 10,
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

        # Assume imu is the first site
        self.torso_euler_prev = np.zeros(3, dtype=np.float32)
        self.motor_vel_prev = np.zeros(self.model.nu, dtype=np.float32)

        # if fixed_base:
        #     self.controller = PositionController()
        # else:
        motor_indices = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot.motor_ordering
            ]
        )
        self.controller = MotorController(
            motor_indices,
            np.array(self.robot.get_joint_attrs("type", "dynamixel", "kp_real")),
            np.array(self.robot.get_joint_attrs("type", "dynamixel", "kd_real")),
            np.array(self.robot.get_joint_attrs("type", "dynamixel", "tau_max")),
            np.array(self.robot.get_joint_attrs("type", "dynamixel", "q_dot_tau_max")),
            np.array(self.robot.get_joint_attrs("type", "dynamixel", "q_dot_max")),
        )

        self.target_motor_angles = np.zeros(self.model.nu, dtype=np.float32)

        self.visualizer: MuJoCoRenderer | MuJoCoViewer | None = None
        if vis_type == "render":
            self.visualizer = MuJoCoRenderer(self.model)
        elif vis_type == "view":
            self.visualizer = MuJoCoViewer(self.model, self.data)

    def load_keyframe(self):
        default_qpos = np.array(self.model.keyframe("home").qpos, dtype=np.float32)
        self.data.qpos = default_qpos.copy()
        self.data.qvel = np.zeros(self.model.nv, dtype=np.float32)
        self.forward()

    def get_motor_state(self) -> Dict[str, JointState]:
        motor_state_dict: Dict[str, JointState] = {}
        for name in self.robot.motor_ordering:
            motor_state_dict[name] = JointState(
                time=time.time(),
                pos=self.data.joint(name).qpos.item(),
                vel=self.data.joint(name).qvel.item(),
                tor=self.data.joint(name).qfrc_actuator.item(),
            )

        return motor_state_dict

    def get_joint_state(self) -> Dict[str, JointState]:
        joint_state_dict: Dict[str, JointState] = {}
        for name in self.robot.joint_ordering:
            joint_state_dict[name] = JointState(
                time=time.time(),
                pos=self.data.joint(name).qpos.item(),
                vel=self.data.joint(name).qvel.item(),
            )

        return joint_state_dict

    def get_observation(self) -> Obs:
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
            # torso_lin_vel = np.zeros(3, dtype=np.float32)
            torso_ang_vel = np.zeros(3, dtype=np.float32)
            torso_euler = np.zeros(3, dtype=np.float32)
        else:
            # lin_vel_global = np.array(
            #     self.data.body("torso").cvel[3:],
            #     dtype=np.float32,
            #     copy=True,
            # )
            ang_vel_global = np.array(
                self.data.body("torso").cvel[:3],
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

            # torso_lin_vel = np.asarray(rotate_vec(lin_vel_global, quat_inv(torso_quat)))
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
            # lin_vel=torso_lin_vel,
            ang_vel=torso_ang_vel,
            euler=torso_euler,
            joint_pos=joint_pos_arr,
            joint_vel=joint_vel_arr,
        )
        return obs

    def get_mass(self) -> float:
        subtree_mass = float(self.model.body(0).subtreemass)
        return subtree_mass

    def get_com(self) -> npt.NDArray[np.float32]:
        subtree_com = np.array(self.data.body(0).subtree_com, dtype=np.float32)
        return subtree_com

    def set_motor_kps(self, motor_kps: Dict[str, float]):
        for name, kp in motor_kps.items():
            if isinstance(self.controller, MotorController):
                idx = self.model.actuator(name).id
                self.controller.kp[idx] = kp / 128
            else:
                self.model.actuator(name).gainprm[0] = kp / 128
                self.model.actuator(name).biasprm[1] = -kp / 128

    def set_motor_angles(
        self, motor_angles: Dict[str, float] | npt.NDArray[np.float32]
    ):
        self.target_motor_angles = motor_angles

    def set_joint_angles(self, joint_angles: Dict[str, float]):
        for name in joint_angles:
            self.data.joint(name).qpos = joint_angles[name]

    def set_joint_dynamics(self, joint_dyn: Dict[str, Dict[str, float]]):
        for joint_name, dyn in joint_dyn.items():
            for key, value in dyn.items():
                setattr(self.model.joint(joint_name), key, value)

    def forward(self):
        for _ in range(self.n_frames):
            mujoco.mj_forward(self.model, self.data)

        if self.visualizer is not None:
            self.visualizer.visualize(self.data)

    def step(self):
        for _ in range(self.n_frames):
            self.data.ctrl = self.controller.step(
                self.model, self.data, self.target_motor_angles
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
        if isinstance(self.visualizer, MuJoCoRenderer):
            self.visualizer.save_recording(exp_folder_path, dt, render_every, name)

    def close(self):
        if self.visualizer is not None:
            self.visualizer.close()
