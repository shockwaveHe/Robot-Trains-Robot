import time
from typing import Any, Dict, List

import mujoco  # type: ignore
import mujoco.rollout  # type: ignore
import mujoco.viewer  # type: ignore
import numpy as np
import numpy.typing as npt

from toddlerbot.actuation import JointState
from toddlerbot.sim import BaseSim, state_to_obs
from toddlerbot.sim.mujoco_utils import MuJoCoController, MuJoCoRenderer, MuJoCoViewer
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_robot_file_path
from toddlerbot.utils.math_utils import quat2euler


class MuJoCoSim(BaseSim):
    def __init__(
        self,
        robot: Robot,
        fixed_base: bool = False,
        xml_path: str = "",
        xml_str: str = "",
        assets: Any = None,
        vis_type: str = "",
        device: str = "cpu",
    ):
        """Initialize the MuJoCo simulation environment."""
        super().__init__()

        self.name = "mujoco"
        self.robot = robot
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

        self.default_qpos = np.array(model.keyframe("home").qpos, dtype=np.float32)  # type:ignore
        self.default_action = np.array(model.keyframe("home").ctrl, dtype=np.float32)  # type:ignore

        self.model.opt.timestep = self.dt  # type: ignore
        self.controller = MuJoCoController()
        mujoco.set_mjcb_control(self.controller.process_commands)  # type: ignore

        self.initialize()

        self.visualizer = None
        if vis_type == "render":
            self.visualizer = MuJoCoRenderer(self.model, self.data)  # type: ignore
        elif vis_type == "view":
            self.visualizer = MuJoCoViewer(self.model, self.data)  # type: ignore

    def initialize(self):
        self.data.qpos = self.default_qpos.copy()  # type: ignore
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

        obs = state_to_obs(motor_state_dict, joint_state_dict)

        obs.imu_euler = np.asarray(
            quat2euler(
                np.array(
                    self.data.sensor("orientation").data,  # type: ignore
                    copy=True,
                )
            )
        )
        obs.imu_ang_vel = np.array(
            self.data.sensor("angular_velocity").data,  # type: ignore
            copy=True,
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
        self.controller.add_command(motor_angles)

    def set_joint_angles(
        self, joint_angles: Dict[str, float] | npt.NDArray[np.float32]
    ):
        if isinstance(joint_angles, np.ndarray):
            self.data.qpos = joint_angles.copy()  # type: ignore
        else:
            for name in joint_angles:
                self.data.joint(name).qpos = joint_angles[name]  # type: ignore

    def forward(self):
        mujoco.mj_forward(self.model, self.data)  # type: ignore

    def step(self):
        mujoco.mj_step(self.model, self.data)  # type: ignore

        if self.visualizer is not None:
            self.visualizer.visualize(self.model, self.data)  # type: ignore

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
