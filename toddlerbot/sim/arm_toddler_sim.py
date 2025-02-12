import time
from typing import Any, Dict, List

import mujoco
import numpy as np

from toddlerbot.actuation import JointState
from toddlerbot.sim import Obs
from toddlerbot.sim.arm import BaseArm
from toddlerbot.sim.mujoco_control import JointController, MotorController
from toddlerbot.sim.mujoco_sim import MuJoCoSim
from toddlerbot.sim.robot import Robot
from toddlerbot.utils.file_utils import find_arm_toddler_file_path


class ArmToddlerSim(MuJoCoSim):
    def __init__(
        self,
        robot: Robot,
        arm: BaseArm,
        sensor_names: List[str],  # DISCUSS
        n_frames: int = 20,
        dt: float = 0.001,
        fixed_base: bool = False,
        xml_path: str = "",
        xml_str: str = "",
        assets: Any = None,
        vis_type: str = "",
    ):
        if len(xml_path) == 0 and len(xml_str) == 0:
            suffix = "_fixed_scene.xml" if fixed_base else "_hang_scene.xml"
            xml_path = find_arm_toddler_file_path(arm.name, robot.name, suffix=suffix)
        super(ArmToddlerSim, self).__init__(
            robot, n_frames, dt, fixed_base, 0.0, xml_path, xml_str, assets, vis_type
        )
        self.arm = arm
        # TODO: should I add set function and relaod different attributes like `self.motor_vel_prev` or just reset their values?
        # TODO: what's the best practice for unused attributes like self.controller?
        self.motor_vel_prev = np.zeros(self.model.nu - arm.arm_dofs, dtype=np.float32)
        self.controller = MotorController(robot)
        # if not self.fixed_base:
        #     self.q_start_idx *= 2
        #     self.qd_start_idx *= 2
        #     self.motor_indices -= 1
        self.sensor_names = sensor_names
        # import ipdb; ipdb.set_trace()
        self.arm_controller = JointController()
        self.target_motor_angles = np.zeros(
            self.model.nu - arm.arm_dofs, dtype=np.float32
        )
        self.target_arm_joint_angles = np.zeros(arm.arm_dofs, dtype=np.float32)

    def get_arm_joint_state(self) -> Dict[str, JointState]:
        joint_state_dict: Dict[str, JointState] = {}
        for name in self.arm.joint_ordering:
            joint_state_dict[name] = JointState(
                time=time.time(),
                pos=self.data.joint(name).qpos.item(),
                vel=self.data.joint(name).qvel.item(),
            )
        return joint_state_dict

    def set_target_arm_joint_angles(self, target_arm_joint_angles: np.ndarray):
        self.target_arm_joint_angles = target_arm_joint_angles

    def get_sensor_data(self) -> Dict[str, float | np.ndarray]:  # DISCUSS
        sensor_data = {}
        for name in self.sensor_names:
            sensor_data[name] = self.data.sensor(name).data
        return sensor_data

    def get_observation(self) -> Obs:
        obs = super(ArmToddlerSim, self).get_observation()
        arm_joint_state_dict = self.get_arm_joint_state()
        arm_joint_pos: List[float] = []
        arm_joint_vel: List[float] = []
        for name in self.arm.joint_ordering:
            arm_joint_pos.append(arm_joint_state_dict[name].pos)
            arm_joint_vel.append(arm_joint_state_dict[name].vel)
        obs.arm_joint_pos = np.array(arm_joint_pos, dtype=np.float32)
        obs.arm_joint_vel = np.array(arm_joint_vel, dtype=np.float32)
        obs.ee_force = self.data.sensor("ee_force").data
        obs.ee_torque = self.data.sensor("ee_torque").data
        # self.data.mocap_pos[0] = obs.torso_pos
        obs.mocap_pos = self.data.mocap_pos[0]
        obs.mocap_quat = self.data.mocap_quat[0]
        return obs

    def get_mass(self) -> float:
        # return the mass of the toddlerbot
        subtree_mass = float(self.model.body(self.arm.arm_nbodies).subtreemass)
        return subtree_mass

    def step(self, action=None):
        for _ in range(self.n_frames):
            # import ipdb; ipdb.set_trace()
            self.data.ctrl[: self.arm.arm_dofs] = self.arm_controller.step(
                self.target_arm_joint_angles
            )  # DISCUSS
            # import ipdb; ipdb.set_trace()
            self.data.ctrl[self.arm.arm_dofs :] = self.controller.step(
                self.data.qpos[self.q_start_idx + self.motor_indices],
                self.data.qvel[self.qd_start_idx + self.motor_indices],
                self.target_motor_angles,
            )
            mujoco.mj_step(self.model, self.data)
        if self.visualizer is not None:
            self.visualizer.visualize(self.data)
