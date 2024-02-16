import time

import mujoco
import mujoco.viewer
import numpy as np
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat, quat2mat

from toddlerbot.sim.base_sim import *
from toddlerbot.utils.constants import GRAVITY, TIMESTEP
from toddlerbot.utils.file_utils import find_description_path


class MujoCoSim(AbstractSim):
    def __init__(self, robot: Optional[HumanoidRobot] = None, fixed: bool = False):
        """Initialize the MuJoCo simulation environment."""
        self.model = None
        self.data = None

        if robot is not None:
            xml_path = find_description_path(robot.name, suffix="_scene.xml")
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
            robot.id = 0  # placeholder
            robot.joints_info = self.get_joints_info(robot)
            if not fixed:
                self.put_robot_on_ground(robot)

            if "foot_size_x" in robot.config.offsets:
                self.foot_size_x = robot.config.offsets["foot_size_x"]

            if "foot_size_y" in robot.config.offsets:
                self.foot_size_y = robot.config.offsets["foot_size_y"]

            if "foot_size_z" in robot.config.offsets:
                self.foot_size_z = robot.config.offsets["foot_size_z"]

    def put_robot_on_ground(self, robot: HumanoidRobot, z_offset: float = 0.02):
        """
        Adjust the robot's position to place its lowest point at a specified offset above the ground.

        Args:
            robot (HumanoidRobot): The humanoid robot.
            z_offset (float): The offset from the ground to place the robot. Default is 0.01.
        """
        lowest_z = float("inf")

        mujoco.mj_kinematics(self.model, self.data)
        # Iterate through all body parts to find the lowest point
        for i in range(self.model.nbody):
            if self.data.body(i).name == "world":
                continue
            # To correpond to the PyBullet code, we use xipos instead of xpos
            body_pos = self.data.body(i).xipos
            lowest_z = min(lowest_z, body_pos[2])

        body_link_name = robot.config.canonical_name2link_name["body_link"]
        base_pos = self.data.body(body_link_name).xpos
        desired_z = base_pos[2] - lowest_z + z_offset
        if lowest_z < 0:
            raise ValueError(
                f"Robot is below the ground. Change the z value of {body_link_name} to be {desired_z}"
            )
        elif lowest_z > z_offset:
            raise ValueError(
                f"Robot is too high above the ground. Change the z value of {body_link_name} as {desired_z}"
            )

    def get_joints_info(self, robot: HumanoidRobot):
        joints_info = {}
        for i in range(1, self.model.njnt):
            name = self.model.joint(i).name
            joints_info[name] = {
                "idx": self.model.joint(i).id,
                "type": self.model.joint(i).type,
                "lowerLimit": self.model.joint(i).range[0],
                "upperLimit": self.model.joint(i).range[1],
                "active": name in robot.config.act_params.keys(),
            }

        return joints_info

    def get_link_pos(self, robot: HumanoidRobot, link_name: str):
        mujoco.mj_kinematics(self.model, self.data)
        link_pos = self.data.body(link_name).xpos
        return np.array(link_pos)

    def get_link_quat(self, robot: HumanoidRobot, link_name: str):
        mujoco.mj_kinematics(self.model, self.data)
        link_quat = self.data.body(link_name).xquat
        return np.array(link_quat)

    def get_link_relpose(
        self, robot: HumanoidRobot, link_name_1: str, link_name_2: str
    ):
        mujoco.mj_kinematics(self.model, self.data)
        # Get the position and orientation (quaternion) of each body
        pos1 = self.data.body(link_name_1).xpos
        quat1 = self.data.body(link_name_1).xquat
        pos2 = self.data.body(link_name_2).xpos
        quat2 = self.data.body(link_name_2).xquat

        print(f"pos1: {pos1}, quat1: {quat1}")
        print(f"pos2: {pos2}, quat2: {quat2}")

        # Compute the relative position (simple subtraction)
        rel_pos = pos2 - pos1

        # For orientation, compute the relative quaternion
        rel_mat = np.dot(np.linalg.inv(quat2mat(quat1)), quat2mat(quat2))
        rel_quat = mat2quat(rel_mat)

        return np.concatenate([rel_pos, rel_quat])

    def initialize_joint_angles(self, robot: HumanoidRobot):
        joint_angles = {}
        for name, info in robot.joints_info.items():
            if info["active"]:
                joint_angles[name] = self.data.joint(name).qpos.item()
        return joint_angles

    def get_com_state(self, robot: HumanoidRobot):
        # TODO: Replace this with an IMU sensor
        mujoco.mj_kinematics(self.model, self.data)
        body_link_name = robot.config.canonical_name2link_name["body_link"]
        com_pos = self.data.body(body_link_name).xipos[:2]
        com_vel = self.data.body(body_link_name).cvel[3:5]
        com_acc = self.data.body(body_link_name).cacc[3:5]
        return np.array([com_pos, com_vel, com_acc])

    def set_joint_angles(self, robot: HumanoidRobot, joint_angles: Dict[str, float]):
        for joint_name, angle in joint_angles.items():
            kp = robot.config.act_params[joint_name].kp
            kv = robot.config.act_params[joint_name].kv
            self.data.actuator(f"{joint_name}_act").ctrl = (
                -kp * (self.data.joint(joint_name).qpos - angle)
                - kv * self.data.joint(joint_name).qvel
            )

    def simulate(
        self,
        step_func: Optional[Callable] = None,
        step_params: Optional[Dict] = None,
        sleep_time: float = 0.0,
        vis_flags: Optional[List] = [],
    ):
        viewer = mujoco.viewer.launch_passive(self.model, self.data)

        def vis_foot_steps():
            i = viewer.user_scn.ngeom
            # step_params: sim_step_idx, foot_steps, com_traj, joint_angles
            for foot_step in step_params[1]:
                if foot_step.support_leg == "both":
                    continue

                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_LINEBOX,
                    size=[
                        self.foot_size_x / 2,
                        self.foot_size_y / 2,
                        self.foot_size_z / 2,
                    ],
                    pos=np.array(
                        [
                            foot_step.position[0],
                            foot_step.position[1],
                            self.foot_size_z / 2,
                        ]
                    ),
                    mat=euler2mat(0, 0, foot_step.position[2]).flatten(),
                    rgba=(
                        [0, 0, 1, 1]
                        if foot_step.support_leg == "left"
                        else [0, 1, 0, 1]
                    ),
                )
                i += 1
            viewer.user_scn.ngeom = i

        def vis_com_traj():
            i = viewer.user_scn.ngeom
            # step_params: sim_step_idx, foot_steps, com_traj, joint_angles
            for com_pos in step_params[2]:
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=np.array([0.01, 0.0075, 0.01]),
                    pos=np.array([com_pos[0], com_pos[1], 0.5]),
                    mat=np.eye(3).flatten(),
                    rgba=[1, 0, 0, 1],
                )
                i += 1
            viewer.user_scn.ngeom = i

        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(self.model, self.data)

            with viewer.lock():
                if step_func is not None:
                    if step_params is None:
                        step_func()
                    else:
                        step_params = step_func(*step_params)

                viewer.user_scn.ngeom = 0
                if "foot_steps" in vis_flags:
                    vis_foot_steps()
                if "com_traj" in vis_flags:
                    vis_com_traj()

            viewer.sync()

            time_until_next_step = sleep_time - (time.time() - step_start)
            # time_until_next_step = self.model.opt.timestep - (
            #     time.time() - step_start
            # )
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        viewer.close()


if __name__ == "__main__":
    robot = HumanoidRobot("robotis_op3")
    sim = MujoCoSim()
    sim.simulate()
