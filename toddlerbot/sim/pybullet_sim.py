import time

import numpy as np
import pybullet as p
import pybullet_data

from toddlerbot.sim import *
from toddlerbot.utils.constants import GRAVITY, TIMESTEP
from toddlerbot.utils.file_utils import find_description_path


class PyBulletSim(BaseSim):
    """Class to set up and run a PyBullet simulation with a humanoid robot."""

    def __init__(self, robot: Optional[HumanoidRobot] = None, fixed: bool = False):
        """
        Set up the PyBullet simulation environment.
        Initializes the PyBullet environment in GUI mode and sets the gravity and timestep.
        """
        super().__init__(robot)

        p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -GRAVITY)
        p.setTimeStep(TIMESTEP)
        p.loadURDF("plane.urdf")

        if robot is not None:
            urdf_path = find_description_path(robot.name)
            robot.id = p.loadURDF(urdf_path, useFixedBase=fixed)
            robot.joints_info = self.get_joints_info(robot)
            if not fixed:
                self.put_robot_on_ground(robot)

    def put_robot_on_ground(self, robot: HumanoidRobot, z_offset: float = 0.01):
        """
        Adjust the robot's position to place its lowest point at a specified offset above the ground.

        Args:
            robot (HumanoidRobot): The humanoid robot.
            z_offset (float): The offset from the ground to place the robot. Default is 0.01.
        """
        num_joints = p.getNumJoints(robot.id)
        lowest_z = float("inf")

        for i in range(-1, num_joints):  # -1 for the base link
            if i == -1:
                link_pos, _ = p.getBasePositionAndOrientation(robot.id)
            else:
                link_state = p.getLinkState(robot.id, i, computeForwardKinematics=True)
                link_pos = link_state[0]

            lowest_z = min(lowest_z, link_pos[2])

        # Calculate new base position
        base_pos, base_ori = p.getBasePositionAndOrientation(robot.id)
        new_base_pos = [base_pos[0], base_pos[1], base_pos[2] - lowest_z + z_offset]
        p.resetBasePositionAndOrientation(robot.id, new_base_pos, base_ori)

    def get_joints_info(self, robot: HumanoidRobot):
        joints_info = {}
        for k in range(p.getNumJoints(robot.id)):
            jointInfo = p.getJointInfo(robot.id, k)
            name = jointInfo[1].decode("utf-8")
            joints_info[name] = {
                "idx": k,
                "type": jointInfo[2],
                "lowerLimit": jointInfo[8],
                "upperLimit": jointInfo[9],
                "active": name in robot.config.act_params.keys(),
            }

        return joints_info

    def get_link_pos(self, robot: HumanoidRobot, link_name: str):
        for idx in range(p.getNumJoints(robot.id)):
            link_name_curr = p.getJointInfo(robot.id, idx)[12].decode("UTF-8")
            if link_name_curr == link_name:
                link_pos = p.getLinkState(robot.id, idx, computeForwardKinematics=True)[
                    0
                ]
                return np.array(link_pos)

        raise ValueError(f"Link name {link_name} not found.")

    def initialize_joint_angles(self, robot: HumanoidRobot):
        joint_angles = {}
        for name, info in robot.joints_info.items():
            if info["active"] and info["type"] != p.JOINT_FIXED:
                joint_angles[name] = p.getJointState(robot.id, info["idx"])[0]

        return joint_angles

    def get_zmp(self, robot: HumanoidRobot):
        pass

    def get_com(self, robot: HumanoidRobot):
        pass

    def set_joint_angles(self, robot: HumanoidRobot, joint_angles: Dict[str, float]):
        for joint_name, angle in joint_angles.items():
            joint_idx = robot.joints_info[joint_name]["idx"]
            p.setJointMotorControl2(robot.id, joint_idx, p.POSITION_CONTROL, angle)

    def simulate(
        self,
        step_func: Optional[Callable] = None,
        step_params: Optional[Tuple] = None,
        sleep_time: float = 0.0,
        vis_flags: Optional[List] = [],
    ):
        """
        Run the main simulation loop.
        """
        try:
            while p.isConnected():
                if step_func is not None:
                    if step_params is None:
                        step_func()
                    else:
                        step_params = step_func(*step_params)
                p.stepSimulation()
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            p.disconnect()


if __name__ == "__main__":
    robot = HumanoidRobot("robotis_op3")
    sim = PyBulletSim(robot)
    sim.simulate()
