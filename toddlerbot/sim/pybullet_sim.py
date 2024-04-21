import numpy as np
import pybullet as p
import pybullet_data

from toddlerbot.sim import BaseSim
from toddlerbot.sim.robot import HumanoidRobot
from toddlerbot.utils.constants import GRAVITY
from toddlerbot.utils.file_utils import find_description_path
from toddlerbot.utils.math_utils import quatxyzw2mat
from toddlerbot.utils.misc_utils import precise_sleep


class PyBulletSim(BaseSim):
    """Class to set up and run a PyBullet simulation with a humanoid robot."""

    def __init__(self, robot=None, fixed=False):
        """
        Set up the PyBullet simulation environment.
        """
        super().__init__()
        self.name = "pybullet"
        self.robot = robot

        p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -GRAVITY)
        p.setTimeStep(0.001)
        p.loadURDF("plane.urdf")

        urdf_path = find_description_path(robot.name)
        robot.id = p.loadURDF(urdf_path, useFixedBase=fixed)
        self.name2idx = self.get_name2idx()
        if not fixed:
            self.put_robot_on_ground()

    def put_robot_on_ground(self, z_offset: float = 0.01):
        """
        Adjust the robot's position to place its lowest point above the ground.

        Args:
            robot (HumanoidRobot): The humanoid robot.
            z_offset (float): The offset from the ground to place the robot.
        """
        num_joints = p.getNumJoints(self.robot.id)
        lowest_z = float("inf")

        for i in range(-1, num_joints):  # -1 for the base link
            if i == -1:
                link_pos, _ = p.getBasePositionAndOrientation(self.robot.id)
            else:
                link_state = p.getLinkState(
                    self.robot.id, i, computeForwardKinematics=True
                )
                link_pos = link_state[0]

            lowest_z = min(lowest_z, link_pos[2])

        # Calculate new base position
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot.id)
        new_base_pos = [base_pos[0], base_pos[1], base_pos[2] - lowest_z + z_offset]
        p.resetBasePositionAndOrientation(self.robot.id, new_base_pos, base_ori)

    def get_name2idx(self):
        name2idx = {}
        for i in range(p.getNumJoints(self.robot.id)):
            jointInfo = p.getJointInfo(self.robot.id, i)
            name = jointInfo[1].decode("utf-8")
            name2idx[name] = i

        return name2idx

    def get_link_pos(self, link_name: str):
        for idx in range(p.getNumJoints(self.robot.id)):
            link_name_curr = p.getJointInfo(self.robot.id, idx)[12].decode("UTF-8")
            if link_name_curr == link_name:
                link_pos = p.getLinkState(
                    self.robot.id, idx, computeForwardKinematics=True
                )[0]
                return np.array(link_pos)

        raise ValueError(f"Link name {link_name} not found.")

    def get_torso_pose(self):
        torso_pos, torso_ori = p.getBasePositionAndOrientation(self.robot.id)
        return torso_pos, quatxyzw2mat(torso_ori)

    def get_joint_state(selft):
        pass

    def get_zmp(self):
        pass

    def set_joint_angles(self, joint_angles):
        for name, angle in joint_angles.items():
            p.setJointMotorControl2(
                self.robot.id,
                self.name2idx[name],
                p.POSITION_CONTROL,
                angle,
                # positionGain=robot.config.motor_params[name].kp * 1e-4,
                # velocityGain=robot.config.motor_params[name].kv * 1e-1,
            )

    def simulate(
        self,
        step_func=None,
        step_params=None,
        vis_flags=[],
        sleep_time: float = 0.0,
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
                    precise_sleep(sleep_time)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        p.disconnect()


if __name__ == "__main__":
    robot = HumanoidRobot("robotis_op3")
    sim = PyBulletSim(robot)
    sim.simulate()
