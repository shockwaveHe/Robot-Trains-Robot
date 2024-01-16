import time

import pybullet as p
import pybullet_data

from toddleroid.sim.pybullet.robot import HumanoidRobot
from toddleroid.utils.constants import GRAVITY, TIMESTEP


class PyBulletSim:
    def __init__(self):
        self.setup()

    def setup(self):
        """
        Set up the PyBullet simulation environment.
        Initializes the PyBullet environment in GUI mode and sets the gravity for the simulation.
        """
        p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -GRAVITY)
        p.setTimeStep(TIMESTEP)
        p.loadURDF("plane.urdf")

    def load_robot(self, robot: HumanoidRobot):
        """
        Load the robot URDF and set its initial position.
        """
        urdf_path = robot.get_urdf_path()
        robot.id = p.loadURDF(urdf_path)
        # self.put_robot_on_ground(robot.id, z_offset=0.01)

    def put_robot_on_ground(self, robot_id: int, z_offset: float):
        """
        Adjust the robot's position to place its lowest point at a specified offset above the ground.
        """
        num_joints = p.getNumJoints(robot_id)
        lowest_z = float("inf")

        for i in range(-1, num_joints):  # -1 for the base link
            if i == -1:
                link_pos, _ = p.getBasePositionAndOrientation(robot_id)
            else:
                link_state = p.getLinkState(robot_id, i, computeForwardKinematics=True)
                link_pos = link_state[0]

            lowest_z = min(lowest_z, link_pos[2])

        # Calculate new base position
        base_pos, base_ori = p.getBasePositionAndOrientation(robot_id)
        new_base_pos = [base_pos[0], base_pos[1], base_pos[2] - lowest_z + z_offset]

        p.resetBasePositionAndOrientation(robot_id, new_base_pos, base_ori)

    def run(self):
        """
        Run the main simulation loop.
        """
        try:
            while p.isConnected():
                p.stepSimulation()
                # time.sleep(TIMESTEP)
        finally:
            p.disconnect()


if __name__ == "__main__":
    sim = PyBulletSim()
    robot = HumanoidRobot("Robotis_OP3")
    sim.load_robot(robot)
    sim.run()
