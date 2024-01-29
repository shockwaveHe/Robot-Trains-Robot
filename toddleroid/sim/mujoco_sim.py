import time

import mujoco
import mujoco.viewer
import numpy as np

from toddleroid.sim.base_sim import *
from toddleroid.utils.constants import GRAVITY, TIMESTEP
from toddleroid.utils.file_utils import find_description_path


class MujoCoSim(AbstractSim):
    def __init__(self, robot: Optional[HumanoidRobot] = None):
        """Initialize the MuJoCo simulation environment."""
        self.model = None
        self.data = None

        if robot is not None:
            xml_path = find_description_path(robot.name, suffix="_scene.xml")
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
            robot.id = 0  # placeholder
            robot.joint_name2qidx = self.get_joint_name2qidx(robot)
            self.put_robot_on_ground(robot)

    def put_robot_on_ground(self, robot: HumanoidRobot, z_offset: float = 0.01):
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

        base_link_name = robot.config.canonical_name2link_name["base_link"]
        base_pos = self.data.body(base_link_name).xpos
        desired_z = base_pos[2] - lowest_z + z_offset
        if lowest_z < 0:
            raise ValueError(
                f"Robot is below the ground. Change the z value of {base_link_name} to be {desired_z}"
            )
        elif lowest_z > z_offset:
            raise ValueError(
                f"Robot is too high above the ground. Change the z value of {base_link_name} as {desired_z}"
            )

    def get_joint_name2qidx(self, robot: HumanoidRobot):
        joint_name2qidx = {}
        # 0 is an empty joint
        for i in range(1, self.model.njnt):
            joint_name2qidx[self.model.joint(i).name] = i - 1
        return joint_name2qidx

    def get_link_pos(self, robot: HumanoidRobot, link_name: str):
        mujoco.mj_kinematics(self.model, self.data)
        link_pos = self.data.body(link_name).xpos
        return np.array(link_pos)

    def get_named_zero_joint_angles(self, robot: HumanoidRobot):
        joint_angles = []
        joint_names = []
        # 0 is an empty joint
        for i in range(1, self.model.njnt):
            joint_angles.append(0)
            joint_names.append(self.model.joint(i).name)
        return joint_angles, joint_names

    def set_joint_angles(self, robot: HumanoidRobot, joint_angles: List[float]):
        for i in range(1, self.model.njnt):
            # self.data.joint(i).qpos = joint_angles[i - 1]
            self.data.actuator(i - 1).ctrl = joint_angles[i - 1]

    def simulate(
        self,
        step_func: Optional[Callable] = None,
        step_params: Optional[Tuple] = None,
        sleep_time: float = 0.0,
    ):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                step_start = time.time()

                mujoco.mj_step(self.model, self.data)

                with viewer.lock():
                    if step_func is not None:
                        if step_params is None:
                            step_func()
                        else:
                            step_params = step_func(*step_params)

                viewer.sync()

                time_until_next_step = sleep_time - (time.time() - step_start)
                # time_until_next_step = self.model.opt.timestep - (
                #     time.time() - step_start
                # )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == "__main__":
    robot = HumanoidRobot("robotis_op3")
    sim = MujoCoSim()
    sim.simulate()
