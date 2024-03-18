import time

import mujoco
import mujoco.viewer
import numpy as np
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat, quat2mat

from toddlerbot.sim import *
from toddlerbot.utils.constants import GRAVITY, TIMESTEP
from toddlerbot.utils.file_utils import find_description_path


class MujoCoSim(BaseSim):
    def __init__(self, robot: Optional[HumanoidRobot] = None, fixed: bool = False):
        """Initialize the MuJoCo simulation environment."""
        super().__init__()
        self.name = "mujoco"

        self.model = None
        self.data = None

        if robot is not None:
            self.foot_size = robot.config.foot_size

            xml_path = find_description_path(robot.name, suffix="_scene.xml")
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
            if not fixed:
                self.put_robot_on_ground(robot)

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

    def get_link_pos(self, robot: HumanoidRobot, link_name: str):
        mujoco.mj_kinematics(self.model, self.data)
        link_pos = self.data.body(link_name).xpos
        return np.array(link_pos)

    def get_link_quat(self, robot: HumanoidRobot, link_name: str):
        mujoco.mj_kinematics(self.model, self.data)
        link_quat = self.data.body(link_name).xquat
        return np.array(link_quat)

    def get_torso_pose(self, robot: HumanoidRobot):
        mujoco.mj_kinematics(self.model, self.data)
        torso_pos = self.data.site("torso").xpos.copy()
        torso_mat = self.data.site("torso").xmat.copy().reshape(3, 3)
        return torso_pos, torso_mat

    def get_joint_state(self, robot: HumanoidRobot):
        joint_state_dict = {}
        time_curr = time.time()
        for name, info in robot.joints_info.items():
            if info["active"]:
                joint_state_dict[name] = JointState(
                    time=time_curr, pos=self.data.joint(name).qpos.item()
                )

        return joint_state_dict

    def set_joint_angles(self, robot: HumanoidRobot, joint_angles: Dict[str, float]):
        for name, angle in joint_angles.items():
            kp = robot.config.motor_params[name].kp
            kv = robot.config.motor_params[name].kv
            self.data.actuator(f"{name}_act").ctrl = (
                -kp * (self.data.joint(name).qpos - angle)
                - kv * self.data.joint(name).qvel
            )

    def simulate(
        self,
        step_func: Optional[Callable] = None,
        step_params: Optional[Tuple] = None,
        vis_flags: Optional[List] = [],
        sleep_time: float = 0.0,
    ):
        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if step_params is not None:
            sim_step_idx, path, foot_steps, com_traj, joint_angles = step_params

        def vis_foot_steps():
            i = viewer.user_scn.ngeom
            for foot_step in foot_steps:
                if foot_step.support_leg == "both":
                    continue

                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_LINEBOX,
                    size=[
                        self.foot_size[0] / 2,
                        self.foot_size[1] / 2,
                        self.foot_size[2] / 2,
                    ],
                    pos=np.array(
                        [
                            foot_step.position[0],
                            foot_step.position[1],
                            self.foot_size[2] / 2,
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
            for com_pos in com_traj:
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.001, 0.001, 0.001]),
                    pos=np.array([com_pos[0], com_pos[1], 0.005]),
                    mat=np.eye(3).flatten(),
                    rgba=[1, 0, 0, 1],
                )
                i += 1
            viewer.user_scn.ngeom = i

        def vis_path():
            i = viewer.user_scn.ngeom
            for j in range(len(path) - 1):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_LINE,
                    size=np.array([1, 1, 1]),
                    pos=np.array([0, 0, 0]),
                    mat=np.eye(3).flatten(),
                    rgba=[0, 0, 0, 1],
                )
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[i],
                    mujoco.mjtGeom.mjGEOM_LINE,
                    100,
                    np.array([*path[j], 0.0]),
                    np.array([*path[j + 1], 0.0]),
                )
                i += 1
            viewer.user_scn.ngeom = i

        def vis_torso():
            i = viewer.user_scn.ngeom
            torso_pos = self.data.site("torso").xpos
            torso_mat = self.data.site("torso").xmat
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=np.array([0.005, 0.005, 0.15]),
                pos=torso_pos,
                mat=torso_mat,
                rgba=[1, 0, 0, 1],
            )
            viewer.user_scn.ngeom = i + 1

        try:
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
                            if "path" in vis_flags:
                                vis_path()
                            if "torso" in vis_flags:
                                vis_torso()

                viewer.sync()

                time_until_next_step = sleep_time - (time.time() - step_start)
                # time_until_next_step = self.model.opt.timestep - (
                #     time.time() - step_start
                # )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            pass
        finally:
            viewer.close()

    def close(self):
        pass


if __name__ == "__main__":
    robot = HumanoidRobot("robotis_op3")
    sim = MujoCoSim()
    sim.simulate()
